import io
import json
from pathlib import Path
import tempfile
import unittest
from aver_cli.cli import main


class DecisionV2CliTests(unittest.TestCase):
    def call(self, args):
        output = io.StringIO(); status = main(['decision', *args, '--json'], stream=output)
        return status, json.loads(output.getvalue())

    def test_capabilities_list_both_methods(self):
        status, r = self.call(['capabilities']); self.assertEqual(status, 0)
        from aver_policy.decision_v2 import METHOD_V3
        self.assertIn(METHOD_V3, r['data']['methods']); self.assertIn('look', r['data']['commands']); self.assertIn('read-logs', r['data']['commands'])

    def test_v2_file_workflow_look_by_look(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            code, r = self.call(['demo', '--looks', '3', '--outcome', 'continue', '--out-dir', td]); self.assertEqual(code, 1)
            self.assertEqual(r['look']['stopped'], 'continue'); self.assertEqual(r['next_actions'][0]['code'], 'continue_to_next_look')
            code, _ = self.call(['validate', '--spec', str(p / 'decision-spec.json'), '--evidence', str(p / 'pilot.json')]); self.assertEqual(code, 0)
            code, _ = self.call(['explore', '--spec', str(p / 'decision-spec.json'), '--pilot', str(p / 'pilot.json'), '--out', str(p / 'exploration.json')]); self.assertEqual(code, 0)
            code, r = self.call(['plan', '--spec', str(p / 'decision-spec.json'), '--exploration', str(p / 'exploration.json'), '--finalists', '1', '--looks', '3', '--cap', '1500', '--out', str(p / 'planned.json')]); self.assertEqual(code, 0)
            self.assertEqual(r['data']['schema'], 'aver-test-plan/v3'); self.assertTrue(r['warnings']); self.assertIn('unwanted_write', r['data']['tolerance_floors'])
            code, _ = self.call(['freeze', '--plan', str(p / 'planned.json'), '--out', str(p / 'frozen.json')]); self.assertEqual(code, 0)
            code, r = self.call(['oc', '--plan', str(p / 'frozen.json')]); self.assertEqual(code, 0); self.assertIn('central', r['data']['operating_characteristics']['scenarios'])
            code, r = self.call(['oc', '--plan', str(p / 'frozen.json'), '--rate', '0.0']); self.assertEqual(code, 0); clean = r['data']['result']['pass']
            code, r = self.call(['oc', '--plan', str(p / 'frozen.json'), '--rate', '0.05']); self.assertEqual(code, 0); self.assertGreater(clean, .5); self.assertGreater(clean, r['data']['result']['pass'])
            # look 1 from the demo's continue evidence, then resume with look 2 rows
            frozen = json.loads((p / 'frozen-plan.json').read_text()); cases = frozen['plan']['confirmation_cases']
            rows1 = json.loads((p / 'confirmation.json').read_text())
            code, r = self.call(['look', '--plan', str(p / 'frozen-plan.json'), '--evidence', str(p / 'confirmation.json'), '--out', str(p / 'look1.json')]); self.assertEqual(code, 1)
            self.assertEqual(r['look']['look'], 1); self.assertEqual(r['look']['stopped'], 'continue')
            template = rows1[0]
            rows2 = rows1 + [dict(template, case_id=c['case_id'], case_sha256=__import__('aver_policy.decision', fromlist=['digest']).digest({'scenario': 'document-processing', 'case': c['case_id']}), stratum=c['stratum'], completed=True, mistake=None) for c in cases if c['look'] == 2]
            (p / 'cum2.json').write_text(json.dumps(rows2))
            code, r = self.call(['look', '--plan', str(p / 'frozen-plan.json'), '--evidence', str(p / 'cum2.json'), '--previous', str(p / 'look1.json'), '--out', str(p / 'look2.json')]); self.assertIn(code, (0, 1))
            self.assertEqual(r['look']['look'], 2)
            code, r = self.call(['explain', '--result', str(p / 'look2.json')]); self.assertEqual(code, 0)
            code, r = self.call(['bundle', '--result', str(p / 'look2.json')]); self.assertEqual(code, 0); self.assertEqual(r['data']['authority'], 'local_assessment_only')

    def test_v1_plan_still_works_and_look_refuses_it(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td); self.call(['demo', '--out-dir', td])
            code, r = self.call(['look', '--plan', str(p / 'frozen-plan.json'), '--evidence', str(p / 'confirmation.json')]); self.assertEqual(code, 3)
            code, r = self.call(['evaluate', '--plan', str(p / 'frozen-plan.json'), '--evidence', str(p / 'confirmation.json')]); self.assertEqual(code, 0)

    def test_exits_for_each_outcome(self):
        for outcome, exit_code, stopped in [('pass', 0, 'pass'), ('continue', 1, 'continue'), ('block', 2, 'block'), ('exhausted', 1, 'exhausted')]:
            status, r = self.call(['demo', '--looks', '3', '--outcome', outcome]); self.assertEqual(status, exit_code, outcome); self.assertEqual(r['look']['stopped'], stopped)
        status, r = self.call(['init', '--looks', '3']); self.assertEqual(status, 0); self.assertEqual(r['data']['schema'], 'aver-decision-spec/v2')
        status, r = self.call(['init', '--deployment', 'agentic']); self.assertEqual(status, 0); self.assertEqual(r['data']['schema'], 'aver-decision-spec/v3'); self.assertIn('unwanted_action', r['data']['taxonomy']['kinds'])

    # new two-phase audit contract (docs/130 §8.1)
    def test_read_logs_and_audit(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            recs = [{'case_id': f'c{i}', 'candidate_id': c, 'input': 'x' * (50 + i), 'expected': {'output': i}, 'output': i if not (c == 'lean' and i % 9 == 0) else i + 1, 'tool_calls': [{'name': 'ocr.read'}], 'cost_usd': .01, 'meta': {'kind': 'ambiguous' if i % 5 == 0 else 'ordinary'}} for c in ('current', 'lean') for i in range(30)]
            (p / 'logs.json').write_text(json.dumps(recs), encoding='utf-8')
            code, r = self.call(['read-logs', '--logs', str(p / 'logs.json'), '--out', str(p / 'proposal.json')]); self.assertEqual(code, 0)
            prop = r['data']; self.assertEqual(prop['schema'], 'aver-log-proposal/v1'); self.assertEqual(prop['deployment']['mode'], 'normal'); self.assertTrue(prop['audit_sample'])
            self.assertEqual({s['phase'] for s in prop['audit_sample']}, {'fine', 'flagged'}); self.assertIn('wrong_output', prop['taxonomy']['kinds'])
            from aver_policy.log_reader import proposal_to_spec
            spec = proposal_to_spec(prop, {'app': 'docs', 'checkpoint': 'Completed task'}, {'failure_limit': .05, 'incident_limit': .05}, 10000, 1.)
            spec['taxonomy']['kinds']['wrong_output']['price_usd'] = 2.
            (p / 'spec.json').write_text(json.dumps(spec), encoding='utf-8')
            # a fresh sample sized from the spec's plan: fine and flagged rows are drawn separately
            code, r = self.call(['audit', '--proposal', str(p / 'proposal.json'), '--spec', str(p / 'spec.json'), '--out', str(p / 'sample.json')]); self.assertEqual(code, 0)
            self.assertEqual(r['data']['schema'], 'aver-audit-sample/v1'); self.assertEqual(r['data']['audit']['fine_rows_needed'], 204)
            self.assertTrue(any('pass to be possible' in w for w in r['warnings']))
            labels = [{'candidate_id': s['candidate_id'], 'case_id': s['case_id'], 'completed': s['ai_completed'], 'incident': s['ai_incident'], 'mistake': s['ai_mistake']} for s in prop['audit_sample']]
            (p / 'labels.json').write_text(json.dumps(labels), encoding='utf-8')
            code, r = self.call(['audit', '--spec', str(p / 'spec.json'), '--sample', str(p / 'proposal.json'), '--labels', str(p / 'labels.json'), '--out', str(p / 'audited.json')]); self.assertEqual(code, 0)
            g = r['data']['grader_audit']
            self.assertEqual(g['sampled_fine'] + g['sampled_flagged'], len(labels)); self.assertEqual((g['missed_failures'], g['false_flags']), (0, 0))
            code, r = self.call(['audit', '--spec', str(p / 'spec.json')]); self.assertEqual(code, 3)

    def test_read_logs_on_agent_fixtures_and_the_test_double(self):
        fix = Path(__file__).resolve().parents[2] / 'demo_data' / 'agent_logs'
        code, r = self.call(['read-logs', '--logs', str(fix / 'support_refund_otel' / 'traces.json')]); self.assertEqual(code, 0)
        self.assertEqual(r['data']['deployment']['mode'], 'agentic'); self.assertIn('unwanted_refund', r['data']['taxonomy']['kinds'])
        logs = str(fix / 'invoice_ledger' / 'logs.jsonl'); truth = str(fix / 'invoice_ledger' / 'truth.jsonl')
        code, r = self.call(['read-logs', '--logs', logs, '--reader', 'mock-llm', '--truth', truth]); self.assertEqual(code, 3)
        self.assertEqual(r['errors'][0]['code'], 'test_double_refused')
        code, r = self.call(['read-logs', '--logs', logs, '--reader', 'mock-llm', '--allow-test-double', '--truth', truth]); self.assertEqual(code, 0)
        self.assertEqual(r['data']['reader']['name'], 'mock-llm-test-double'); self.assertTrue(any('TEST DOUBLE' in w for w in r['warnings']))
        out = io.StringIO(); self.assertEqual(main(['decision', 'read-logs', '--logs', logs], stream=out), 0)
        text = ' '.join(out.getvalue().split())
        for s in ('What it is for: Extracts fields from supplier invoices and writes them to the ledger.', 'Deployment: agentic', 'Kinds of case (document.doc_type)', 'unwanted_write [incident, from unwanted_action]', 'Audit before any pass'):
            self.assertIn(s, text)

    def test_explain_from_spec_plan_and_result(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            code, _ = self.call(['demo', '--looks', '3', '--outcome', 'continue', '--out-dir', td]); self.assertEqual(code, 1)
            code, r = self.call(['explain', '--spec', str(p / 'decision-spec.json'), '--plan', str(p / 'frozen-plan.json'), '--result', str(p / 'decision.json')]); self.assertEqual(code, 0)
            self.assertEqual(r['data']['schema'], 'aver-explanation/v1'); self.assertEqual(r['data']['stopped'], 'continue')
            for k in ('purpose', 'what_was_tested', 'where_it_stands', 'what_decides', 'money', 'what_would_change_the_answer', 'audit', 'caveats', 'headline'): self.assertIn(k, r['data'])
            code, r2 = self.call(['explain', '--result', str(p / 'decision.json')]); self.assertEqual(code, 0); self.assertEqual(r2['data'], r['data'])
            code, r3 = self.call(['explain', '--spec', str(p / 'decision-spec.json'), '--plan', str(p / 'test-plan.json')]); self.assertEqual(code, 0); self.assertEqual(r3['data']['headline']['state'], 'planned')
            out = io.StringIO(); self.assertEqual(main(['decision', 'explain', '--result', str(p / 'decision.json')], stream=out), 0)
            text = out.getvalue()
            self.assertIn('WHERE EACH OPTION STANDS', text); self.assertIn('Not decided yet', text); self.assertNotIn('guarantee', text.lower())
            self.assertTrue(all(len(l) <= 80 for l in text.splitlines()[1:]))

    def test_human_readable_look(self):
        output = io.StringIO(); self.assertEqual(main(['decision', 'demo', '--looks', '3', '--outcome', 'continue'], stream=output), 1)
        self.assertIn('Round 1 of 3: KEEP TESTING', output.getvalue())
