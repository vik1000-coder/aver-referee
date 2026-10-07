"""Portable agent handoffs: no hosted service or private test helpers."""
import io
import json
from pathlib import Path
import tempfile
import unittest

from aver_cli.cli import main
from aver_cli.workflow import hosted_request
from aver_policy import decision_v2, label_assistance, log_reader
from aver_policy.decision import DecisionError


def call(args):
    output = io.StringIO()
    code = main(['decision', *args, '--json'], stream=output)
    return code, json.loads(output.getvalue())


class WorkflowHandoffTests(unittest.TestCase):
    def test_next_waits_for_logs_and_review_before_running_a_labeler(self):
        logs = [{'case_id': str(i), 'candidate_id': c, 'input': {'text': 'invoice'},
                 'output': {'amount': i}, 'expected': {'amount': i}, 'cost_usd': .01}
                for i in range(10) for c in ('current', 'candidate')]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, answer = call(['next', directory])
            self.assertEqual(code, 1)
            self.assertEqual(answer['data']['step']['stage'], 'logs')
            (root / 'logs.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in logs))
            code, answer = call(['label-request', '--logs', str(root / 'logs.jsonl'),
                                 '--model', 'chosen-model', '--include-content',
                                 '--out', str(root / 'label-request.json')])
            self.assertEqual(code, 0)
            code, answer = call(['next', directory])
            self.assertEqual(code, 1)
            self.assertEqual(answer['data']['step']['stage'], 'label-response')
            (root / 'labeler-command.txt').write_text('my-agent --request {request} --out {out}')
            code, answer = call(['next', directory])
            self.assertEqual(code, 0)
            self.assertEqual(answer['data']['step']['command'],
                             'my-agent --request label-request.json --out label-response.json')
            self.assertFalse((root / 'label-response.json').exists())

    def test_blinded_response_is_bound_to_model_and_unknown_is_not_success(self):
        logs = [{'case_id': str(i), 'candidate_id': c, 'input': {'api_key': 'private-value'},
                 'output': {'amount': i}, 'expected': {'amount': i}, 'cost_usd': .01}
                for i in range(10) for c in ('current', 'candidate')]
        request = label_assistance.make_request(logs, 'chosen-model', True)
        sent = json.dumps(request['messages'])
        self.assertNotIn('private-value', sent)
        self.assertNotIn('cost_usd', sent)
        records = json.loads(request['messages'][1]['content'])['records']
        response = {'purpose': 'Extract amounts.',
                    'deployment': {'mode': 'normal', 'why': 'Reviewed by a person.'},
                    'strata': {'field': '', 'why': 'Use the offline split.'},
                    'kinds': {'wrong_amount': {'label': 'Wrong amount', 'severity': 'mistake',
                                              'standard': 'wrong_output', 'evidence': ['case_0']}},
                    'labels': [{**{key: row[key] for key in ('case_id', 'candidate_id')},
                                'mistake': None, 'unsure': True, 'rationale': 'Cannot determine.'}
                               for row in records]}
        envelope = {'schema': 'aver-label-response/v1', 'model': request['model'],
                    'request_digest': request['request_digest'], 'response': response}
        proposal = log_reader.read_logs(logs, label_assistance.ResponseReader(request, envelope))
        self.assertTrue(all(row['completed'] is None for row in proposal['rows']))
        envelope['model'] = 'different-model'
        with self.assertRaises(DecisionError):
            label_assistance.ResponseReader(request, envelope)

    def test_hosted_packets_retry_identically_and_cannot_approve_a_plan(self):
        first = hosted_request('submit_evidence', 'app', 'plan', [{'case_id': 'one'}], 1)
        self.assertEqual(first, hosted_request('submit_evidence', 'app', 'plan', [{'case_id': 'one'}], 1))
        other = hosted_request('submit_evidence', 'app', 'plan', [{'case_id': 'two'}], 1)
        arguments = first['request']['params']['arguments']
        self.assertNotEqual(arguments['idempotency_key'], other['request']['params']['arguments']['idempotency_key'])
        self.assertNotIn('token', arguments)
        for operation in ('approve_plan', 'revoke', 'human_signoff'):
            with self.assertRaises(DecisionError):
                hosted_request(operation, 'app', 'plan')
        with self.assertRaises(DecisionError):
            hosted_request('submit_evidence', 'app', 'plan', [], 1)

    def test_memo_recomputes_result_and_fees_cannot_manufacture_savings(self):
        result = decision_v2.demo()['result']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            path.write_text(json.dumps(result))
            code, answer = call(['memo', '--result', str(path), '--monthly-fee', '10000'])
            self.assertEqual(code, 0)
            self.assertEqual(answer['data']['authority'], 'unsigned_local_economic_memo')
            self.assertIsNone(answer['data']['months_to_recover_checking_at_lower_saving'])
            altered = json.loads(json.dumps(result))
            altered['report']['choice'] = 'fabricated-option'
            path.write_text(json.dumps(altered))
            code, answer = call(['memo', '--result', str(path)])
            self.assertEqual(code, 3)
