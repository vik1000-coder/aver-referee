import io
import json
from pathlib import Path
import tempfile
import unittest
from aver_cli.cli import main


class DecisionCliTests(unittest.TestCase):
    def call(self,args):
        output=io.StringIO();status=main(['decision',*args,'--json'],stream=output)
        return status,json.loads(output.getvalue())

    def test_discovery_and_schema(self):
        status,r=self.call(['capabilities']);self.assertEqual(status,0);self.assertTrue(r['data']['offline'])
        for name in ['spec','evidence','plan','frozen-plan','result','bundle']:
            status,r=self.call(['schema','--name',name]);self.assertEqual(status,0);self.assertIn('$defs',r['data'])

    def test_complete_file_workflow_and_replay(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            code,r=self.call(['demo','--out-dir',td]);self.assertEqual(code,0)
            code,_=self.call(['demo','--out-dir',td]);self.assertEqual(code,0)
            code,_=self.call(['validate','--spec',str(p/'decision-spec.json'),'--evidence',str(p/'pilot.json')]);self.assertEqual(code,0)
            code,_=self.call(['explore','--spec',str(p/'decision-spec.json'),'--pilot',str(p/'pilot.json'),'--out',str(p/'exploration.json')]);self.assertEqual(code,0)
            code,_=self.call(['plan','--spec',str(p/'decision-spec.json'),'--exploration',str(p/'exploration.json'),'--finalists','1','--out',str(p/'planned.json')]);self.assertEqual(code,0)
            code,_=self.call(['freeze','--plan',str(p/'planned.json'),'--out',str(p/'frozen.json')]);self.assertEqual(code,0)
            code,r=self.call(['evaluate','--plan',str(p/'frozen.json'),'--evidence',str(p/'confirmation.json'),'--out',str(p/'evaluated.json')]);self.assertEqual(code,0)
            self.assertEqual(r['release_eligibility'],'not_evaluated')
            code,_=self.call(['explain','--result',str(p/'evaluated.json')]);self.assertEqual(code,0)
            code,r=self.call(['bundle','--result',str(p/'evaluated.json')]);self.assertEqual(code,0);self.assertEqual(r['data']['authority'],'local_assessment_only')
            p.joinpath('planned.json').write_text('{}')
            code,r=self.call(['plan','--spec',str(p/'decision-spec.json'),'--exploration',str(p/'exploration.json'),'--finalists','1','--out',str(p/'planned.json')]);self.assertEqual(code,3);self.assertEqual(r['errors'][0]['code'],'output_exists')

    def test_structured_errors_and_distinct_verdict_exits(self):
        for args in [[],['evaluate'],['schema','--name','missing'],['validate','--spec','/does/not/exist'],['unknown']]:
            status,result=self.call(args);self.assertEqual(status,3);self.assertTrue(result['errors'])
        for outcome,exit_code in [('pass',0),('refuse',1),('block',2)]:
            status,result=self.call(['demo','--outcome',outcome]);self.assertEqual(status,exit_code);self.assertEqual(result['verdict']['state'],outcome)
        status,result=self.call(['init']);self.assertEqual(status,0);self.assertIsNone(result['data']['requirements']['failure_limit']);self.assertTrue(result['warnings'])

    def test_jsonl_and_resume_binding(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);self.call(['demo','--out-dir',td])
            rows=json.loads((p/'confirmation.json').read_text())
            p.joinpath('partial.jsonl').write_text('\n'.join(json.dumps(r) for r in rows[:20]))
            code,_=self.call(['evaluate','--plan',str(p/'frozen-plan.json'),'--evidence',str(p/'partial.jsonl'),'--out',str(p/'partial-result.json')]);self.assertEqual(code,1)
            rows[0]['completed']=False;p.joinpath('changed.json').write_text(json.dumps(rows))
            code,r=self.call(['evaluate','--plan',str(p/'frozen-plan.json'),'--evidence',str(p/'changed.json'),'--previous',str(p/'partial-result.json')]);self.assertEqual(code,3);self.assertEqual(r['errors'][0]['code'],'earlier_evidence_changed')

    def test_human_readable_output(self):
        output=io.StringIO();self.assertEqual(main(['decision','demo'],stream=output),0)
        self.assertIn('Savings: not evaluated',output.getvalue());self.assertIn('local assessment',output.getvalue())
