"""The agent workflow around `aver decision` (docs/136): `accept`, `cases` and `next`.

A coding agent (or a cron job) runs the whole decision from a workspace directory with fixed file names.
`next` reads which files exist and says exactly which command comes next, what a person must decide first, and what must
not be done. Nothing here calls a model, the network or a harness; nothing here decides a verdict (the engine does).

The CLI may import only aver_policy, aver_cass and the stdlib allow-list (docs/104 §1), so no shlex or glob here.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

from aver_policy.decision import DecisionError, digest, need
from aver_policy import decision_v2 as v2
from aver_policy import log_reader as lr

CLI = 'python3 -m aver_cli decision'
NEXT_SCHEMA = 'aver-next-step/v1'
CASES_SCHEMA = 'aver-round-cases/v1'
CASE_ROW_SCHEMA = 'aver-round-case/v1'

# Fixed file names in a workspace (docs/136 §2). round-N-cases.jsonl, round-N.jsonl and result-N.json repeat per round.
FILES = {
    'logs': 'logs.jsonl (or logs.json): the pipeline\'s logged runs, as exported',
    'label-request.json': 'optional: label-request binds the reviewed request to the export and chosen model',
    'label-response.json': 'optional: your labeling agent returns proposed labels, request digest and reported model identity',
    'proposal.json': 'read-logs: what the agent is for, kinds of case and of mistake, a label per run, the audit sample',
    'spec.json': 'accept: the accepted proposal with the person\'s prices, tolerances, limits and volume',
    'pilot.jsonl': 'accept: the labelled logged runs, one aver-outcomes/v2 row each',
    'audit-sample.json': 'audit: the runs a person must re-label (AI-fine and AI-flagged, drawn at random)',
    'audit-labels.jsonl': 'the person: one label per sampled run',
    'audited-spec.json': 'audit: spec.json plus the audit counts (grader_audit)',
    'exploration.json': 'explore: options ranked on the pilot',
    'plan.json': 'plan: the priced test (rounds, cases per round, cost, chance of an answer)',
    'frozen.json': 'freeze: the plan, frozen; nothing upstream may change after this',
    'round-N-cases.jsonl': 'cases: the cases the harness must run in round N',
    'round-N.jsonl': 'your harness: the outcomes of round N (aver-outcomes/v2 rows)',
    'result-N.json': 'look: the decision after round N (verifiable)',
    'explanation.json': 'explain: what the evidence says, in the person\'s terms',
    'bundle.json': 'bundle: the result packaged for review',
    'economic-memo.json': 'optional: memo verifies the result and includes your allocated monthly fee; unsigned, never release authority',
    'harness-command.txt': 'optional: your harness command with {cases}, {out} and {round}; next fills it in',
    'labeler-command.txt': 'optional: your chosen labeling agent with {request} and {out}; configuring it authorizes that local handoff',
}

ALWAYS_DO_NOT = [
    'A model may propose labels through label-request and read-logs --reader model-response. It must not decide the verdict, '
    'supply human audit labels, or invent outcomes. Outcomes come from your harness; a person independently audits proposed labels.',
    'Do not hand-edit any file Aver wrote (proposal, spec, plan, frozen plan, results). Re-run the command instead.',
]


# ----------------------------------------------------------------------------- small helpers

def quote(s):
    s = str(s)
    if re.fullmatch(r'[\w@%+=:,./<>{}-]+', s): return s
    return "'" + s.replace("'", "'\"'\"'") + "'"


def shell(argv):
    return CLI + ' ' + ' '.join(quote(a) for a in argv)


def load(path, rows=False):
    text = Path(path).read_text(encoding='utf-8')
    if rows and text.lstrip() and not text.lstrip().startswith('['):
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def number_arg(text, what, percent=False):
    t = str(text).strip()
    try:
        if percent and t.endswith('%'): return float(t[:-1].replace(',', '')) / 100
        return float(t.replace(',', ''))
    except ValueError:
        raise DecisionError('invalid_arguments', what, f'{what} must be a number, got "{text}".') from None


def pairs_arg(text, what, kinds, unit):
    """'wrong_field=2,missed_page=2' -> {kind: float}, with plain-word errors."""
    out = {}
    if not text: return out
    for part in str(text).split(','):
        if not part.strip(): continue
        need('=' in part, 'invalid_arguments', what, f'Write {what} as kind=value pairs separated by commas, for example {next(iter(kinds), "wrong_field")}=2. Got "{part.strip()}".')
        k, v = (x.strip() for x in part.split('=', 1))
        need(k in kinds, 'unknown_kind', what, f'There is no mistake kind called "{k}" in the proposal. The kinds are: {", ".join(sorted(kinds))}.')
        need(k not in out, 'invalid_arguments', what, f'"{k}" is given twice in {what}.')
        try:
            x = float(v.replace(',', '').lstrip('$'))
        except ValueError:
            raise DecisionError('invalid_arguments', what, f'The {unit} for {k} must be a number, like {k}=2. Got "{v}".') from None
        out[k] = x
    return out


# ----------------------------------------------------------------------------- accept

def accept(proposal, prices=None, tolerances=None, monthly_uses=None, price_per_case=None, failure_limit=None, incident_limit=None,
           app=None, checkpoint=None, candidates=None, baseline=None, budget_runs=None, horizon_months=12, purpose=None):
    """An accepted proposal -> (aver-decision-spec/v3, pilot rows, warnings). Agents never hand-write specs (docs/136)."""
    need(isinstance(proposal, dict) and proposal.get('schema') == 'aver-log-proposal/v1', 'unsupported_schema', 'proposal',
         'accept needs proposal.json from read-logs (schema aver-log-proposal/v1).')
    kinds = proposal['taxonomy']['kinds']
    pr = pairs_arg(prices, '--prices', kinds, 'price in dollars')
    tol = pairs_arg(tolerances, '--tolerances', kinds, 'tolerance (how many a month)')
    for k, x in pr.items(): need(0 <= x <= 1_000_000, 'invalid_price', '--prices', f'The price for {k} must be zero or more dollars per occurrence.')
    for k, x in tol.items(): need(x > 0, 'invalid_tolerance', '--tolerances', f'The tolerance for {k} must be more than zero a month (it is "at most N a month"). To forbid it entirely, choose a small N such as 1.')
    need(monthly_uses is not None, 'missing_volume', '--monthly-uses', 'Say how many times a month the pipeline runs (--monthly-uses 30000): tolerances are per month.')
    v = number_arg(monthly_uses, '--monthly-uses')
    need(v == int(v) and 1 <= v <= 1_000_000, 'invalid_volume', '--monthly-uses', 'Monthly uses must be a whole number from 1 to 1,000,000.')
    need(price_per_case is not None, 'missing_price_per_case', '--price-per-case', 'Say what checking one test case costs in dollars (--price-per-case 2): it prices the test before it runs.')
    pc = number_arg(price_per_case, '--price-per-case')
    need(pc >= 0, 'invalid_price', '--price-per-case', 'The price per checked case must be zero or more dollars.')
    need(failure_limit is not None, 'missing_failure_limit', '--failure-limit', 'Say the overall failure limit as a share of runs (--failure-limit 8%): the most failed runs you accept, of every kind together.')
    fl = number_arg(failure_limit, '--failure-limit', True)
    need(0 < fl < 1, 'invalid_limit', '--failure-limit', 'The failure limit is a share of runs above 0% and below 100%, like 8% or 0.08.')
    il = fl if incident_limit is None else number_arg(incident_limit, '--incident-limit', True)
    need(0 < il < 1, 'invalid_limit', '--incident-limit', 'The incident limit is a share of runs above 0% and below 100%.')
    ids = None
    logged = sorted({r['candidate_id'] for r in proposal['rows']})
    if candidates:
        ids = [c.strip() for c in str(candidates).split(',') if c.strip()]
        for c in ids: need(c in logged, 'unknown_candidate', '--candidates', f'There is no option called "{c}" in the logs. The options are: {", ".join(logged)}.')
    need(2 <= len(ids or logged) <= 4, 'candidate_count', '--candidates', f'Aver compares two to four options; the logs have {len(logged)} ({", ".join(logged)}). Pick some with --candidates.')
    if baseline: need(baseline in (ids or logged), 'unknown_candidate', '--baseline', f'The baseline must be one of the options compared: {", ".join(ids or logged)}.')
    keep = set(ids or logged)
    rows = [r for r in proposal['rows'] if r['candidate_id'] in keep]
    if budget_runs is not None:
        b = number_arg(budget_runs, '--budget-runs')
        need(b == int(b) and b > len(rows), 'invalid_budget', '--budget-runs', f'The budget counts every run, the {len(rows)} logged ones included, so it must be more than {len(rows)}.')
        budget_runs = int(b)
    need(purpose is None or (isinstance(purpose, str) and 0 < len(purpose.strip()) <= 300), 'invalid_purpose', '--purpose', 'Say what the agent is for in one sentence of at most 300 characters.')
    spec = lr.proposal_to_spec(proposal, {'app': app or 'agent pipeline', 'checkpoint': checkpoint or 'completed run'},
                               {'failure_limit': fl, 'incident_limit': il}, int(v), pc, int(horizon_months), ids, baseline, budget_runs, purpose)
    for k, kind in spec['taxonomy']['kinds'].items():
        kind['price_usd'] = pr.get(k); kind['tolerance_per_month'] = tol.get(k)
    spec = v2.validate_spec_v2(spec)
    v2.validate_mistakes(spec, rows)
    warnings = []
    loose = [k for k in kinds if k not in pr and k not in tol]
    if loose:
        warnings.append('No price and no tolerance for ' + ', '.join(sorted(loose)) + ': they still count as failures against the failure limit, but cost nothing in the comparison and have no limit of their own.')
    both = sorted(set(pr) & set(tol))
    if both: warnings.append('Both a price and a tolerance for ' + ', '.join(both) + ': the price feeds the monthly cost and the tolerance is a limit.')
    if incident_limit is None: warnings.append(f'Incident limit not given: using the failure limit ({fl:.1%}) for incidents in each kind of case.')
    for k, n in sorted(tol.items()):
        if n / v < 1e-3: warnings.append(f'The tolerance for {k} is {n:g} a month at {int(v):,} uses ({n / v:.3%} of runs): showing it may need many thousands of cases; plan reports the exact number.')
    if any(r.get('origin') == 'ai-labeled' for r in rows):
        warnings.append('The labels came from a log reader, so a person must audit a sample before any pass (audit).')
    return spec, rows, warnings


# ----------------------------------------------------------------------------- cases for one round

def round_cases(frozen, j):
    """The cases the harness must run in round j, one entry per (case, option)."""
    need(isinstance(frozen, dict) and frozen.get('schema') in ('aver-frozen-plan/v2', 'aver-frozen-plan/v3'), 'unsupported_schema', '--plan',
         'cases needs frozen.json from freeze (a v2 or v3 frozen plan).')
    plan = frozen['plan']; K = plan['looks']
    need(type(j) is int and 1 <= j <= K, 'invalid_round', '--round', f'This plan has rounds 1 to {K}; got {j}.')
    digests = {c['id']: c['digest'] for c in plan['exploration']['spec']['candidates']}
    options = list(plan['selected'])
    if plan.get('paired') and plan['exploration']['spec']['baseline'] not in options:
        options.append(plan['exploration']['spec']['baseline'])
    out = []
    for case in plan['confirmation_cases']:
        if case['look'] != j: continue
        for c in options:
            out.append({'schema': CASE_ROW_SCHEMA, 'round': j, 'case_id': case['case_id'], 'stratum': case['stratum'], 'candidate_id': c, 'configuration_sha256': digests[c]})
    return {'schema': CASES_SCHEMA, 'round': j, 'rounds': K, 'plan_digest': frozen['plan_digest'], 'options': options,
            'cases': out, 'how': ('For each entry, draw a NEW case of this kind of case (stratum) from your traffic, never one from the logs, '
                                  'run this option on it with your harness, and write one aver-outcomes/v2 row with the same case_id, '
                                  'candidate_id, configuration_sha256 and stratum. The same case_id must be the same case content for every option.')}


def outcome_example():
    return {'schema': 'aver-outcomes/v2', 'case_id': 'confirmation-scanned-0', 'case_sha256': '<sha256 of the case content, 64 hex>',
            'candidate_id': 'guarded', 'configuration_sha256': '<from round-N-cases.jsonl>', 'stratum': 'scanned', 'phase': 'confirmation',
            'origin': 'ai-labeled', 'completed': False, 'incident': False, 'cost_usd': 0.0043, 'mistake': 'wrong_field'}


def hosted_request(operation, app_id, plan_id, rows=None, look=None):
    """A ready-to-send MCP request for an already human-approved plan. No token, network or approval powers."""
    need(operation in ('submit_evidence', 'run_check', 'read_verdict') and all(isinstance(x, str) and 0 < len(x) <= 120 for x in (app_id, plan_id)),
         'invalid_hosted_request', 'request', 'Name the operation, App and approved decision plan.')
    args = {'app_id': app_id, 'decision_plan_id': plan_id}
    if operation == 'submit_evidence':
        need(type(look) is int and 1 <= look <= 5 and isinstance(rows, list) and 0 < len(rows) <= 200_000 and all(isinstance(r, dict) for r in rows),
             'invalid_hosted_request', 'round', 'Submitting evidence needs a round number and its outcome rows.')
        payload = {'look': look, 'rows': rows, 'client_report': None, 'client_report_digest': None}
        args['decision_round'] = payload
    else:
        need(rows is None and look is None, 'invalid_hosted_request', 'request', 'Only submit_evidence takes round rows.')
        payload = {}
    if operation != 'read_verdict':
        args['idempotency_key'] = 'decision:' + digest({'operation': operation, 'app_id': app_id, 'plan_id': plan_id, 'payload': payload})
    return {'endpoint': '/api/mcp', 'requires': 'App-bound bearer with this operation; a person approves the plan first.',
            'request': {'jsonrpc': '2.0', 'id': f'{operation}:{plan_id}:{look or 0}', 'method': 'tools/call', 'params': {'name': operation, 'arguments': args}}}


def merge_round_rows(frozen, rows, previous):
    """look accepts either the rows of the current round only (with --previous) or all rows so far. Returns (rows, note)."""
    plan = frozen['plan']; manifest = {c['case_id']: c['look'] for c in plan['confirmation_cases']}
    for i, r in enumerate(rows):
        need(isinstance(r, dict) and 'case_id' in r and 'candidate_id' in r, 'invalid_evidence', f'evidence[{i}]', 'Each outcome row needs case_id and candidate_id (see look --help).')
        need(r['case_id'] in manifest, 'unplanned_case', f'evidence[{i}]', f'"{r["case_id"]}" is not a case in the frozen plan. Run only the cases in round-N-cases.jsonl.')
    rounds_in = sorted({manifest[r['case_id']] for r in rows})
    if previous is None:
        need(not rows or rounds_in[0] == 1, 'previous_required', '--previous',
             f'These rows are for round {rounds_in[0] if rows else "?"}; pass --previous with the result of round {rounds_in[0] - 1 if rows else "?"} (result-{rounds_in[0] - 1 if rows else "N"}.json).')
        return rows, None
    prev_rows = previous['evidence']; done = previous['report']['look']
    prev_keys = {(r['candidate_id'], r['case_id']) for r in prev_rows}
    keys = {(r['candidate_id'], r['case_id']) for r in rows}
    if not (keys & prev_keys):
        need(rounds_in == [done + 1], 'wrong_round', '--evidence',
             f'With --previous (round {done}) a round-only file must hold round {done + 1} cases only; these rows are for round(s) {", ".join(map(str, rounds_in)) or "none"}.')
        return prev_rows + rows, f'Round-only rows: added the {len(prev_rows)} rows of rounds 1 to {done} from the previous result.'
    need(prev_keys <= keys, 'mixed_round_rows', '--evidence',
         'The rows repeat some earlier rows but not all. Give either the rows of the new round only, or every row so far (earlier rows unchanged).')
    new_rounds = sorted({manifest[c] for _, c in keys - prev_keys})
    need(bool(new_rounds), 'round_already_recorded', '--evidence', f'These are the rows of rounds 1 to {done} again. A round is never re-run; give the rows of round {done + 1}.')
    need(new_rounds == [done + 1], 'wrong_round', '--evidence', f'After round {done} the new rows must be round {done + 1} cases only; these include round(s) {", ".join(map(str, new_rounds))}.')
    return rows, None


# ----------------------------------------------------------------------------- next

def _round_files(ws):
    found = {'cases': {}, 'outcomes': {}, 'results': {}}
    for p in sorted(ws.iterdir()):
        m = re.fullmatch(r'round-(\d+)-cases\.jsonl', p.name)
        if m: found['cases'][int(m.group(1))] = p.name; continue
        m = re.fullmatch(r'round-(\d+)\.jsonl', p.name)
        if m: found['outcomes'][int(m.group(1))] = p.name; continue
        m = re.fullmatch(r'result-(\d+)\.json', p.name)
        if m: found['results'][int(m.group(1))] = p.name
    return found


def _step(stage, actor, why, argv=None, command=None, person=None, do_not=(), writes=(), template=False, fill=None):
    return {'stage': stage, 'actor': actor, 'command': command if command is not None else (shell(argv) if argv else None),
            'argv': (['python3', '-m', 'aver_cli', 'decision', *argv] if argv else None), 'template': template, 'fill': fill or {},
            'why': why, 'person_decides': person, 'do_not': list(do_not) + ALWAYS_DO_NOT, 'writes': list(writes)}


class Inconsistent(Exception):
    def __init__(self, why, fix):
        super().__init__(why); self.why = why; self.fix = fix


def _check(cond, why, fix):
    if not cond: raise Inconsistent(why, fix)


def _ai_labeled(rows):
    return any(r.get('origin') == 'ai-labeled' for r in rows)


def audit_plan(spec, pilot):
    """How many AI-fine runs the audit needs: enough for the TIGHTEST failure or tolerance limit, not only the overall one
    (the plan's audit.fine_rows_needed is sized the same way since 2026-10-01; this keeps the CLI's wording). Capped by the runs the logs hold."""
    raw = {k: v for k, v in v2.raw_limits(spec).items() if v is not None}
    names = {'average:failure': 'the overall failure limit'}
    tightest = min(raw, key=lambda k: (raw[k], k))
    label = names.get(tightest, f'the tolerance for {tightest[len("tolerance:"):]}' if tightest.startswith('tolerance:') else tightest)
    needed = v2.fine_rows_needed(raw[tightest])
    fine = sum(1 for r in pilot if r.get('origin') == 'ai-labeled' and r['completed'] is True and r['incident'] is not True)
    flagged = sum(1 for r in pilot if r.get('origin') == 'ai-labeled' and (r['completed'] is False or r['incident'] is True))
    enough = needed is not None and needed <= fine
    n = min(fine, needed) if needed is not None else fine
    why = (f"The tightest limit is {label} ({raw[tightest]:.2%} of runs): a pass on it needs about {needed:,} AI-fine runs checked with no missed failures"
           if needed is not None else f'The tightest limit is {label}, which is zero: no audit can support a pass on it')
    why += f"; the logs hold {fine:,}. The sample: {n:,} AI-fine and {min(flagged, 100):,} AI-flagged runs."
    return {'fine': n, 'flagged': min(flagged, 100), 'tightest': label, 'tightest_limit': raw[tightest], 'fine_rows_needed_tightest': needed,
            'ai_fine_rows': fine, 'ai_flagged_rows': flagged, 'enough': enough, 'why': why}


def _accept_template(prop):
    kinds = prop['taxonomy']['kinds']; d = prop.get('defaults', {})
    priced = [k for k in sorted(kinds) if d.get(k, 'price') == 'price']
    tolerated = [k for k in sorted(kinds) if d.get(k) == 'tolerance']
    argv = ['accept', '--proposal', 'proposal.json', '--prices', ','.join(f'{k}=<$>' for k in priced) or '<kind>=<$>',
            '--tolerances', ','.join(f'{k}=<N a month>' for k in tolerated) or '<kind>=<N a month>',
            '--failure-limit', '<share of runs, e.g. 8%>', '--monthly-uses', '<runs a month>', '--price-per-case', '<$ to check one case>',
            '--out', 'spec.json', '--pilot-out', 'pilot.jsonl']
    fill = {'--prices': 'Dollars per occurrence, for kinds that have a cost (proposed as prices: ' + (', '.join(priced) or 'none') + ').',
            '--tolerances': 'At most N a month, for kinds with no dollar cost (proposed as tolerances: ' + (', '.join(tolerated) or 'none') + ').',
            '--failure-limit': 'The most failed runs accepted overall, as a share of runs.',
            '--monthly-uses': 'How many runs a month the pipeline handles.',
            '--price-per-case': 'What it costs to run and check one test case, in dollars.',
            'optional': '--candidates (2 to 4 options), --baseline (the one in use now), --budget-runs (total runs you will pay for, logged ones included), --incident-limit.'}
    return argv, fill


def next_step(workspace='.'):
    """(exit code, aver-next-step/v1). 0: a next command; 1: a person must decide first; 2: finished; 3: inconsistent."""
    ws = Path(workspace)
    need(ws.is_dir(), 'not_a_workspace', str(ws), 'The workspace must be an existing directory.')
    names = {p.name for p in ws.iterdir() if p.is_file()}
    logs = sorted(n for n in names if re.fullmatch(r'logs\.(jsonl|json)', n))
    rf = _round_files(ws)
    present = sorted(n for n in names if n in FILES or n in logs) + sorted(rf['cases'].values()) + sorted(rf['outcomes'].values()) + sorted(rf['results'].values())
    data = {'schema': NEXT_SCHEMA, 'workspace': str(ws), 'cwd': 'Run every command from the workspace directory.', 'files_present': present, 'state': {}}
    try:
        code, step = _decide(ws, names, logs, rf, data['state'])
    except Inconsistent as exc:
        code, step = 3, _step('inconsistent', 'person', exc.why, person=exc.fix,
                              do_not=['Do not delete or rewrite files to make the workspace look consistent without a person agreeing: the earlier files are the record of what was decided.'])
    except (DecisionError, OSError, ValueError, KeyError, TypeError) as exc:
        code, step = 3, _step('inconsistent', 'person', f'A workspace file could not be read or checked: {exc}', person='Find which file is damaged or hand-edited; re-create it with its command, or start a new workspace.')
    data['step'] = step
    return code, data


def _decide(ws, names, logs, rf, state):
    has = lambda n: n in names
    rounds_present = bool(rf['cases'] or rf['outcomes'] or rf['results'])
    # chain order: nothing may exist without what it is made from
    chain = [('proposal.json', bool(logs), 'a logs file'), ('spec.json', has('proposal.json'), 'proposal.json'), ('pilot.jsonl', has('spec.json'), 'spec.json'),
             ('audit-sample.json', has('spec.json'), 'spec.json'), ('audit-labels.jsonl', has('audit-sample.json'), 'audit-sample.json'),
             ('audited-spec.json', has('audit-labels.jsonl'), 'audit-labels.jsonl'), ('exploration.json', has('pilot.jsonl'), 'pilot.jsonl'),
             ('plan.json', has('exploration.json'), 'exploration.json'), ('frozen.json', has('plan.json'), 'plan.json'),
             ('explanation.json', bool(rf['results']), 'a round result'), ('bundle.json', bool(rf['results']), 'a round result')]
    for name, ok, src in chain:
        _check(not has(name) or ok, f'{name} exists but {src}, which it is made from, does not.', f'Restore {src}, or start a new workspace from the logs.')
    _check(not rounds_present or has('frozen.json'), 'Round files exist but frozen.json does not.', 'Restore frozen.json, or start a new workspace.')
    _check(len(logs) <= 1, 'Two log files (' + ', '.join(logs) + ').', 'Keep one: logs.jsonl or logs.json.')

    if not logs:
        return 1, _step('logs', 'person', 'There are no logs in this workspace yet.',
                        person='Export the pipeline\'s logged runs (every option on the same cases) to logs.jsonl in this directory: JSONL in the shape your harness writes, or an OTLP/JSON trace export as logs.json.',
                        writes=['logs.jsonl'])
    if not has('proposal.json'):
        if has('label-response.json'):
            _check(has('label-request.json'), 'label-response.json exists without label-request.json.', 'Restore the original labeling request.')
            return 0, _step('read-logs', 'agent', 'Validate the labeling agent\'s response against this export and read its proposed labels.',
                            ['read-logs', '--logs', logs[0], '--reader', 'model-response', '--label-request', 'label-request.json',
                             '--label-response', 'label-response.json', '--out', 'proposal.json'], writes=['proposal.json'])
        if has('label-request.json'):
            if has('labeler-command.txt'):
                line = next((l.strip() for l in (ws / 'labeler-command.txt').read_text(encoding='utf-8').splitlines() if l.strip() and not l.lstrip().startswith('#')), '')
                _check(bool(line) and '{request}' in line and '{out}' in line, 'labeler-command.txt needs {request} and {out}.', 'Configure the labeling agent command explicitly.')
                return 0, _step('label-response', 'harness', 'Run the labeling agent you configured, then validate its saved proposal.',
                                command=line.replace('{request}', 'label-request.json').replace('{out}', 'label-response.json'),
                                writes=['label-response.json'], do_not=['Send only the request messages and response_schema to the provider; never use this agent for human audit labels.'])
            return 1, _step('label-response', 'person', 'A labeling handoff is prepared; no model has been called.',
                            person='Review and redact the request, then have your chosen labeling agent use its messages and response_schema. Save the request-bound response as label-response.json. It proposes labels; a person still audits them.')
        return 0, _step('read-logs', 'agent', 'Read the logs: what the agent is for, the kinds of case and of mistake, a label per run.',
                        ['read-logs', '--logs', logs[0], '--out', 'proposal.json'], writes=['proposal.json'],
                        do_not=['Do not use --reader mock-llm: it is a test double that reads hidden labels.'])
    prop = load(ws / 'proposal.json')
    _check(prop.get('schema') == 'aver-log-proposal/v1', 'proposal.json is not a read-logs proposal.', 'Re-run read-logs into a new workspace.')
    state['proposal'] = {'purpose': prop['purpose'], 'deployment': prop['deployment']['mode'], 'kinds': sorted(prop['taxonomy']['kinds']), 'options': sorted({r['candidate_id'] for r in prop['rows']})}
    if not has('spec.json'):
        argv, fill = _accept_template(prop)
        return 1, _step('accept', 'person', 'The proposal needs a person to accept it and to say what each kind of mistake is worth.', argv=argv, template=True, fill=fill,
                        person=('Accept the proposal (purpose, kinds of case, kinds of mistake) and decide, kind by kind, a price in dollars if the mistake has one '
                                'or a tolerance ("at most N a month") if it does not; also the overall failure limit, the monthly volume, the price of checking one case '
                                'and, optionally, the budget. Then run the accept command with those numbers.'),
                        do_not=['Do not choose prices, tolerances or limits yourself: they are the person\'s decision.', 'Do not hand-write spec.json; use accept.'],
                        writes=['spec.json', 'pilot.jsonl'])
    _check(has('pilot.jsonl'), 'spec.json exists but pilot.jsonl does not (accept writes both).', 'Re-run the same accept command with --pilot-out pilot.jsonl.')
    spec = v2.validate_spec_v2(load(ws / 'spec.json'))
    pilot = load(ws / 'pilot.jsonl', True)
    v2.validate_mistakes(spec, pilot)
    _check('grader_audit' not in spec, 'spec.json already carries an audit; audits belong in audited-spec.json.', 'Re-run accept into a new workspace.')
    needs_audit = _ai_labeled(pilot)
    state['audit'] = {'needed': needs_audit}
    after_freeze = ['Do not change spec.json, audited-spec.json, plan.json or frozen.json: the frozen plan binds them, and a changed file makes every result unverifiable.']
    if needs_audit:
        early = [n for n in ('exploration.json', 'plan.json', 'frozen.json') if has(n)]
        _check(has('audited-spec.json') or not early, ', '.join(early) + ' exist but the audit has not been applied: a plan made from unaudited reader labels can never pass (audit_required).',
               'Start a new workspace, or remove ' + ', '.join(early) + ' and later files if nothing was run, then audit before explore and plan.')
        if not has('audit-sample.json'):
            size = audit_plan(spec, pilot)
            state['audit'].update(size)
            argv = ['audit', '--proposal', 'proposal.json', '--spec', 'spec.json', '--fine', str(size['fine']), '--flagged', str(size['flagged']), '--out', 'audit-sample.json']
            why = ('The labels came from a log reader, so a person must re-label a random sample before any pass. Draw it now: '
                   'runs the reader called fine (to bound the failures it misses) and runs it flagged (to measure false flags). ' + size['why'])
            dn = ['Do not pick the cases to audit yourself; the sample is drawn at random by Aver.']
            if not size['enough']:
                return 1, _step('audit-sample', 'person', why, argv, writes=['audit-sample.json'], do_not=dn,
                                person=('No audit these logs can hold makes a pass possible on ' + size['tightest'] + '. A test can still show a fail. '
                                        'Decide: run this command anyway (only a fail or an inconclusive answer is then possible on that clause), or loosen that limit, '
                                        'price the mistake instead, or collect more logs, and start a new workspace.'))
            return 0, _step('audit-sample', 'agent', why, argv, writes=['audit-sample.json'], do_not=dn)
        sample = load(ws / 'audit-sample.json')
        _check(sample.get('schema') == 'aver-audit-sample/v1', 'audit-sample.json is not an audit sample from audit.', 'Re-draw it into a new workspace.')
        entries = sample['audit_sample']; a = sample['audit']
        state['audit'].update({'fine_rows': a['fine_rows'], 'flagged_rows': a['flagged_rows'], 'fine_rows_needed': a.get('fine_rows_needed')})
        if not has('audit-labels.jsonl'):
            tight = audit_plan(spec, pilot)
            return 1, _step('audit-labels', 'person', f"A person must re-label the {len(entries)} sampled runs ({a['fine_rows']} the reader called fine, {a['flagged_rows']} it flagged): "
                            "the fine ones bound how many failures the reader misses, the flagged ones how often it cries wolf. " + tight['why'].split('; the logs hold')[0] + '.',
                            person=('Open each sampled run in your own tools and write one line per run to audit-labels.jsonl: '
                                    '{"candidate_id": ..., "case_id": ..., "completed": true|false, "incident": true|false, "mistake": "<kind>" or null}. '
                                    'Judge the run itself; do not look at the AI\'s label (ai_completed, ai_incident, ai_mistake) first.'),
                            do_not=['Do not label the audit sample yourself, with a model, or by copying the AI labels: an audit the agent wrote is no audit.'],
                            writes=['audit-labels.jsonl'])
        labels = load(ws / 'audit-labels.jsonl', True)
        have = {(l.get('candidate_id'), l.get('case_id')) for l in labels if isinstance(l, dict)}
        missing = [s for s in entries if (s['candidate_id'], s['case_id']) not in have]
        if missing:
            return 1, _step('audit-labels', 'person', f'{len(missing)} of the {len(entries)} sampled runs have no label in audit-labels.jsonl yet (first: {missing[0]["candidate_id"]} {missing[0]["case_id"]}).',
                            person='Finish labelling every sampled run; the audit counts only a complete sample.',
                            do_not=['Do not fill in the missing labels yourself or with a model.'], writes=['audit-labels.jsonl'])
        if not has('audited-spec.json'):
            return 0, _step('audit-apply', 'agent', 'Every sampled run has the person\'s label. Record the audit: it sets the limit the AI labels are held to.',
                            ['audit', '--spec', 'spec.json', '--sample', 'audit-sample.json', '--labels', 'audit-labels.jsonl', '--out', 'audited-spec.json'],
                            writes=['audited-spec.json'])
        work = v2.validate_spec_v2(load(ws / 'audited-spec.json'))
        _check('grader_audit' in work and {k: v for k, v in work.items() if k != 'grader_audit'} == spec,
               'audited-spec.json is not spec.json plus the audit (one of them changed).', 'Re-run the audit into a new workspace from the current spec.json.')
        state['audit']['counts'] = work['grader_audit']
        work_name = 'audited-spec.json'
    else:
        _check(not has('audit-sample.json') and not has('audited-spec.json'), 'An audit file exists but the pilot labels did not come from a reader.', 'Remove the audit files or start a new workspace.')
        work, work_name = spec, 'spec.json'
    state['spec'] = work_name
    if not has('exploration.json'):
        return 0, _step('explore', 'agent', 'Compare the options on the logged runs, all-in per month (running cost plus priced mistakes).',
                        ['explore', '--spec', work_name, '--pilot', 'pilot.jsonl', '--out', 'exploration.json'], writes=['exploration.json'])
    ex = load(ws / 'exploration.json')
    _check(digest(ex.get('spec')) == digest(v2.v1_view(work)), f'exploration.json was made from a different {work_name}.', 'The spec changed after explore. Start a new workspace (or remove exploration.json and later files if nothing was frozen).')
    if not has('plan.json'):
        q = 1
        return 0, _step('plan', 'agent', 'Price the test before it runs: how many rounds and cases, what it costs, the chance of an answer.',
                        ['plan', '--spec', work_name, '--exploration', 'exploration.json', '--finalists', str(q), '--rounds', '3', '--out', 'plan.json'], writes=['plan.json'],
                        person='Nothing new if the budget was set in accept (--budget-runs). The person may ask for more finalists (--finalists 2 or 3, to test several options) or a smaller cap (--cap N cases per option).')
    plan = load(ws / 'plan.json')
    _check(digest(plan.get('exploration')) == digest(ex), 'plan.json was made from a different exploration.json.', 'Re-plan in a new workspace.')
    _check(digest(v2.spec_to_v2(plan['exploration']['spec'], plan)) == digest(work), f'plan.json was made from a different {work_name}.', f'{work_name} changed after plan. Start a new workspace.')
    state['plan'] = {'options': plan['selected'], 'rounds': plan['looks'], 'cases_per_option': plan['allocation']['total'], 'feasible': plan['feasible'],
                     'chance_of_pass_central': plan['operating_characteristics']['scenarios']['central']['pass']}
    forecast = plan['operating_characteristics']['scenarios']['central']
    state['plan'].update({'chance_of_limits_met_central': forecast['pass'],
                          'chance_of_clear_answer_central': forecast['pass'] + forecast['block'],
                          'chance_of_saving_shown_central': forecast.get('p_saving_shown'),
                          'estimated_net_value_usd': forecast['net_value_usd'] if 'p_saving_shown' in forecast else None,
                          'forecast_assumptions': 'Logged rates hold; switch only after a pass with positive saving shown. Subscription fees and unlogged setup effort are extra.'})
    if needs_audit and v2.is_v3(work): state['plan']['audit'] = plan.get('audit')
    if not plan['feasible'] and not has('frozen.json'):
        floors = {'average failure limit': plan.get('average_floor'), **{f'tolerance for {k}': f for k, f in (plan.get('tolerance_floors') or {}).items()}}
        worst = max(((k, f) for k, f in floors.items() if f is not None), key=lambda kv: kv[1], default=None)
        return 1, _step('plan-infeasible', 'person', f"This test cannot pass even with no mistakes: {plan['allocation']['total']:,} cases per option, but "
                        + (f'the {worst[0]} needs {worst[1]:,} with none seen.' if worst else 'a limit is zero.'),
                        person='Loosen a tolerance, price that mistake instead, raise the budget (--budget-runs in accept, or --cap in plan) or test fewer options. Then start a new workspace from the logs.',
                        do_not=['Do not edit spec.json or plan.json by hand.'])
    if not has('frozen.json'):
        return 0, _step('freeze', 'agent', f"Freeze the test: {', '.join(plan['selected'])} on up to {plan['allocation']['total']:,} new cases each, in {plan['looks']} round(s).",
                        ['freeze', '--plan', 'plan.json', '--out', 'frozen.json'], writes=['frozen.json'], do_not=after_freeze,
                        person='If the price of the test in plan.json is more than the person agreed to spend, stop and ask before freezing.')
    frozen = load(ws / 'frozen.json')
    _check(frozen.get('plan') == plan and frozen.get('plan_digest') == digest(v2.plan_core(plan)), 'frozen.json does not freeze plan.json (one of them changed).', 'Start a new workspace; a frozen plan cannot be edited.')
    _check(not (needs_audit and v2.is_v3(work) and 'grader_audit' not in plan['economics_inputs']),
           'The plan was made from labels a reader wrote but carries no audit, so no evidence can pass (audit_required).', 'Start a new workspace and audit before planning.')
    return _rounds(ws, names, rf, frozen, plan, state, after_freeze)


def _rounds(ws, names, rf, frozen, plan, state, after_freeze):
    K = plan['looks']
    results = rf['results']
    rounds_do_not = after_freeze + ['Do not re-run a round, replace or delete rows already recorded, or run a later round\'s cases early.',
                                    'Do not add cases that are not in round-N-cases.jsonl, and never reuse a logged (pilot) case as a round case.']
    for name in list(rf['cases'].values()) + list(rf['outcomes'].values()) + list(results.values()):
        n = int(re.findall(r'\d+', name)[0]); _check(1 <= n <= K, f'{name} is outside the plan\'s rounds (1 to {K}).', 'Remove it; it is not part of this plan.')
    done = max(results) if results else 0
    _check(sorted(results) == list(range(1, done + 1)), 'A round result is missing: ' + ', '.join(f'result-{j}.json' for j in range(1, done + 1) if j not in results) + '.', 'Restore the missing result, or start a new workspace.')
    prev_rows, report = [], None
    for j in range(1, done + 1):
        res = load(ws / results[j])
        _check(res.get('frozen_plan') == frozen, f'result-{j}.json belongs to a different frozen plan.', 'Start a new workspace.')
        rep = res['report']
        _check(rep.get('look') == j, f'result-{j}.json is not a complete round {j} (it reports round {rep.get("look")}, {rep.get("stopped")}).',
               f'Complete the missing rows in round-{j}.jsonl (never change rows already recorded), remove result-{j}.json and run look again.')
        keys = {(r['candidate_id'], r['case_id']) for r in res['evidence']}
        _check(all((r['candidate_id'], r['case_id']) in keys and r in res['evidence'] for r in prev_rows), f'result-{j}.json changed rows of an earlier round.', 'Start a new workspace; recorded rows cannot change.')
        _check(j == done or rep['stopped'] == 'continue', f'result-{j}.json stopped the plan ({rep["stopped"]}) but later rounds exist.', 'Remove the later round files; a stopped plan takes no more evidence.')
        _check(j in rf['outcomes'], f'result-{j}.json exists but round-{j}.jsonl does not.', f'Restore round-{j}.jsonl.')
        prev_rows, report = res['evidence'], rep
    if done:   # cheap integrity checks; look, explain and bundle re-verify the whole result
        last = load(ws / results[done])
        _check(last['report']['lineage']['evidence_digest'] == digest(last['evidence']) and last['report']['lineage']['plan_digest'] == frozen['plan_digest'],
               f'result-{done}.json does not match its own evidence or plan (it was edited).', 'Start a new workspace; results cannot be edited.')
        prior = load(ws / results[done - 1])['evidence'] if done > 1 else []
        given = load(ws / rf['outcomes'][done], True)
        expect = sorted(json.dumps(r, sort_keys=True) for r in last['evidence'])
        _check(expect in (sorted(json.dumps(r, sort_keys=True) for r in prior + given), sorted(json.dumps(r, sort_keys=True) for r in given)),
               f'round-{done}.jsonl is not the evidence result-{done}.json was made from (rows changed after the look).',
               'Recorded rows cannot change. Restore the original round file, or start a new workspace.')
    for j in list(rf['cases']) + list(rf['outcomes']):
        _check(j <= done + 1, f'Round {j} files exist before round {j - 1} has a result.', f'Remove the round {j} files; run rounds in order.')
    for j in rf['outcomes']:
        _check(j in rf['cases'], f'round-{j}.jsonl exists but round-{j}-cases.jsonl does not.', f'Re-create the case list with cases --round {j}.')
    stopped = report['stopped'] if report else None
    state['rounds'] = {'planned': K, 'done': done, 'stopped': stopped, 'verdict': report['verdict'] if report else None, 'choice': report['choice'] if report else None}
    if stopped in ('pass', 'block', 'exhausted'):
        last = results[done]
        words = {'pass': f"{report['choice']} passes: every clause holds at the error limit", 'block': 'the option tested fails a limit; more cases will not change this',
                 'exhausted': 'the planned rounds ran out without an answer (inconclusive)'}[stopped]
        finished_do_not = ['Do not run more rounds or reuse this plan: more evidence needs a new plan in a new workspace.',
                           'Do not describe a pass as more than it is: it is a bound at a 5% error limit, on traffic like the cases tested, not a promise about every case.']
        if not (ws / 'explanation.json').exists():
            return 0, _step('explain', 'agent', f'The decision is in after round {done}: {words}. Explain it in the person\'s terms.',
                            ['explain', '--result', last, '--out', 'explanation.json'], writes=['explanation.json'], do_not=finished_do_not)
        if not (ws / 'bundle.json').exists():
            return 0, _step('bundle', 'agent', 'Package the result for review (case ids, hashes, labels and costs; no prompts).',
                            ['bundle', '--result', last, '--out', 'bundle.json'], writes=['bundle.json'], do_not=finished_do_not)
        person = {'pass': f"Read explanation.json. Aver does not approve a change: the person decides whether to switch to {report['choice']}.",
                  'block': 'Read explanation.json: keep the current option, change the option tested, or change the requirements in a new workspace.',
                  'exhausted': 'Read explanation.json: keep the current option, make a new plan with more cases, or raise the bar, in a new workspace.'}[stopped]
        return 2, _step('finished', 'none', f'Finished after round {done} of {K}: {words}.', command=None, person=person, do_not=finished_do_not)
    j = done + 1
    _check(j <= K, f'Round {done} did not stop the plan but there are no rounds left.', 'Start a new workspace.')
    if j not in rf['cases']:
        why = 'The test is frozen; list the cases for round 1.' if j == 1 else f'Round {done} says keep testing ({report["verdict"]["reason_code"]}). List the cases for round {j}.'
        return 0, _step('cases', 'agent', why, ['cases', '--plan', 'frozen.json', '--round', str(j), '--out', f'round-{j}-cases.jsonl'],
                        writes=[f'round-{j}-cases.jsonl'], do_not=rounds_do_not)
    if j not in rf['outcomes']:
        cmd, template = None, True
        harness = ws / 'harness-command.txt'
        if harness.exists():
            line = next((l.strip() for l in harness.read_text(encoding='utf-8').splitlines() if l.strip() and not l.lstrip().startswith('#')), '')
            if line: cmd, template = line.replace('{cases}', f'round-{j}-cases.jsonl').replace('{out}', f'round-{j}.jsonl').replace('{round}', str(j)), False
        if cmd is None: cmd = f'<your harness> --cases round-{j}-cases.jsonl --out round-{j}.jsonl'
        n = sum(1 for l in (ws / rf['cases'][j]).read_text(encoding='utf-8').splitlines() if l.strip())
        return 0, _step('harness', 'harness', f'Run the {n} entries of round-{j}-cases.jsonl with your own harness and write one aver-outcomes/v2 row per entry to round-{j}.jsonl '
                        '(this round\'s rows only; see look --help for the row shape).', command=cmd, template=template,
                        writes=[f'round-{j}.jsonl'], do_not=rounds_do_not + ['Do not invent or edit outcomes: they must be what the pipeline did, labelled the same way as the logs.'])
    argv = ['look', '--plan', 'frozen.json', '--evidence', f'round-{j}.jsonl'] + (['--previous', results[done]] if done else []) + ['--out', f'result-{j}.json']
    return 0, _step('look', 'agent', f'Round {j} outcomes are in. Look at the evidence so far (rounds 1 to {j}).', argv, writes=[f'result-{j}.json'], do_not=rounds_do_not)


def next_text(data, code, wrap):
    s = data['step']
    out = [f"Stage: {s['stage']} (exit {code}: {({0: 'run the next command', 1: 'a person must decide first', 2: 'finished', 3: 'inconsistent workspace'})[code]})"]
    out.append(wrap('Why: ' + s['why']))
    if s['command']: out.append(('Template (fill in the <...>): ' if s['template'] else 'Next: ') + s['command'])
    for k, v in s['fill'].items(): out.append(wrap(f'{k}: {v}', '    '))
    if s['person_decides']: out.append(wrap('A person must decide: ' + s['person_decides']) if code in (1, 3) else wrap('Person: ' + s['person_decides']))
    if s['writes']: out.append('Writes: ' + ', '.join(s['writes']))
    out.append('Do not:')
    for d in s['do_not']: out.append(wrap(d, '  - '))
    return '\n'.join(out) + '\n'
