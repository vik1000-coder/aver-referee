"""Portable cost/reliability decision workflow; local assessments, never issuance.

Method and scope: docs/126. Binary independent cases, one frozen look, exact
Bonferroni upper tests; separate family-controlled lower tests for blocking.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from functools import lru_cache

METHOD = 'aver-decision-binary-v1'
ALPHA = .05


class DecisionError(ValueError):
    def __init__(self, code, path, message):
        super().__init__(message)
        self.code, self.path = code, path


def need(ok, code, path, message):
    if not ok:
        raise DecisionError(code, path, message)


def fields(obj, keys, path):
    need(isinstance(obj, dict), 'invalid_object', path, 'Expected an object.')
    need(set(obj) == set(keys.split()), 'invalid_fields', path,
         'Required fields: ' + keys + '. Unknown fields are not ignored.')


def number(x):
    return type(x) in (int, float) and math.isfinite(x) and abs(x) <= 2**53-1


def label(x):
    return isinstance(x, str) and 0 < len(x.strip()) <= 240


def valid_hash(x):
    return isinstance(x, str) and len(x) == 64 and all(c in '0123456789abcdef' for c in x)


def sequential_sum(values):
    # Match JavaScript left-to-right addition for portable artifact hashes.
    value = 0
    for item in values:
        value += item
    return value


def canonical(x):
    """Cross-language typed encoding; numbers use exact IEEE754 bytes, not rounding."""
    if x is None: return 'null'
    if isinstance(x, bool): return 'true' if x else 'false'
    if number(x): return 'n' + struct.pack('>d', 0. if x == 0 else float(x)).hex()
    if isinstance(x, str): return json.dumps(x, ensure_ascii=False, separators=(',', ':'))
    if isinstance(x, list): return '[' + ','.join(canonical(v) for v in x) + ']'
    if isinstance(x, dict):
        need(all(isinstance(k, str) and k.isascii() for k in x), 'invalid_key', '$', 'Object keys must be ASCII strings.')
        return '{' + ','.join(canonical(k)+':'+canonical(x[k]) for k in sorted(x)) + '}'
    raise DecisionError('invalid_value', '$', 'Only finite JSON values are accepted.')


def digest(x):
    return hashlib.sha256(canonical(x).encode('utf-8')).hexdigest()


def validate_spec(spec):
    fields(spec, 'schema scope candidates baseline strata definitions requirements monthly_uses consequence_usd budget_runs population_statement', 'spec')
    need(spec['schema'] == 'aver-decision-spec/v1', 'unsupported_schema', 'spec.schema', 'Expected aver-decision-spec/v1.')
    fields(spec['scope'], 'app checkpoint', 'spec.scope')
    fields(spec['definitions'], 'completion incident grader', 'spec.definitions')
    for prefix in ['scope', 'definitions']:
        need(all(label(v) for v in spec[prefix].values()), 'missing_definition', 'spec.'+prefix, 'Supply names and explicit outcome/grader definitions.')
    cs = spec['candidates']
    need(isinstance(cs, list) and 2 <= len(cs) <= 4, 'candidate_count', 'spec.candidates', 'Provide two to four pipeline configurations.')
    for c in cs:
        fields(c, 'id name digest', 'spec.candidates[]')
        need(label(c['id']) and c['id'].isascii() and label(c['name']) and valid_hash(c['digest']), 'candidate_identity', 'spec.candidates[]', 'Each pipeline needs a unique ID, name and SHA256 configuration digest.')
    need(len({c['id'] for c in cs}) == len(cs) and len({c['digest'] for c in cs}) == len(cs) and spec['baseline'] in [c['id'] for c in cs], 'candidate_identity', 'spec.baseline', 'Configurations must be distinct and baseline must name a candidate.')
    ss = spec['strata']
    need(isinstance(ss, dict) and 1 <= len(ss) <= 8, 'invalid_strata', 'spec.strata', 'Provide one to eight disjoint populations.')
    for s, entry in ss.items():
        fields(entry, 'weight failure_limit', 'spec.strata.'+s)
        need(label(s) and s.isascii() and number(entry['weight']) and 0 < entry['weight'] <= 1, 'invalid_weight', 'spec.strata.'+s, 'Population names and positive traffic weights are required.')
        f = entry['failure_limit']
        need(f is None or (number(f) and 0 <= f <= 1), 'invalid_limit', 'spec.strata.'+s, 'Failure limit must be null or between zero and one.')
    need(abs(sequential_sum(v['weight'] for v in ss.values())-1) < 1e-10, 'invalid_weight', 'spec.strata', 'Traffic weights must sum to one; evaluation proportions are not traffic weights.')
    fields(spec['requirements'], 'failure_limit incident_limit', 'spec.requirements')
    need(all(number(v) and 0 <= v <= 1 for v in spec['requirements'].values()), 'invalid_limit', 'spec.requirements', 'Explicit rate limits between zero and one are required.')
    for k in ['monthly_uses', 'budget_runs']:
        need(number(spec[k]) and int(spec[k]) == spec[k] and 1 <= spec[k] <= 1000000, 'invalid_budget', 'spec.'+k, 'Use an integer between 1 and 1,000,000.')
    v = spec['consequence_usd']
    need(v is None or (number(v) and 0 <= v <= 1000000), 'invalid_consequence', 'spec.consequence_usd', 'Use a nonnegative dollar assumption or null for unknown.')
    need(label(spec['population_statement']), 'missing_population', 'spec.population_statement', 'Describe the sampling population and traffic-weight assumptions.')
    digest(spec)
    return spec


def validate_rows(spec, rows, phase, candidates=None):
    validate_spec(spec)
    need(isinstance(rows, list) and len(rows) <= 1000000, 'invalid_evidence', 'evidence', 'Evidence must be an array of at most 1,000,000 episodes.')
    configs = {c['id']: c['digest'] for c in spec['candidates']}
    expected = candidates if candidates is not None else list(configs)
    seen, identities, contents = set(), {}, {}
    for i, r in enumerate(rows):
        p = f'evidence[{i}]'
        fields(r, 'schema case_id case_sha256 candidate_id configuration_sha256 stratum phase origin completed incident cost_usd', p)
        need(r['schema'] == 'aver-outcomes/v1' and r['phase'] == phase, 'incorrect_phase', p, 'Use aver-outcomes/v1 evidence from the requested phase.')
        c, key, content, s = r['candidate_id'], r['case_id'], r['case_sha256'], r['stratum']
        need(isinstance(c, str) and c in expected and r['configuration_sha256'] == configs[c], 'configuration_changed', p, 'Evidence must match the planned candidate digest.')
        need(label(key) and key.isascii() and valid_hash(content) and isinstance(s, str) and s in spec['strata'], 'case_identity', p, 'Case ID, SHA256 content digest and declared population are required.')
        need((c, key) not in seen, 'dependent_repeat', p, 'One episode per independent case and candidate is supported; duplicates/repeats cannot increase evidence.')
        seen.add((c, key))
        need(key not in identities or identities[key] == (content, s), 'case_identity', p, 'Case content or population changed across candidates.')
        need(content not in contents or contents[content] == key, 'duplicate_content', p, 'Identical case content cannot appear under multiple IDs.')
        identities[key], contents[content] = (content, s), key
        need(r['origin'] in ['synthetic', 'external', 'customer'], 'invalid_origin', p, 'Declare synthetic, external or customer provenance.')
        for metric in ['completed', 'incident']:
            need(r[metric] is None or type(r[metric]) is bool, 'unsupported_outcome', p+'.'+metric, 'Outcomes must be boolean or null; repeat averages are unsupported.')
        need(r['cost_usd'] is None or (number(r['cost_usd']) and 0 <= r['cost_usd'] <= 1000000), 'invalid_cost', p+'.cost_usd', 'Use complete episode cost in dollars or null, never an invented zero.')
    need(len({r['origin'] for r in rows}) <= 1, 'mixed_origins', 'evidence', 'Keep different evidence origins in separate studies.')
    return sorted(rows, key=lambda r: (r['stratum'], r['case_id'], r['candidate_id']))


@lru_cache(maxsize=50000)
def upper(n, k, alpha):
    if not n: return 1.
    if k == n: return 1.
    if k == 0: return -math.expm1(math.log(alpha)/n)
    low, high = k/n, 1.
    for _ in range(42):
        p = (low+high)/2
        logterm = math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)+k*math.log(p)+(n-k)*math.log1p(-p)
        term = total = 1.
        for j in range(k, 0, -1):
            term *= j/(n-j+1)*(1-p)/p
            total += term
            if term < total*1e-16: break
        if math.exp(logterm+math.log(total)) > alpha: low = p
        else: high = p
    return high


def zero_floor(limit, alpha):
    if limit == 0: return None
    if limit == 1: return 1
    raw = math.log(alpha)/math.log1p(-limit)
    if not math.isfinite(raw) or raw > 2**53-1:
        return None
    n = max(1, math.ceil(raw))
    while upper(n, 0, alpha) > limit: n += 1
    return n


def explore(spec, rows):
    rows = validate_rows(spec, rows, 'pilot')
    groups = {c['id']: {s: [r for r in rows if r['candidate_id'] == c['id'] and r['stratum'] == s] for s in spec['strata']} for c in spec['candidates']}
    for s in spec['strata']:
        sets = [{r['case_id'] for r in groups[c['id']][s]} for c in spec['candidates']]
        need(bool(sets[0]) and all(x == sets[0] for x in sets), 'incomplete_pilot', 'evidence', 'Pilot needs the same independent cases for every candidate in each population.')
    summaries = []
    for c in spec['candidates']:
        failure, cost, known, eligible = 0., 0., True, True
        for s in sorted(spec['strata']):
            setting = spec['strata'][s]
            rs = groups[c['id']][s]; n = len(rs)
            f = sequential_sum(r['completed'] is not True for r in rs)/n
            incident = sequential_sum(r['incident'] is not False for r in rs)/n
            failure += setting['weight']*f
            known = known and all(r['cost_usd'] is not None and r['completed'] is not None and r['incident'] is not None for r in rs)
            cost += setting['weight']*sequential_sum(r['cost_usd'] or 0 for r in rs)/n
            eligible = eligible and incident <= spec['requirements']['incident_limit'] and (setting['failure_limit'] is None or f <= setting['failure_limit'])
        score = cost + spec['consequence_usd']*failure if known and spec['consequence_usd'] is not None else None
        summaries.append({'candidate_id': c['id'], 'observed_eligible': eligible and failure <= spec['requirements']['failure_limit'], 'failure_rate': failure, 'execution_cost_usd': cost if known else None, 'score_usd': score})
    ranked = sorted(summaries, key=lambda c: (not c['observed_eligible'], c['score_usd'] is None, c['score_usd'] or 0, c['candidate_id']))
    return {'schema': 'aver-exploration/v1', 'spec': spec, 'pilot': rows, 'ranking': ranked, 'scope': 'exploration_only'}


def make_plan(spec, rows, finalists):
    ex = explore(spec, rows)
    need(type(finalists) is int and 1 <= finalists <= len(spec['candidates']), 'invalid_finalists', 'finalists', 'Select one to the full candidate count.')
    need(all(r['score_usd'] is not None for r in ex['ranking']) or finalists == len(spec['candidates']), 'missing_economics', 'spec.consequence_usd', 'Screening needs complete pilot costs, outcomes and a declared failure consequence. Compare all candidates instead, or repair these inputs.')
    ns = len(spec['strata']); spent = len(rows)
    n = max(0, (int(spec['budget_runs'])-spent)//(finalists*ns))
    a = ALPHA/(finalists*ns)
    limits = [*spec['requirements'].values(), *[v['failure_limit'] for v in spec['strata'].values() if v['failure_limit'] is not None]]
    floors = [zero_floor(x, a) for x in limits]
    minimum = None if None in floors else max(floors)
    selected = [r['candidate_id'] for r in ex['ranking'][:finalists]]
    manifest = [{'case_id': f'confirmation-{si}-{i}', 'stratum': s} for si, s in enumerate(sorted(spec['strata'])) for i in range(n)]
    body = {'schema': 'aver-test-plan/v1', 'method_version': METHOD, 'exploration': ex, 'selected': selected,
            'pilot_executions': spent, 'cases_per_population': n, 'confirmation_cases': manifest,
            'confirmation_executions': n*ns*finalists, 'minimum_cases_per_population': minimum,
            'feasible': minimum is not None and n >= minimum,
            'scope': 'one_frozen_look_independent_binary_cases'}
    return body


def freeze(plan):
    fields(plan, 'schema method_version exploration selected pilot_executions cases_per_population confirmation_cases confirmation_executions minimum_cases_per_population feasible scope', 'plan')
    ex = plan['exploration']
    need(isinstance(ex, dict) and isinstance(plan['selected'], list), 'invalid_plan', 'plan', 'Supply a plan created by Aver.')
    need(digest(make_plan(ex.get('spec'), ex.get('pilot'), len(plan['selected']))) == digest(plan), 'plan_changed', 'plan', 'Plan does not reproduce from its specification and pilot.')
    need(plan['feasible'], 'infeasible_plan', 'plan', 'This budget cannot support these requirements even with zero failures. Increase the budget or test fewer candidates.')
    return {'schema': 'aver-frozen-plan/v1', 'plan': plan, 'plan_digest': digest(plan)}


def evaluate(frozen, rows, previous=None):
    fields(frozen, 'schema plan plan_digest', 'frozen_plan')
    need(frozen['schema'] == 'aver-frozen-plan/v1' and freeze(frozen['plan']) == frozen, 'plan_changed', 'frozen_plan', 'The frozen plan changed.')
    plan = frozen['plan']; spec = plan['exploration']['spec']; selected = plan['selected']
    rows = validate_rows(spec, rows, 'confirmation', selected)
    pilot = plan['exploration']['pilot']
    need(not ({r['case_id'] for r in rows} & {r['case_id'] for r in pilot}) and not ({r['case_sha256'] for r in rows} & {r['case_sha256'] for r in pilot}), 'pilot_leakage', 'evidence', 'Pilot cases or content cannot be confirmation evidence.')
    need(not rows or rows[0]['origin'] == pilot[0]['origin'], 'mixed_origins', 'evidence', 'Pilot and confirmation origins must agree.')
    expected = {c['case_id']: c['stratum'] for c in plan['confirmation_cases']}
    need(all(expected.get(r['case_id']) == r['stratum'] for r in rows), 'unplanned_case', 'evidence', 'Collect only cases assigned in the frozen manifest; additional looks require a new plan.')
    if previous is not None:
        verified = verify_result(previous)
        need(verified['frozen_plan'] == frozen, 'previous_plan_changed', 'previous', 'The previous receipt belongs to a different plan.')
        current = {(r['candidate_id'], r['case_id']): r for r in rows}
        need(all(current.get((r['candidate_id'], r['case_id'])) == r for r in verified['evidence']), 'earlier_evidence_changed', 'evidence', 'Earlier confirmation rows must remain unchanged during resume.')
    complete = len(rows) == plan['confirmation_executions']
    ns, q = len(spec['strata']), len(selected)
    au, al = ALPHA/(q*ns), ALPHA/(q*ns*2)
    summaries = []
    for c in selected:
        if not complete:
            summaries.append({'candidate_id': c, 'state': 'refuse', 'clauses': [], 'execution_usd_per_use': None, 'modeled_usd_per_month': None})
            continue
        clauses, average, lower_avg, estimate_avg, unknown_total, cost, cost_known = [], 0., 0., 0., 0, 0., True
        for s in sorted(spec['strata']):
            setting = spec['strata'][s]
            rs = [r for r in rows if r['candidate_id'] == c and r['stratum'] == s]; n = len(rs)
            for metric in ['completed', 'incident']:
                known_bad = sequential_sum(r[metric] is (False if metric == 'completed' else True) for r in rs)
                unknown = sequential_sum(r[metric] is None for r in rs); unknown_total += unknown
                lo = 1-upper(n, n-known_bad, al) if n else 0.
                hi = upper(n, known_bad+unknown, au)
                est = known_bad/n if n and not unknown else None
                limit = setting['failure_limit'] if metric == 'completed' else spec['requirements']['incident_limit']
                if metric == 'completed':
                    average += setting['weight']*hi; lower_avg += setting['weight']*lo
                    estimate_avg += setting['weight']*(est or 0)
                if limit is not None:
                    state = 'refuse' if not complete or unknown else 'pass' if hi <= limit else 'block' if lo > limit else 'refuse'
                    clauses.append({'key': s+':'+metric, 'population': s, 'metric': 'failure' if metric == 'completed' else 'incident', 'estimate': est, 'lower': lo, 'upper': hi, 'limit': limit, 'cases': n, 'unknown': unknown, 'state': state})
            cost_known = cost_known and n > 0 and all(r['cost_usd'] is not None for r in rs)
            cost += setting['weight']*sequential_sum(r['cost_usd'] or 0 for r in rs)/max(1,n)
        lim = spec['requirements']['failure_limit']
        avg_state = 'refuse' if not complete or unknown_total else 'pass' if average <= lim else 'block' if lower_avg > lim else 'refuse'
        clauses.insert(0, {'key': 'average:failure', 'population': 'all', 'metric': 'failure', 'estimate': None if unknown_total else estimate_avg, 'lower': lower_avg, 'upper': average, 'limit': lim, 'cases': sequential_sum(r['candidate_id'] == c for r in rows), 'unknown': unknown_total, 'state': avg_state})
        state = 'refuse' if not complete else 'block' if any(v['state'] == 'block' for v in clauses) else 'pass' if all(v['state'] == 'pass' for v in clauses) else 'refuse'
        score = cost + spec['consequence_usd']*estimate_avg if complete and cost_known and not unknown_total and spec['consequence_usd'] is not None else None
        summaries.append({'candidate_id': c, 'state': state, 'clauses': clauses, 'execution_usd_per_use': cost if cost_known and complete else None, 'modeled_usd_per_month': None if score is None else score*spec['monthly_uses']})
    supported = [c for c in summaries if c['state'] == 'pass']
    ranked = sorted([c for c in supported if c['modeled_usd_per_month'] is not None], key=lambda c: (c['modeled_usd_per_month'], c['candidate_id']))
    choice = ranked[0]['candidate_id'] if ranked else None
    state = 'pass' if choice else 'review' if supported else 'block' if all(c['state'] == 'block' for c in summaries) else 'refuse'
    reason = 'lowest_observed_cost_supported' if choice else 'missing_economics' if supported else 'requirements_violated' if state == 'block' else 'insufficient_independent_cases' if not complete else 'requirements_unresolved'
    report = {'method_version': METHOD, 'verdict': {'state': state, 'reason_code': reason}, 'choice': choice,
              'candidates': summaries, 'savings': {'status': 'not_evaluated'}, 'release_eligibility': 'not_evaluated',
              'progress': {'received_executions': len(rows), 'required_executions': plan['confirmation_executions'], 'complete': complete},
              'lineage': {'plan_digest': frozen['plan_digest'], 'evidence_digest': digest(rows)},
              'origin': pilot[0]['origin'], 'scope': 'local_assessment_of_declared_population',
              'next_actions': [{'code': 'review_local_result' if state == 'pass' else 'repair_cost_assumptions' if supported else 'change_candidate_or_requirements' if state == 'block' else 'complete_frozen_collection' if not complete else 'review_missing_labels_or_design_new_study', 'actor': 'engineer', 'automatic_retry': False}],
              'limitations': ['Local assessment; not a signed record or release authorization.', 'Sampling and grader validity are declared, not independently verified.', 'Exact upper tests control unsupported selection for one frozen question; lower violation tests have a separate error allowance.', 'Savings have not been confirmed against current.']}
    return {'schema': 'aver-decision-result/v1', 'frozen_plan': frozen, 'evidence': rows, 'report': report}


def verify_result(result):
    fields(result, 'schema frozen_plan evidence report', 'result')
    need(result['schema'] == 'aver-decision-result/v1', 'unsupported_schema', 'result', 'Expected a decision result.')
    expected = evaluate(result['frozen_plan'], result['evidence'])
    # Numerical bounds can differ slightly across libm and browser math.
    def same(a, b):
        if number(a) and number(b): return abs(a-b) <= 1e-9*max(1,abs(a),abs(b))
        if isinstance(a, dict) and isinstance(b, dict): return set(a) == set(b) and all(same(a[k],b[k]) for k in a)
        if isinstance(a, list) and isinstance(b, list): return len(a) == len(b) and all(same(x,y) for x,y in zip(a,b))
        return type(a) is type(b) and a == b
    need(same(expected, result), 'result_changed', 'result', 'Result does not reproduce from its evidence. Recompute it.')
    return expected


def demo(scenario='document-processing', outcome='pass'):
    need(scenario in ['document-processing', 'support-routing'], 'unknown_scenario', 'scenario', 'Choose document-processing or support-routing.')
    need(outcome in ['pass', 'refuse', 'block'], 'unknown_outcome', 'outcome', 'Choose pass, refuse or block.')
    spec = {'schema': 'aver-decision-spec/v1', 'scope': {'app': scenario, 'checkpoint': 'Completed task'},
            'candidates': [{'id': c, 'name': name, 'digest': digest({'scenario': scenario, 'configuration': c})} for c,name in [('current','Today’s model'),('lean','Cheaper model'),('balanced','Mid-price model'),('guarded','Careful model')]],
            'baseline': 'current', 'strata': {'ordinary': {'weight': .9, 'failure_limit': None}, 'ambiguous': {'weight': .07, 'failure_limit': .1}, 'tool_failure': {'weight': .03, 'failure_limit': .1}},
            'definitions': {'completion': 'Task output matches the approved expected fields or routing destination.', 'incident': 'The pipeline changed an external record incorrectly.', 'grader': 'Deterministic synthetic fixture; not customer evidence.'},
            'requirements': {'failure_limit': .032, 'incident_limit': .05}, 'monthly_uses': 10000, 'consequence_usd': 1., 'budget_runs': 1200,
            'population_statement': 'Synthetic illustration with assumed 90% ordinary, 7% ambiguous and 3% tool-failure traffic.'}
    def row(c,s,key,phase,bad=False):
        return {'schema': 'aver-outcomes/v1', 'case_id': key, 'case_sha256': digest({'scenario': scenario, 'case': key}), 'candidate_id': c['id'], 'configuration_sha256': c['digest'], 'stratum': s, 'phase': phase, 'origin': 'synthetic', 'completed': not bad, 'incident': False, 'cost_usd': {'current': .03, 'lean': .008, 'balanced': .015, 'guarded': .025}[c['id']]}
    pilot = [row(c,s,f'pilot-{s}-{i}','pilot') for s in sorted(spec['strata']) for i in range(20) for c in spec['candidates']]
    plan = make_plan(spec,pilot,1); frozen = freeze(plan)
    chosen = next(c for c in spec['candidates'] if c['id'] == plan['selected'][0])
    confirmation = [row(chosen,r['stratum'],r['case_id'],'confirmation', outcome == 'block' and i % 2 == 0) for i,r in enumerate(plan['confirmation_cases'])]
    if outcome == 'refuse': confirmation = confirmation[:20]
    return {'spec': spec, 'pilot': pilot, 'plan': plan, 'frozen_plan': frozen, 'confirmation': confirmation, 'result': evaluate(frozen,confirmation)}
