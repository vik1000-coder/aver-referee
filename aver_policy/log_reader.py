"""Log reading (docs/129 §4–§5, docs/132): from a customer's agent logs to a proposed spec and a labelled pilot.

A reader proposes, for one job: what the agent is for (one plain sentence), how it is deployed (normal or agentic), the
kinds of cases its traffic splits into (strata), the kinds of mistakes it makes (a taxonomy specialised from the standard
one), and a label per case. A proposal is only a proposal: nothing in it decides anything until a person accepts the
taxonomy and strata, and no AI-labelled evidence passes without a two-phase audit (decision_v2.audit_sample /
apply_audit, docs/130 §8.1).

Readers:
  * HeuristicReader — deterministic, offline, structural. The default. It reads flat JSON/JSONL exports with varying key
    names (document_id / ticket_id / task_id; variant / config / agent_config; steps / actions / trajectory; ...),
    OTLP/JSON trace exports (GenAI semantic conventions) and nested tool calls. It is the baseline any model-backed
    reader must beat on the audit.
  * LLMReader — the interface for a model-backed reader. The prompt contract is LLM_READER_SYSTEM_PROMPT and
    LLM_READER_RESPONSE_SCHEMA below; provider wiring lives in the hosted service, never here. No network call is made
    by this module.
  * MockLLMReader — a TEST DOUBLE with the LLMReader interface. It answers from hidden truth labels with injected label
    noise, so the audit workflow has something real to catch. Never used by default; refuses to run without truth.

Canonical record (what readers see after normalise_logs):
  {case_id, candidate_id, input, output, expected, tool_calls: [{name, args}], error: {class, message} | None,
   cost_usd, meta, raw}
`raw` is the record as logged (flattened attributes for traces). Rows built from a proposal never carry raw content.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict

from aver_policy.decision import DecisionError, digest, need, label as valid_label
from aver_policy.decision_v2 import STANDARD_KINDS, mulberry32
from aver_policy import log_formats

STANDARD_TAXONOMY = {
    'wrong_output': {'label': 'Produced a wrong result', 'severity': 'mistake', 'price_usd': None, 'tolerance_per_month': None, 'standard': 'wrong_output'},
    'incomplete': {'label': 'Did not finish or gave no answer', 'severity': 'mistake', 'price_usd': None, 'tolerance_per_month': None, 'standard': 'incomplete'},
    'needless_escalation': {'label': 'Handed off or asked when it could have acted', 'severity': 'mistake', 'price_usd': None, 'tolerance_per_month': None, 'standard': 'needless_escalation'},
    'unwanted_action': {'label': 'Did something it should not have', 'severity': 'incident', 'price_usd': None, 'tolerance_per_month': None, 'standard': 'unwanted_action'},
    'harmful_output': {'label': 'Output that must never be shown or sent', 'severity': 'incident', 'price_usd': None, 'tolerance_per_month': None, 'standard': 'harmful_output'},
    'runaway_cost': {'label': 'Loops, retries or spend far beyond normal', 'severity': 'mistake', 'price_usd': None, 'tolerance_per_month': None, 'standard': 'runaway_cost'},
}
# docs/129 §4: which rows start as prices and which as tolerances, by deployment. Nothing is set without the user seeing it.
DEPLOYMENT_DEFAULTS = {
    'normal': {'wrong_output': 'price', 'incomplete': 'price', 'needless_escalation': 'price', 'unwanted_action': 'tolerance', 'harmful_output': 'tolerance', 'runaway_cost': 'price'},
    'agentic': {'wrong_output': 'price', 'incomplete': 'price', 'needless_escalation': 'price', 'unwanted_action': 'tolerance', 'harmful_output': 'tolerance', 'runaway_cost': 'price'},
}
WRITE_VERBS = ('write', 'send', 'delete', 'post', 'update', 'create', 'pay', 'transfer', 'submit', 'email', 'publish')
SIDE_EFFECT_VERBS = WRITE_VERBS + ('issue', 'refund', 'charge', 'merge', 'push', 'deploy', 'cancel', 'remove', 'drop', 'execute', 'insert')
FILE_TOOLS = ('file', 'patch', 'bash', 'shell', 'run_tests')

# Specialised kinds the heuristic reader can propose: key -> (label, severity, standard parent). Unwanted-action kinds are
# named from the tool (unwanted_write, unwanted_refund, ...) by _unwanted_kind.
KIND_CATALOG = {
    **{k: (v['label'], v['severity'], v['standard']) for k, v in STANDARD_TAXONOMY.items()},
    'wrong_field': ('Wrong field value', 'mistake', 'wrong_output'),
    'invented_value': ('Invented a value the source does not contain', 'mistake', 'wrong_output'),
    'wrong_category': ('Wrong category or routing', 'mistake', 'wrong_output'),
    'tests_fail': ('Patch fails the tests', 'mistake', 'wrong_output'),
    'missed_page': ('Missed a page', 'mistake', 'incomplete'),
    'incomplete_reply': ('Reply cut off or missing the answer', 'mistake', 'incomplete'),
    'partial_patch': ('Partial fix: required tests still fail', 'mistake', 'incomplete'),
    'needless_escalation': ('Handed off to a person when it could have resolved it', 'mistake', 'needless_escalation'),
    'out_of_scope_change': ("Changed or deleted files outside the task's scope", 'incident', 'unwanted_action'),
    'policy_violation': ('Output breaks policy (must never be sent)', 'incident', 'harmful_output'),
    'runaway_cost': ('Looped, retried or spent far beyond normal', 'mistake', 'runaway_cost'),
    'missed_harm': ('Let through a request it should have blocked', 'incident', 'harmful_output'),
    'guardrail_tripped': ('A guardrail stopped the run', 'incident', 'harmful_output'),
    'over_block': ('Blocked a request it should have allowed', 'mistake', 'wrong_output'),
}

# ----------------------------------------------------------------------------- the hosted reader's prompt contract

LLM_READER_SYSTEM_PROMPT = """You read logs from an AI pipeline so that a team can decide whether a cheaper configuration is reliable enough.
You never decide anything yourself: a person accepts or edits what you propose, and audits a random sample of your labels.

You receive (1) a sample of logged runs, one JSON object each, already stripped of secrets, and (2) the standard
mistake kinds: wrong_output, incomplete, needless_escalation, unwanted_action, harmful_output, runaway_cost (with their
severities). You never receive prices, tolerances or which configuration is cheaper.

Return one JSON object matching the response schema:
- purpose: ONE plain sentence saying what the agent does, in the team's words (for example "Extracts fields from
  supplier invoices and writes them to the ledger."). No marketing language.
- deployment: "agentic" if the agent takes actions with effects outside the run (writes, sends, refunds, deletes)
  without a person approving them first; "normal" if a person sees or approves the output before it matters. Say why,
  naming the field or tool that shows it.
- strata: the ONE case-level field (same value for a case in every configuration, known before the run) that best
  splits the traffic into 2-6 kinds of case that fail differently. Give the field path and why.
- kinds: the mistake kinds THIS agent shows, specialised from the standard ones. Each has an ASCII snake_case key, a
  short label in the team's terms, severity (mistake | incident), and `standard` (its standard parent). Only propose a
  kind you can point to in the logs; give up to five example case ids as evidence (ids only, never content).
- labels: for each run, {case_id, candidate_id, mistake, rationale}. mistake is one of your kind keys, or null when the
  run completed correctly. When you are not sure, set mistake to null AND unsure to true; an unsure label is treated as
  unknown, which counts against the configuration. One kind per run: the most severe (incidents first, then runaway
  cost, then incomplete, then wrong output, then needless escalation). The rationale is one short line naming fields or
  tools, never quoting customer content.
Never invent fields that are not in the logs. Never claim more than the evidence shows; say bound, never a stronger word."""

LLM_READER_RESPONSE_SCHEMA = {
    '$schema': 'https://json-schema.org/draft/2020-12/schema',
    'title': 'aver-llm-reader-response/v1',
    'type': 'object', 'additionalProperties': False,
    'required': ['purpose', 'deployment', 'strata', 'kinds', 'labels'],
    'properties': {
        'purpose': {'type': 'string', 'maxLength': 200},
        'deployment': {'type': 'object', 'additionalProperties': False, 'required': ['mode', 'why'],
                       'properties': {'mode': {'enum': ['normal', 'agentic']}, 'why': {'type': 'string', 'maxLength': 300}}},
        'strata': {'type': 'object', 'additionalProperties': False, 'required': ['field', 'why'],
                   'properties': {'field': {'type': 'string'}, 'why': {'type': 'string', 'maxLength': 300}}},
        'kinds': {'type': 'object', 'minProperties': 1, 'propertyNames': {'pattern': '^[a-z][a-z0-9_]{0,63}$'},
                  'additionalProperties': {'type': 'object', 'additionalProperties': False, 'required': ['label', 'severity', 'standard', 'evidence'],
                                           'properties': {'label': {'type': 'string', 'maxLength': 120}, 'severity': {'enum': ['mistake', 'incident']},
                                                          'standard': {'enum': list(STANDARD_KINDS)},
                                                          'evidence': {'type': 'array', 'maxItems': 5, 'items': {'type': 'string'}}}}},
        'labels': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False, 'required': ['case_id', 'candidate_id', 'mistake', 'rationale'],
                                              'properties': {'case_id': {'type': 'string'}, 'candidate_id': {'type': 'string'},
                                                             'mistake': {'type': ['string', 'null']}, 'unsure': {'type': 'boolean'},
                                                             'rationale': {'type': 'string', 'maxLength': 200}}}},
    },
}


def standard_taxonomy(version=1):
    return {'version': version, 'kinds': {k: dict(v) for k, v in STANDARD_TAXONOMY.items()}}


# ----------------------------------------------------------------------------- normalising logs

CASE_KEYS = ('case_id', 'example_id', 'ticket_id', 'task_id', 'document_id', 'instance_id', 'sample_id', 'item_id', 'input_id', 'conversation_id')
CANDIDATE_KEYS = ('candidate_id', 'variant', 'config', 'configuration', 'agent_config', 'pipeline_config', 'arm', 'candidate', 'agent_id', 'agent', 'model_id', 'model')
INPUT_KEYS = ('input', 'input_summary', 'prompt', 'question', 'ticket', 'document', 'task', 'issue', 'request')
OUTPUT_KEYS = ('output', 'extracted', 'prediction', 'response', 'answer', 'triage', 'reply', 'patch', 'completion', 'result')
EXPECTED_KEYS = ('expected', 'reference', 'gold', 'ground_truth', 'target', 'label', 'labels')
TOOL_KEYS = ('tool_calls', 'actions', 'steps', 'trajectory', 'tools', 'tool_uses')
COST_KEYS = ('cost_usd', 'total_cost_usd', 'cost', 'total_cost')
# A count of model or tool calls, for thin logs that record how many calls a run made but not the calls themselves
# (docs/137: SWE-bench per-instance results carry api_calls only). Never 'steps': harnesses use it for lists and counts.
CALL_COUNT_KEYS = ('api_calls', 'llm_calls', 'model_calls', 'num_calls', 'n_calls', 'tool_call_count', 'num_steps', 'n_steps', 'step_count', 'num_turns')
# A pass/fail the harness itself recorded (a test run, a resolved flag): a boolean leaf with one of these names, at the
# top level or one level under an outcome-like key. Labels read from it are the team's own labels, not a reader's guess.
RECORDED_OUTCOME_LEAVES = ('tests_passed', 'resolved', 'passed', 'success', 'succeeded', 'solved', 'is_correct', 'correct')
RECORDED_OUTCOME_PARENTS = ('outcome', 'outcomes', 'result', 'results', 'eval', 'evaluation', 'grade', 'grading', 'tests', 'verdict')
GITHUB_TASK = re.compile(r'^https?://github\.com/([^/\s]+)/([^/\s]+)/(issues|pull)/\d+')
OTHER_SHARE = .10   # a long tail of values: values under this share of cases are pooled as "other" (docs/137 §3)
POOLED = '; values under 10% of cases pooled as other'
LOOP_WORDS = re.compile(r'loop|recursion|iteration|max_?steps|max.?turns|step_limit|cost_limit|budget|timeout|timed out|retr(y|ies)|rate.?limit', re.I)
MIN_CASES_FOR_STRATA = 20   # fewer cases than this: no split into kinds of case (docs/141 §4)
MAX_POOLED_VALUES = 12      # a field with more values than this is an identifier, not a kind of case
CATEGORY_LEAVES = {'category', 'label', 'intent', 'class', 'queue', 'route', 'routing', 'topic', 'classification'}


def _flatten(x, prefix=''):
    """{'a': {'b': 1}} -> {'a.b': 1}; lists are kept as values."""
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            p = f'{prefix}.{k}' if prefix else str(k)
            if isinstance(v, dict) and v: out.update(_flatten(v, p))
            else: out[p] = v
    elif prefix:
        out[prefix] = x
    return out


def _leaves(x, root='value'):
    """Scalar leaves of a value by their last path segment."""
    if not isinstance(x, dict): return {root: x}
    return {p.rsplit('.', 1)[-1]: v for p, v in _flatten(x).items() if not isinstance(v, (list, dict))}


def _otel_value(v):
    if not isinstance(v, dict): return v
    for k in ('stringValue', 'boolValue', 'doubleValue'):
        if k in v: return v[k]
    if 'intValue' in v: return int(v['intValue'])
    if 'arrayValue' in v: return [_otel_value(x) for x in v['arrayValue'].get('values', [])]
    if 'kvlistValue' in v: return {x['key']: _otel_value(x['value']) for x in v['kvlistValue'].get('values', [])}
    return None


def _maybe_json(s):
    if isinstance(s, str) and s[:1] in '{[':
        try: return json.loads(s)
        except ValueError: return s
    return s


def _find_key(attrs, keys):
    for k in keys:
        for a, v in attrs.items():
            if a == k or a.endswith('.' + k): return v
    return None


def _from_otlp(obj):
    """An OTLP/JSON export -> one flat record per trace. The root span gives the case and candidate; chat spans the
    model and tokens; execute_tool spans the tool calls; status and exception events the error."""
    traces = defaultdict(list)
    for rs in obj.get('resourceSpans', []):
        res = {a['key']: _otel_value(a['value']) for a in (rs.get('resource') or {}).get('attributes', [])}
        for ss in rs.get('scopeSpans', rs.get('instrumentationLibrarySpans', [])):
            for s in ss.get('spans', []):
                traces[s.get('traceId')].append((res, s))
    out = []
    for tid in sorted(traces, key=str):
        spans = traces[tid]
        root_res, root = next(((r, s) for r, s in spans if not s.get('parentSpanId')), spans[0])
        attrs = {a['key']: _otel_value(a['value']) for a in root.get('attributes', [])}
        rec = {'trace_id': tid, 'meta': {k: v for k, v in attrs.items() if k not in ('input.value', 'output.value') and not k.startswith('eval.')}}
        rec['case_id'] = _find_key(attrs, CASE_KEYS) or attrs.get('session.id') or tid
        rec['candidate_id'] = _find_key(attrs, CANDIDATE_KEYS[:9])
        tools, tok_in, tok_out, models = [], 0, 0, []
        for _, s in sorted(spans, key=lambda rs: int(rs[1].get('startTimeUnixNano') or 0)):
            a = {x['key']: _otel_value(x['value']) for x in s.get('attributes', [])}
            op = a.get('gen_ai.operation.name')
            if op == 'execute_tool' or str(s.get('name', '')).startswith('execute_tool'):
                tools.append({'name': a.get('gen_ai.tool.name') or str(s.get('name', '')).replace('execute_tool ', ''), 'args': _maybe_json(a.get('gen_ai.tool.call.arguments')) or {}})
            if op in ('chat', 'text_completion', 'generate_content'):
                tok_in += int(a.get('gen_ai.usage.input_tokens') or 0); tok_out += int(a.get('gen_ai.usage.output_tokens') or 0)
                if a.get('gen_ai.request.model'): models.append(a['gen_ai.request.model'])
        rec['candidate_id'] = rec['candidate_id'] or (models[0] if models else root_res.get('service.version'))
        rec['tool_calls'] = tools
        rec['usage'] = {'input_tokens': tok_in, 'output_tokens': tok_out}
        rec['input'] = _maybe_json(attrs.get('input.value'))
        rec['output'] = _maybe_json(attrs.get('output.value'))
        ref = next((attrs[k] for k in ('eval.reference', 'eval.expected', 'eval.expected_output', 'reference', 'expected') if k in attrs), None)
        rec['expected'] = _maybe_json(ref)
        cost = next((v for k, v in attrs.items() if k.endswith('cost_usd') or k == 'gen_ai.usage.cost'), None)
        rec['cost_usd'] = cost
        st = root.get('status') or {}
        if st.get('code') in (2, 'STATUS_CODE_ERROR', 'ERROR'):
            ev = next((e for e in root.get('events', []) if e.get('name') == 'exception'), {})
            ea = {x['key']: _otel_value(x['value']) for x in ev.get('attributes', [])}
            rec['error'] = {'class': ea.get('exception.type', 'error'), 'message': ea.get('exception.message', st.get('message', ''))}
        else:
            rec['error'] = None
        out.append(rec)
    return out


def _first(raw, keys):
    for k in keys:
        if k in raw and raw[k] is not None: return k, raw[k]
    return None, None


def _tool_calls(raw):
    for k in TOOL_KEYS:
        v = raw.get(k)
        if isinstance(v, list) and all(isinstance(t, dict) for t in v):
            out = []
            for t in v:
                fn = t.get('function') if isinstance(t.get('function'), dict) else {}
                name = t.get('name') or t.get('tool') or t.get('tool_name') or t.get('action') or fn.get('name') or '?'
                args = t.get('args', t.get('arguments', t.get('input', t.get('parameters', fn.get('arguments')))))
                if args is None: args = {k2: v2 for k2, v2 in t.items() if k2 not in ('name', 'tool', 'tool_name', 'action', 'function')}
                out.append({'name': str(name), 'args': _maybe_json(args)})
            return out
    return []


def _error(raw):
    e = raw.get('error', raw.get('exception'))
    if isinstance(e, dict) and e:
        return {'class': str(e.get('class') or e.get('type') or e.get('name') or e.get('code') or 'error'), 'message': str(e.get('message') or '')}
    if isinstance(e, str) and e:
        return {'class': e.split(':')[0][:80], 'message': e}
    for k in ('exit_status', 'exit_reason', 'stop_reason', 'termination_reason'):
        v = raw.get(k)
        if isinstance(v, str) and LOOP_WORDS.search(v): return {'class': v, 'message': f'{k}={v}'}
    if str(raw.get('status', '')).lower() in ('error', 'failed', 'failure'):
        return {'class': 'error', 'message': 'status=' + str(raw.get('status'))}
    return None


def normalise_record(raw):
    """One logged run, in whatever shape the harness wrote it, as a canonical record (see the module docstring)."""
    if not isinstance(raw, dict): return raw
    _, case = _first(raw, CASE_KEYS)
    _, cand = _first(raw, CANDIDATE_KEYS)
    def group(keys):
        present = [(k, raw[k]) for k in keys if k in raw]
        if not present: return None
        if len(present) == 1 and present[0][0] in ('input', 'output', 'expected'): return present[0][1]
        return {k: v for k, v in present}
    expected = group(EXPECTED_KEYS)
    output = group(OUTPUT_KEYS)
    if isinstance(output, dict) and all(v is None for v in output.values()): output = None
    _, cost = _first(raw, COST_KEYS)
    if cost is None and isinstance(raw.get('usage'), dict): cost = raw['usage'].get('cost_usd')
    rec = {'case_id': None if case is None else str(case), 'candidate_id': None if cand is None else str(cand), 'input': group(INPUT_KEYS),
           'output': output, 'expected': expected, 'tool_calls': _tool_calls(raw), 'error': _error(raw),
           'cost_usd': cost if isinstance(cost, (int, float)) and not isinstance(cost, bool) else None, 'meta': raw.get('meta') or {}, 'raw': raw}
    count = _call_count(raw)
    if count is not None: rec['call_count'] = count
    outcome = _recorded_outcome(raw)
    if outcome is not None: rec['recorded_outcome'] = outcome
    aver = raw.get('_aver')
    if isinstance(aver, dict):   # from a format adapter (log_formats); never copied into rows
        rec['aver'] = aver
        if aver.get('recorded_outcome'): rec['recorded_outcome'] = dict(aver['recorded_outcome'])
        if 'output_logged' in aver: rec['output_logged'] = bool(aver['output_logged'])
    return rec


def _call_count(raw):
    """{'api_calls': 63} -> ('api_calls', 63); also under usage. None when no count is logged."""
    for src, prefix in ((raw, ''), (raw.get('usage') if isinstance(raw.get('usage'), dict) else {}, 'usage.')):
        for k in CALL_COUNT_KEYS:
            v = src.get(k)
            if isinstance(v, int) and not isinstance(v, bool) and v >= 0: return {'field': prefix + k, 'n': v}
    return None


def _recorded_outcome(raw):
    """{'outcome': {'tests_passed': False}} -> {'field': 'outcome.tests_passed', 'passed': False}. Only boolean leaves with
    an outcome name, at the top level or directly under an outcome-like key; never inside the reference."""
    for k in RECORDED_OUTCOME_LEAVES:
        if isinstance(raw.get(k), bool): return {'field': k, 'passed': raw[k]}
    for parent in RECORDED_OUTCOME_PARENTS:
        v = raw.get(parent)
        if not isinstance(v, dict): continue
        for k in RECORDED_OUTCOME_LEAVES:
            if isinstance(v.get(k), bool): return {'field': f'{parent}.{k}', 'passed': v[k]}
    return None


def _n_tokens(rec):
    """Total tokens a run used, when a format adapter logged them; else None."""
    t = (rec.get('aver') or {}).get('tokens')
    return sum(a + b for a, b in t.values()) if t else None


def _n_calls(rec):
    """Calls a run made: the logged tool calls, else a logged count of calls (thin logs), else 0."""
    calls = rec.get('tool_calls') or []
    if calls: return len(calls), 'tool calls'
    c = rec.get('call_count')
    return (c['n'], c['field']) if c else (0, 'tool calls')


def normalise_logs(logs):
    """A list of records (JSON/JSONL as logged), an OTLP/JSON export (dict with resourceSpans), or a list mixing both ->
    canonical records. Canonical records pass through unchanged except for the added keys."""
    items = logs if isinstance(logs, list) else [logs]
    out = []
    for x in items:
        if isinstance(x, dict) and 'resourceSpans' in x: out.extend(normalise_record(r) for r in _from_otlp(x))
        else: out.append(normalise_record(x))
    return out


# ----------------------------------------------------------------------------- the heuristic reader

def _tokens(name):
    return [t for t in re.split(r'[._\-/: ]+', str(name).lower()) if t]


def _side_effect(name):
    return any(t in SIDE_EFFECT_VERBS for t in _tokens(name))


def _file_tool(name):
    return any(f in str(name).lower() for f in FILE_TOOLS)


def _verb(toks):
    """The side-effect verb of a tool name: the last verb token (object.verb is the usual convention: refund.issue)."""
    vs = [t for t in toks if t in SIDE_EFFECT_VERBS]
    return vs[-1] if vs else None


def _unwanted_kind(name):
    """ledger.write -> unwanted_write 'Wrote to the ledger when it should not have'; refund.issue -> unwanted_refund."""
    toks = _tokens(name)
    verb = _verb(toks) or (toks[0] if toks else 'action')
    obj = ' '.join(t for t in toks if t != verb) or 'it'
    if verb in ('issue', 'execute', 'create', 'submit', 'insert') and obj != 'it':
        key = 'unwanted_' + obj.replace(' ', '_'); lab = f'{"Issued" if verb == "issue" else verb.capitalize() + "d"} a {obj} it should not have'
    elif verb in ('refund', 'pay', 'charge', 'transfer'):
        key = f'unwanted_{verb}'; lab = {'refund': 'Issued a refund', 'pay': 'Made a payment', 'charge': 'Charged a customer', 'transfer': 'Moved money'}[verb] + ' it should not have'
    else:
        past = {'write': 'Wrote to', 'send': 'Sent', 'delete': 'Deleted', 'post': 'Posted to', 'update': 'Updated', 'email': 'Emailed', 'publish': 'Published',
                'merge': 'Merged', 'push': 'Pushed to', 'deploy': 'Deployed', 'cancel': 'Cancelled', 'remove': 'Removed', 'drop': 'Dropped'}.get(verb, f'Called {name} on')
        key = f'unwanted_{verb}'; lab = f'{past} the {obj} when it should not have' if obj != 'it' else f'{past} something when it should not have'
    return re.sub(r'[^a-z0-9_]', '_', key)[:64], lab[:120]


def _same(a, b):
    if isinstance(a, bool) or isinstance(b, bool): return a == b
    try:
        fa, fb = float(str(a).replace(',', '')), float(str(b).replace(',', ''))
        return abs(fa - fb) <= .005 + 1e-9 * max(abs(fa), abs(fb))
    except (TypeError, ValueError):
        return _normal(a) == _normal(b)


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


class HeuristicReader:
    """Structural rules only. No network, no model. Deterministic."""
    name = 'heuristic'
    version = '2'

    def __init__(self):
        self._prepared_for = None; self._ctx = {}; self._findings = {}

    # ---------------------------------------------------------------- context shared across records
    def prepare(self, records):
        if self._prepared_for is records: return
        self._prepared_for = records
        costs, ncalls, peers, case_costs = defaultdict(list), defaultdict(list), defaultdict(dict), defaultdict(dict)
        tokens, nruns = defaultdict(list), Counter()
        for r in records:
            c = r.get('candidate_id'); nruns[c] += 1
            if r.get('cost_usd') is not None: costs[c].append(r['cost_usd']); case_costs[r.get('case_id')][c] = r['cost_usd']
            ncalls[c].append(_n_calls(r)[0])
            t = _n_tokens(r)
            if t is not None: tokens[c].append(t)
            if not r.get('error'):
                peers[r.get('case_id')][c] = {t['name'] for t in (r.get('tool_calls') or []) if _side_effect(t['name'])}
        self._ctx = {'cost_median': {c: _median(v) for c, v in costs.items()}, 'calls_median': {c: _median(v) for c, v in ncalls.items()}, 'peers': peers, 'case_costs': case_costs,
                     'n_runs': dict(nruns), 'n_costs': {c: len(v) for c, v in costs.items()},
                     'tokens_median': {c: _median(v) for c, v in tokens.items() if c not in costs}}
        self._findings = {(r.get('candidate_id'), r.get('case_id')): self._find(r) for r in records}

    def finding(self, record):
        key = (record.get('candidate_id'), record.get('case_id'))
        if key not in self._findings: self._findings[key] = self._find(record)
        return self._findings[key]

    def label_new_run(self, record, peers, taxonomy):
        """Label a run that was NOT in the logs read (a confirmation case), by the same rules and with the context learned
        from the logs (each configuration's median cost and calls). `peers`: the other configurations' runs on the same
        case, used only by the cost rule (as on the logs). Returns {mistake, rationale, origin}; origin is 'customer'
        when the label rests on a recorded outcome and fixed thresholds only, as for the logs (docs/137)."""
        need(self._prepared_for is not None, 'invalid_reader', 'reader', 'Read the logs first (prepare), so new runs are labelled with their context.')
        case = record.get('case_id')
        saved = {k: self._ctx[k].get(case) for k in ('case_costs', 'peers') if case in self._ctx[k]}
        self._ctx['case_costs'][case] = {p['candidate_id']: p['cost_usd'] for p in [*peers, record] if p.get('cost_usd') is not None}
        self._ctx['peers'][case] = {p['candidate_id']: {t['name'] for t in (p.get('tool_calls') or []) if _side_effect(t['name'])} for p in [*peers, record] if not p.get('error')}
        try:
            f = self._find(record)
        finally:
            for k in ('case_costs', 'peers'):
                if k in saved: self._ctx[k][case] = saved[k]
                else: self._ctx[k].pop(case, None)
        m = _map_kind(f['mistake'], f['spec'][2] if f['mistake'] else None, taxonomy['kinds'])
        origin = 'customer' if record.get('recorded_outcome') and f.get('basis') in ('recorded', 'threshold') else 'ai-labeled'
        return {'mistake': m, 'rationale': f['rationale'], 'origin': origin}

    def labels_origin(self, records):
        """('customer', field) when every run carries an outcome the harness recorded and no label rests on the reader's
        judgement; else ('ai-labeled', None). Customer labels need no audit (docs/129 §3.5 audits reader labels only)."""
        if not records or not all(r.get('recorded_outcome') for r in records): return 'ai-labeled', None
        if not all(self.finding(r).get('basis') in ('recorded', 'threshold') for r in records): return 'ai-labeled', None
        return 'customer', Counter(r['recorded_outcome']['field'] for r in records).most_common(1)[0][0]

    # ---------------------------------------------------------------- what the agent is for, and how it is deployed
    def describe(self, records):
        records = normalise_logs(records) if records and 'raw' not in records[0] else records
        self.prepare(records)
        tools = Counter(t['name'] for r in records for t in (r.get('tool_calls') or []))
        side = [t for t, _ in tools.most_common() if _side_effect(t)]
        gate, gate_field = _review_gate(records)
        cands = sorted({r.get('candidate_id') for r in records if r.get('candidate_id')})
        trace = _trace_format(records)
        handoffs = sum((r.get('aver') or {}).get('handoffs') or 0 for r in records)
        guards = sum(len((r.get('aver') or {}).get('guardrails') or []) for r in records)
        if gate:
            dep = {'mode': 'normal', 'signal': 'review_gate', 'why': f'every run carries a human review gate ({gate_field}); nothing the agent does takes effect before a person approves it'}
        elif side:
            dep = {'mode': 'agentic', 'signal': 'side_effect_tools', 'why': f"the agent calls {', '.join(side[:3])} on its own and no human-review field is logged"}
        elif trace and (handoffs or guards):
            what = ' and '.join(x for x in (f'{handoffs} handoff(s) between agents' if handoffs else '', f'{guards} guardrail check(s)' if guards else '') if x)
            dep = {'mode': 'agentic', 'signal': 'handoffs_or_guardrails', 'why': f'the traces show {what}: agents decide and hand work on by themselves, and no human-review field is logged'}
        elif trace:
            dep = {'mode': 'agentic', 'signal': 'no_review_field', 'why': (f'no human-review field is logged in this {trace} export, so nothing shows a person approving an output before it takes effect'
                                                                           + (f'; the tools called ({", ".join(list(tools)[:3])}) do not change anything outside the run' if tools else '; no tool calls are logged')
                                                                           + '. Read as agentic; set normal if a person reviews each output first')}
        elif not tools:
            counted = Counter(r['call_count']['field'] for r in records if r.get('call_count'))
            what = f', only a count of calls ({counted.most_common(1)[0][0]})' if counted else ''
            dep = {'mode': 'normal', 'signal': 'no_tool_calls', 'why': (f'no tool calls are logged{what}: nothing in the logs shows the agent acting outside the run, and no review '
                                             'field is logged either. Read as normal; confirm that a person reviews or approves each output before it takes effect')}
        else:
            dep = {'mode': 'normal', 'signal': 'no_side_effects', 'why': 'no tool call in the logs changes anything outside the run; the output is only shown'}
        structured = sum(isinstance(r.get('output'), dict) and r.get('expected') is not None for r in records) >= len(records) / 2
        purpose = _purpose(records, tools, side, gate) if (not trace or structured) else None
        purpose_why = None
        if purpose is None or purpose.startswith('Produces an answer for'):
            purpose, purpose_why = _evidence_purpose(records, tools, side)
        gh = _github_tasks(records) if not tools else None
        out_f = Counter(r['recorded_outcome']['field'] for r in records if r.get('recorded_outcome'))
        purpose_why = purpose_why or ((f'From the task links ({gh[0]}: {len(gh[1])} GitHub repositories, {gh[2]} URLs)' + (f' and the recorded outcome ({out_f.most_common(1)[0][0]})' if out_f else '')
                        + '. Nothing logged says more than that: no tool calls, no inputs, no outputs.') if gh else 'From the tool names, output fields and case-id field of the logs.')
        return {'purpose': purpose, 'deployment': dep, 'purpose_why': purpose_why,
                'summary': f'{len({r.get("case_id") for r in records})} cases x {len(cands)} configurations ({", ".join(cands)}); {len(records)} logged runs.'}

    # ---------------------------------------------------------------- kinds of case
    def propose_strata(self, records, labels=None):
        """The case-level categorical field (2-6 values, same value for a case in every configuration, not part of the
        output or the reference) whose values differ most in how often runs fail. Falls back to meta.kind, then to input
        length tertiles."""
        records = normalise_logs(records) if records and 'raw' not in records[0] else records
        self.prepare(records)
        n_cases = len({r.get('case_id') for r in records})
        if n_cases < MIN_CASES_FOR_STRATA:
            self.strata_rationale = (f'None evident: {n_cases} case(s) are too few to split into kinds of case (at least {MIN_CASES_FOR_STRATA} are needed '
                                     'for a field\'s values to show different failure rates). One kind of case.')
            return {'all': {'weight': 1., 'failure_limit': None, 'how': 'none evident'}}
        if labels is None: labels = {(r.get('candidate_id'), r.get('case_id')): self.finding(r)['mistake'] for r in records}
        best = _best_stratum_field(records, labels) or _best_stratum_field(records, labels, pool_tail=True)
        if best:
            path, levels, score, considered, pooled = best
            cases = {}
            for r in records: cases.setdefault(r['case_id'], _strat_value(r, path))
            if pooled: cases = {k: pooled['mapping'][v] for k, v in cases.items()}
            counts = Counter(cases.values()); total = sum(counts.values())
            how = f'{path} as logged' + (POOLED if pooled else '')
            out = {k: {'weight': counts[k] / total, 'failure_limit': None, 'how': how} for k in sorted(counts)}
            self.strata_rationale = _strata_why(path, counts, total, levels, considered, score, pooled)
            return out
        kinds = Counter((r.get('meta') or {}).get('kind') for r in records)
        if None not in kinds and len(kinds) >= 2:
            total = sum(kinds.values())
            self.strata_rationale = 'Split by meta.kind as logged.'
            return {k: {'weight': kinds[k] / total, 'failure_limit': None, 'how': 'meta.kind as logged'} for k in sorted(kinds)}
        lengths = sorted(len(str(r.get('input') or '')) for r in records)
        if not lengths: return {'ordinary': {'weight': 1., 'failure_limit': None, 'how': 'no inputs logged'}}
        t1, t2 = lengths[len(lengths) // 3], lengths[2 * len(lengths) // 3]
        def bucket(r):
            n = len(str(r.get('input') or ''))
            return 'short' if n < t1 else 'long' if n >= t2 else 'typical'
        counts = Counter(bucket(r) for r in records); total = sum(counts.values())
        self.strata_rationale = 'No case-level field with 2-6 values was logged, so cases are split by input length (tertiles).'
        return {k: {'weight': counts[k] / total, 'failure_limit': None, 'how': f'input length tertile ({t1}, {t2})'} for k in sorted(counts)}

    # ---------------------------------------------------------------- kinds of mistake
    def propose_taxonomy(self, records, standard=None):
        """Start from the standard kinds, keep only kinds with evidence in the logs, and specialise each to the use case
        (key, label, `standard` parent). A taxonomy the user already edited (keys other than the standard set) is kept."""
        tax = standard or standard_taxonomy()
        if set(tax['kinds']) != set(STANDARD_TAXONOMY): return tax
        records = normalise_logs(records) if records and 'raw' not in records[0] else records
        self.prepare(records)
        kinds = {}
        for r in records:
            f = self.finding(r)
            if f['mistake'] and f['mistake'] not in kinds:
                lab, sev, std = f['spec']
                kinds[f['mistake']] = {'label': lab, 'severity': sev, 'price_usd': None, 'tolerance_per_month': None, 'standard': std}
        return {'version': tax['version'], 'kinds': dict(sorted(kinds.items())) or {k: dict(v) for k, v in tax['kinds'].items()}}

    def label(self, record, taxonomy):
        f = self.finding(record)
        m = _map_kind(f['mistake'], f['spec'][2] if f['mistake'] else None, taxonomy['kinds'])
        return {'mistake': m, 'rationale': f['rationale']}

    # ---------------------------------------------------------------- the rules
    def _find(self, rec):
        """One finding per run: the most severe mistake the structure shows (incidents, runaway cost, incomplete, wrong
        output, needless escalation), with a one-line rationale naming fields or tools, never content."""
        raw = rec.get('raw') or rec
        flat = _flatten(raw)
        out, exp = rec.get('output'), rec.get('expected')
        calls = rec.get('tool_calls') or []
        recorded = rec.get('recorded_outcome')
        # basis: 'recorded' (the harness logged it), 'threshold' (a fixed rule on logged numbers), 'judged' (the reader's
        # reading of content or structure). Only labels that are all recorded or threshold count as the team's own (origin
        # customer, no audit); anything judged is a reader's label (origin ai-labeled, audited).
        def hit(kind, why, spec=None, basis='judged'):
            return {'mistake': kind, 'rationale': why, 'spec': spec or KIND_CATALOG[kind], 'basis': basis}
        # 1. unwanted actions: out of scope, then side-effect tools the reference (or every peer) did not call
        scope = _list_at(flat, ('scope', 'allowed_paths', 'allowed_files'))
        changed = _list_at(flat, ('files_changed', 'changed_files', 'modified_files'))
        deleted = _list_at(flat, ('files_deleted', 'deleted_files', 'removed_files'))
        if deleted:
            return hit('out_of_scope_change', f'deleted {len(deleted)} file(s) (patch.files_deleted)')
        if scope and changed:
            outside = [f for f in changed if not any(str(f).startswith(str(s)) for s in scope)]
            if outside: return hit('out_of_scope_change', f'{len(outside)} changed file(s) outside task.scope')
        exp_actions = _expected_actions(exp)
        should = {k[len('should_'):]: v for k, v in _leaves(exp).items() if isinstance(k, str) and k.startswith('should_') and isinstance(v, bool)} if isinstance(exp, dict) else {}
        for t in calls:
            name = t['name']
            if not _side_effect(name) or (scope and _file_tool(name)): continue
            why = None
            if exp_actions is not None:
                if name not in exp_actions: why = f'{name} is not among the expected actions'
            elif should:
                flag = next((v for x, v in should.items() if x in _tokens(name)), None)
                if flag is False: why = f'{name} called where the reference says should_{next(x for x in should if x in _tokens(name))} = false'
            else:
                peers = self._ctx.get('peers', {}).get(rec.get('case_id'), {})
                others = [v for c, v in peers.items() if c != rec.get('candidate_id')]
                if (others and not any(name in v for v in others)) or (not others and rec.get('expected') is not None):
                    why = f'{name} called; no reference action list, and no other configuration called it on this case' if others else f'{name} not in the expected calls'
            if why:
                key, lab = _unwanted_kind(name)
                return hit(key, why, (lab, 'incident', 'unwanted_action'))
        # 2. output that must never be sent
        flagged = [k for k, v in flat.items() if v is True and re.search(r'moderation|guardrail|safety|policy', k, re.I) and re.search(r'flag|violat|block', k, re.I)]
        if flagged:
            return hit('policy_violation', f'{flagged[0]} is true', ('Reply breaks policy (must never be sent)' if _has_reply(out) else KIND_CATALOG['policy_violation'][0], 'incident', 'harmful_output'))
        # 2a. what a trace export records (log_formats): an unwanted action only when the run itself shows it (a guardrail
        # tripped, or the tool call was refused), a tripped guardrail, then failing score or feedback fields
        aver = rec.get('aver') or {}
        if aver:
            refused = {e.get('tool') for e in aver.get('tool_errors') or [] if log_formats.DENIED.search(str(e.get('message', '')))}
            tripped = [g for g in aver.get('guardrails') or [] if g.get('triggered')]
            for t in calls:
                if _side_effect(t['name']) and (tripped or t['name'] in refused):
                    key, lab = _unwanted_kind(t['name'])
                    return hit(key, f"{t['name']} called and " + (f"guardrail {tripped[0].get('name')} tripped" if tripped else 'the call was refused'), (lab, 'incident', 'unwanted_action'))
            if tripped:
                return hit('guardrail_tripped', f"guardrail {tripped[0].get('name')} triggered = true (recorded)", basis='recorded')
            fails = sorted(aver.get('score_fails') or [], key=lambda f: (f['severity'] != 'incident', f['field']))
            if fails:
                f0 = fails[0]
                return hit(f0['kind'], f"{f0['field']} = {f0['value']} (recorded score)", (f0['label'], f0['severity'], f0['standard']), basis='recorded')
        # 2b. a failure the harness recorded (tests failed, not resolved). It comes before runaway cost: the recorded
        # outcome is the team's own label and carries the larger cost (someone must still fix the task); runaway cost is
        # a heuristic on numbers and is kept for runs the harness recorded as passing (docs/137 §3).
        if recorded and recorded['passed'] is False:
            f = recorded['field']
            return hit('tests_fail' if 'test' in f else 'wrong_output', f'{f} = false (recorded by the harness)', basis='recorded')
        # 3. runaway cost
        err = rec.get('error')
        if err and LOOP_WORDS.search(f"{err.get('class', '')} {err.get('message', '')}"):
            return hit('runaway_cost', f"error: {err.get('class')}", basis='threshold' if recorded else 'judged')
        c = rec.get('candidate_id'); cm = self._ctx.get('cost_median', {}).get(c); km = self._ctx.get('calls_median', {}).get(c)
        if rec.get('cost_usd') is not None and cm:
            # cost relative to this configuration's median, compared with the other configurations on the same case (a
            # long document is expensive for everyone; a loop is expensive for one)
            z = rec['cost_usd'] / cm
            others = [v / self._ctx['cost_median'][o] for o, v in self._ctx.get('case_costs', {}).get(rec.get('case_id'), {}).items() if o != c and self._ctx['cost_median'].get(o)]
            if (others and z > 3 and z > 4 * min(others)) or (not others and z > 5 and self._ctx.get('n_costs', {}).get(c, 0) >= 10):
                return hit('runaway_cost', f"cost {z:.1f}x this configuration's median" + (f', {z / min(others):.1f}x the other configurations on this case' if others else ''), basis='threshold')
        tm, tk = self._ctx.get('tokens_median', {}).get(c), _n_tokens(rec)
        if rec.get('cost_usd') is None and tm and tk is not None and tk / tm > 5 and self._ctx.get('n_runs', {}).get(c, 0) >= 10:
            return hit('runaway_cost', f"tokens {tk} = {tk / tm:.1f}x this configuration's median (no cost logged; tokens stand in for cost)", basis='threshold')
        n, what = _n_calls(rec)
        if km is not None and n >= 10 and n >= 2.5 * max(km, 1):
            return hit('runaway_cost', f'{n} {what} against a median of {km}' if what == 'tool calls' else f'{what} = {n} against a median of {km}', basis='threshold')
        if recorded:
            return {'mistake': None, 'rationale': f"{recorded['field']} = true (recorded by the harness)", 'spec': None, 'basis': 'recorded'}
        if err:
            return hit('incomplete', f"error: {err.get('class')}")
        # 4. incomplete
        cov = _coverage(flat)
        if cov:
            unit, done, total = cov
            return hit('missed_page' if unit.startswith('page') else 'incomplete', f'{unit}: processed {done} of {total}')
        fin = [v for k, v in flat.items() if k.rsplit('.', 1)[-1] in ('finish_reason', 'stop_reason') and str(v).lower() in ('length', 'max_tokens', 'truncated')]
        if fin: return hit('incomplete_reply' if _has_reply(out) else 'incomplete', f'finish_reason = {fin[0]}')
        qa = [k for k, v in flat.items() if re.search(r'qa|review', k, re.I) and isinstance(v, list) and any(re.search(r'incomplete|unanswered|missing', str(x), re.I) for x in v)]
        if qa: return hit('incomplete_reply' if _has_reply(out) else 'incomplete', f'{qa[0]} names an incomplete answer')
        tests = _tests(flat)
        if tests:
            f2p, f2p_n, p2p, p2p_n = tests
            if p2p_n and p2p < p2p_n: return hit('tests_fail', f'pass_to_pass {p2p}/{p2p_n}: a regression')
            if f2p_n and f2p == 0: return hit('tests_fail', f'fail_to_pass 0/{f2p_n}')
            if f2p_n and f2p < f2p_n: return hit('partial_patch', f'fail_to_pass {f2p}/{f2p_n}')
        if changed is not None and not changed and isinstance(raw, dict) and 'patch' in raw:
            return hit('partial_patch', 'patch.files_changed is empty')
        if (out is None or out == '' or out == {}) and rec.get('output_logged', True):
            return hit('incomplete', 'no output')
        # 5. wrong output: invented values, mismatched fields, routing against the reference
        if exp is not None and not isinstance(exp, (list,)):
            o, e = _leaves(out, 'output'), _leaves(exp, 'output' if not isinstance(exp, dict) else 'value')
            if not isinstance(out, dict) and not isinstance(exp, dict):
                if not _same(out, exp): return hit('wrong_output', 'output differs from the expected value')
            shared = [k for k in o if k in e]
            if o and len(shared) >= max(1, len(o) / 2):
                invented = [k for k, v in o.items() if v not in (None, '', []) and (k not in e or e[k] in (None, ''))]
                if invented: return hit('invented_value', f'{invented[0]} has a value; the reference has none')
            wrong = [k for k in shared if o[k] is not None and not _same(o[k], e[k])]
            if wrong:
                if all(k in CATEGORY_LEAVES for k in wrong): return hit('wrong_category', f'{wrong[0]} differs from the reference')
                if all(k in ('output', 'value', 'answer', 'result') for k in wrong): return hit('wrong_output', 'output differs from the expected value')
                return hit('wrong_field', f'{", ".join(sorted(wrong)[:3])} differ(s) from the reference')
            for x, v in should.items():
                done = _did(x, o, calls)
                if done is None: continue
                if v and not done:
                    return hit('wrong_category' if x.startswith('escalat') else 'wrong_output', f'should_{x} = true, not done')
                if not v and done and x.startswith('escalat'):
                    return hit('needless_escalation', f'escalated where should_{x} = false')
        return {'mistake': None, 'rationale': 'completed', 'spec': None, 'basis': 'judged'}


def _map_kind(kind, standard, kinds):
    """A finding's kind in the taxonomy actually in use: itself, else the first kind with the same standard parent."""
    if kind is None: return None
    if kind in kinds: return kind
    same = sorted(k for k, v in kinds.items() if v.get('standard') == standard)
    return same[0] if same else None


def _list_at(flat, names):
    for k, v in flat.items():
        if k.rsplit('.', 1)[-1] in names and isinstance(v, list): return v
    return None


def _expected_actions(exp):
    if not isinstance(exp, dict): return None
    for k, v in _flatten(exp).items():
        leaf = k.rsplit('.', 1)[-1]
        if isinstance(v, list) and ('action' in leaf or leaf in ('tool_calls', 'tools', 'expected_tools')):
            return {str(x.get('name') if isinstance(x, dict) else x) for x in v}
    return None


def _has_reply(out):
    if not isinstance(out, dict): return False
    parts = {seg for path in _flatten(out) for seg in path.split('.')}
    return bool(parts & {'reply', 'text', 'message', 'response'})


def _coverage(flat):
    for k, v in flat.items():
        m = re.match(r'(?:.*\.)?(\w+?)_(processed|read|seen|covered)$', k)
        if not m or not isinstance(v, int) or isinstance(v, bool): continue
        units = {m.group(1), m.group(1).rstrip('s'), m.group(1).rstrip('s') + 's'}
        unit = m.group(1)
        for k2, total in flat.items():
            leaf = k2.rsplit('.', 1)[-1]
            if any(leaf in (f'{x}_count', f'num_{x}', f'{x}_total', f'total_{x}') for x in units) or (leaf in units and k2 != k):
                if isinstance(total, int) and not isinstance(total, bool) and v < total: return unit, v, total
    return None


def _tests(flat):
    g = lambda *names: next((v for k, v in flat.items() if k.endswith(names) and isinstance(v, int) and not isinstance(v, bool)), None)
    f2p, f2p_n = g('fail_to_pass.passed'), g('fail_to_pass.total')
    p2p, p2p_n = g('pass_to_pass.passed'), g('pass_to_pass.total')
    if f2p_n is None and p2p_n is None:
        failed, passed = g('tests_failed'), g('tests_passed')
        if failed is None: return None
        return (0, 0, passed or 0, (passed or 0) + failed)
    return (f2p or 0, f2p_n or 0, p2p or 0, p2p_n or 0)


def _did(x, out_leaves, calls):
    stem = x[:6]
    for k, v in out_leaves.items():
        if isinstance(v, bool) and k.startswith(stem): return v
    if any(stem in t['name'].lower() or ('handoff' in t['name'].lower() and stem.startswith('escala')) for t in calls): return True
    return None


def _review_gate(records):
    n = 0; field = None
    for r in records:
        raw = r.get('raw') or {}
        for k in ('review', 'human_review', 'approval', 'requires_approval', 'review_required'):
            v = raw.get(k)
            if (isinstance(v, dict) and (v.get('required') is True or 'state' in v or 'status' in v)) or v is True:
                n += 1; field = f'{k}.required' if isinstance(v, dict) and 'required' in v else k; break
    return (n >= .9 * len(records) and n > 0), field


def _case_noun(records):
    raw = records[0].get('raw') or {}
    keys = set(_flatten(raw))
    text = ' '.join(str(k).lower() for k in keys) + ' ' + ' '.join(str(v).lower() for r in records[:50] for v in _flatten(r.get('raw') or {}).values() if isinstance(v, str) and len(v) < 40)
    case_key = next((k for k in CASE_KEYS if k in raw), 'case_id')
    if 'invoice' in text: return 'supplier invoices' if 'supplier' in text or 'vendor' in text else 'invoices'
    if case_key == 'ticket_id' or 'ticket' in text: return 'support tickets'
    if 'repo' in text and ('patch' in text or 'files_changed' in text): return 'engineering tasks'
    return {'document_id': 'documents', 'task_id': 'tasks', 'example_id': 'requests', 'conversation_id': 'conversations'}.get(case_key, 'cases')


def _github_tasks(records):
    """(field, {owner/repo}, 'issues'|'pull') when every run links its task to a GitHub issue or pull request."""
    field, repos, kinds = None, set(), Counter()
    for r in records:
        hit = next(((k, m) for k, v in _flatten(r.get('raw') or {}).items() if isinstance(v, str) for m in [GITHUB_TASK.match(v)] if m), None)
        if hit is None: return None
        field = field or hit[0]; repos.add(f'{hit[1].group(1)}/{hit[1].group(2)}'); kinds[hit[1].group(3)] += 1
    return (field, repos, kinds.most_common(1)[0][0]) if records else None


def _purpose(records, tools, side, gate):
    gh = _github_tasks(records)
    outcome = Counter(r['recorded_outcome']['field'] for r in records if r.get('recorded_outcome'))
    if gh and not tools:
        field, repos, kind = gh
        head = (f'Resolves GitHub issues from {len(repos)} repositories by producing a code change' if kind == 'issues'
                else f'Produces a code change for tasks from {len(repos)} GitHub repositories ({field} links each task to a pull request on github.com)')
        if outcome:
            f = outcome.most_common(1)[0][0]
            head += f"; the repository's tests decide each run ({f})" if 'test' in f else f'; {f} records whether each run succeeded'
        return head + ('; a person reviews each change before it takes effect' if gate else '') + '.'
    noun = _case_noun(records)
    leaves = Counter(k for r in records[:200] if isinstance(r.get('output'), dict) for k, v in _leaves(r['output']).items())
    raw0 = records[0].get('raw') or {}
    if 'patch' in raw0 or any('files_changed' in k for k in _flatten(raw0)):
        head = f'Writes code changes for {noun}'
    elif any(k in CATEGORY_LEAVES for k in leaves):
        head = f'Classifies {noun}' + (' and drafts a reply' if any(k in ('text', 'reply') for k in leaves) else '')
    elif len(leaves) >= 4:
        head = f'Extracts fields from {noun}'
    else:
        head = f'Produces an answer for {noun}'
    acts = []
    for t in ([] if gate else side):
        toks = _tokens(t); verb = _verb(toks) or ''; obj = ' '.join(x for x in toks if x != verb)
        if verb == 'write' and head.startswith('Extracts'): acts.append(f'writes them to the {obj or "store"}')
        elif verb == 'write': acts.append(f'writes to the {obj or "store"}')
        elif verb == 'issue': acts.append(f'issues {obj}s')
        elif verb == 'send': acts.append(f'sends the {obj}' if obj else 'sends messages')
        elif verb: acts.append(f'{verb}s {obj}'.strip())
    if not gate and any('handoff' in t.lower() or 'escalat' in t.lower() for t in tools): acts.append('hands off to a person when needed')
    acts = list(dict.fromkeys(acts))[:4]
    s = head
    if acts and head.startswith('Extracts') and len(acts) == 1: s += ' and ' + acts[0]
    elif acts: s += '; it ' + (', '.join(acts[:-1]) + ' and ' + acts[-1] if len(acts) > 1 else acts[0])
    if gate: s += '; a person reviews each change before it takes effect'
    return s + '.'


def _trace_format(records):
    """The trace format (otel, runs, observations, agents_sdk) when every run came through that adapter, else None."""
    fmts = {(r.get('aver') or {}).get('format') for r in records}
    return next(iter(fmts)) if len(fmts) == 1 and next(iter(fmts)) in log_formats.TRACE_FORMATS else None


PURPOSE_STOP = {'get', 'set', 'run', 'running', 'tool', 'tools', 'agent', 'agents', 'execute', 'call', 'calls', 'final', 'result', 'results', 'suggestion',
                'suggestions', 'mcp', 'test', 'chat', 'llm', 'invoke', 'invocation', 'workflow', 'the', 'and', 'app', 'main', 'trace', 'index', 'span', 'task',
                'create', 'created', 'kickoff', 'core', 'generate', 'text', 'content', 'function', 'some', 'api', 'client', 'server', 'request', 'response',
                'default', 'new', 'with', 'from', 'you', 'are', 'helpful', 'assistant', 'say', 'conversation', 'completion', 'completions', 'openai', 'who',
                'helps', 'information', 'friendly', 'that', 'for', 'your', 'can', 'use', 'user', 'query', 'answer', 'answers', 'generator', 'langgraph', 'execute_tool'}
PLACEHOLDER_WORDS = {'foo', 'bar', 'baz', 'qux', 'dummy', 'example', 'sample', 'my'}
GENERIC_ROLES = {'helpful assistant', 'assistant', 'ai assistant', 'helpful ai assistant', 'helpful and harmless assistant'}


def _words(name):
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', str(name))
    return [w for w in re.split(r'[^A-Za-z]+', s.lower()) if len(w) >= 3 and w not in PURPOSE_STOP and w not in PLACEHOLDER_WORDS]


def _role(prompt):
    m = re.search(r"\byou are (?:an? |the )?([^.\n;:!?]{3,90})", prompt or '', re.I)
    role = m.group(1).strip() if m else None
    return None if not role or role.lower() in GENERIC_ROLES else role


def _join(xs):
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' and ' + xs[-1] if xs else ''


def _evidence_purpose(records, tools, side):
    """Purpose from what the logs name (docs/141 §5): agent names, the system prompt (first 200 characters, read here
    only), tool names and run names. When none of them is logged, or every one is a placeholder, it says what is
    missing instead of guessing. Returns (sentence, rationale)."""
    av = [r.get('aver') or {} for r in records]
    n = len(records)
    is_ph = lambda x: bool(log_formats.PLACEHOLDER.match(str(x)))
    generic = lambda x: str(x).lower() in log_formats.GENERIC_NAMES or bool(re.match(r'^(invoke_agent|chat)\b', str(x).lower()))
    named = lambda a: not log_formats.UUIDISH.match(a) and not generic(a)
    agents = Counter(a for x in av for a in dict.fromkeys(x.get('agents') or []) if named(a))
    models = Counter(m for x in av for m in dict.fromkeys(x.get('models') or []))
    run_names = Counter(str(x['run_name']) for x in av if x.get('run_name'))
    descr_names = [k for k in run_names if not generic(k) and not is_ph(k)]
    roles = Counter(_role(x.get('system_prompt')) for x in av if _role(x.get('system_prompt')))
    run_tools = [list(dict.fromkeys(t['name'] for t in (r.get('tool_calls') or []))) for r in records]
    handoffs = sum(x.get('handoffs') or 0 for x in av)
    guards = sum(len(x.get('guardrails') or []) for x in av)
    names_all = list(tools) + list(agents) + descr_names
    placeholder = bool(names_all) and all(is_ph(x) for x in list(tools) + list(agents)) and not descr_names and not roles
    def topics(i):
        x = av[i]
        ws = [w for t in run_tools[i] for w in _words(t)] + [w for a in dict.fromkeys(x.get('agents') or []) if named(a) for w in _words(a)]
        if x.get('run_name') and not generic(x['run_name']): ws += _words(x['run_name'])
        if _role(x.get('system_prompt')): ws += _words(_role(x['system_prompt']))
        return ws   # with repeats: a word in more of the run's names describes it better
    per_run_all = [topics(i) for i in range(n)]
    per_run = [list(dict.fromkeys(ws)) for ws in per_run_all]
    freq = Counter(w for ws in per_run for w in ws)
    shared = [w for w, c in freq.most_common() if c >= max(2, (n + 1) // 2) or n == 1]
    has_ev = [bool(run_tools[i] or [a for a in (av[i].get('agents') or []) if named(a)] or _role(av[i].get('system_prompt'))
                   or (av[i].get('run_name') and not generic(av[i]['run_name']) and not is_ph(av[i]['run_name']))) for i in range(n)]
    extras = []   # verb phrases: '...; it acts outside the run (...), hands off between agents and runs guardrail checks'
    if side: extras.append(f'acts outside the run ({_join(side[:3])})')
    if handoffs: extras.append('hands off between agents')
    if guards: extras.append('runs guardrail checks')
    sources = [f'tool names ({_join(list(tools)[:5])})' if tools else None, f'agent names ({_join(list(agents)[:4])})' if agents else None,
               f'the system prompt (its first {log_formats.SYSTEM_PROMPT_CHARS} characters; a role in {sum(roles.values())} of {n} runs)' if roles else None,
               f'run names ({_join(descr_names[:3])})' if descr_names else None]
    sources = [x for x in sources if x]
    missing = [w for w, ok in (('tool names', tools), ('agent names', agents), ('a system prompt', roles), ('descriptive run names', descr_names)) if not ok]
    if not names_all and not roles:
        what = f"Calls {_join([m for m, _ in models.most_common(3)])}" if models else f"Runs {n} logged run{'s' if n != 1 else ''}"
        gen = [k for k in run_names if generic(k) or is_ph(k)]
        s = (what + (f" ({_join(gen[:3])})" if gen else '') + f"{' for single requests' if models else ''}; the logs name no agent, tool, system prompt or descriptive run name, "
             'so what it is for is not evident.')
        return s, f"Nothing logged names what the runs are for: missing {_join(missing)}." + (f" Models seen: {_join(list(models)[:4])}." if models else '')
    if placeholder:
        s = (f"Runs agents ({_join(list(agents)[:4])})" if agents else 'Runs an agent') + (f" that call {_join(list(tools)[:4])}" if tools else '')
        s += ('; it ' + _join(extras) if extras else '') + f"; the names are placeholders, so what it is for is not evident."
        return s, f"From {_join(sources)}; every name is a placeholder (test, foo, some_function, ...). Missing: {_join(missing) or 'nothing else'}."
    if shared:
        topic = ' '.join(shared[:2])
        s = f"Handles {topic} requests" + (f" by calling {_join(list(tools)[:5])}" if tools else '') + (f" (agents: {_join(list(agents)[:3])})" if agents else '')
        if roles:
            role, k = roles.most_common(1)[0]
            s += f"; its system prompt casts it as {'an' if role[:1].lower() in 'aeiou' else 'a'} {role}" + (f' (in {k} of {n} runs)' if k < n else '')
    else:
        descs = []
        for i in range(n):
            if not has_ev[i]: continue
            within = Counter(per_run_all[i])
            ws = sorted(per_run[i], key=lambda w: (-within[w], -freq[w], per_run[i].index(w)))[:2]
            d = ' '.join(ws) or 'agent'
            if run_tools[i]: d += f" ({_join(run_tools[i][:3])})"
            if d not in descs: descs.append(d)
        s = 'Runs several different agents: ' + '; '.join(descs[:5])
    if extras: s += '; it ' + _join(extras)
    blind = n - sum(has_ev)
    if blind: s += f"; for {blind} of {n} runs nothing logged says what they are for"
    return s + '.', f"From {_join(sources)}." + (f" Missing: {_join(missing)}." if missing else '')


OUT_OR_REF = set(OUTPUT_KEYS) | set(EXPECTED_KEYS) | {'usage', 'error', 'exception', 'steps', 'qa_review', 'review', 'moderation', 'tests', 'trajectory', '_aver', 'feedback', 'scores'} | set(RECORDED_OUTCOME_PARENTS)


def _strat_value(r, path):
    flat = _flatten(r.get('raw') or {})
    if path not in flat and path.startswith('meta.'): flat = {**flat, **{'meta.' + k: v for k, v in _flatten(r.get('meta') or {}).items()}}
    v = flat.get(path)
    return 'true' if v is True else 'false' if v is False else str(v)


def _pool_tail(vals, n_cases):
    """A long tail of values -> {value: kind of case}: values with at least OTHER_SHARE of the cases keep their name (at
    most five, most common first), the rest are pooled as 'other'. None when that leaves fewer than two kinds of case."""
    named = [v for v, c in sorted(vals.items(), key=lambda vc: (-vc[1], vc[0])) if c >= max(2, OTHER_SHARE * n_cases)][:5]
    if not named: return None
    other = 'other (pooled)' if 'other' in named else 'other'
    mapping = {v: (v if v in named else other) for v in vals}
    return mapping if len(set(mapping.values())) >= 2 else None


def _best_stratum_field(records, labels, pool_tail=False):
    """The case-level field whose values differ most in failure rate. With pool_tail, only fields with more than six values
    are scored, each with its tail pooled as 'other' (_pool_tail); used only when no field has 2-6 values (docs/137)."""
    by_case = defaultdict(dict)
    for r in records:
        raw = r.get('raw') or {}
        flat = {k: v for k, v in _flatten(raw).items() if k.split('.')[0] not in OUT_OR_REF and k.split('.')[0] not in CASE_KEYS + CANDIDATE_KEYS
                and isinstance(v, (str, bool)) and not re.search(r'(^|\.)(id|.*_id|.*_at|time.*|.*timestamp|trace.*|span.*|input\.value|body|text|summary|issue|name)$', k)}
        if 'meta' in raw: flat.update({'meta.' + k: v for k, v in _flatten(raw['meta']).items() if isinstance(v, (str, bool))})
        by_case[r['case_id']].setdefault('runs', []).append((r, flat))
    candidates = None
    for case, d in by_case.items():
        keys = set.intersection(*[set(f) for _, f in d['runs']])
        keys = {k for k in keys if len({f[k] for _, f in d['runs']}) == 1}   # same value in every configuration
        candidates = keys if candidates is None else candidates & keys
    if not candidates: return None
    n_cases = len(by_case); scored = []
    for path in sorted(candidates):
        vals = Counter(str(d['runs'][0][1][path]) for d in by_case.values())
        mapping = None
        if pool_tail:
            if len(vals) <= 6 or len(vals) > MAX_POOLED_VALUES: continue
            mapping = _pool_tail(vals, n_cases)
            if mapping is None: continue
            raw_values = len(vals)
            vals = Counter(mapping[v] for v in vals.elements())
        if not (2 <= len(vals) <= 6) or min(vals.values()) < max(2, .03 * n_cases): continue
        fails = Counter(); runs = Counter()
        for d in by_case.values():
            v = str(d['runs'][0][1][path])
            if mapping: v = mapping[v]
            for r, _ in d['runs']:
                runs[v] += 1; fails[v] += labels.get((r['candidate_id'], r['case_id'])) is not None
        N = sum(runs.values()); p = sum(fails.values()) / N
        chi = 0. if p in (0, 1) else sum(runs[v] * (fails[v] / runs[v] - p) ** 2 for v in runs) / (p * (1 - p))
        k = len(vals) - 1
        z = ((chi / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))   # Wilson-Hilferty: chi-square on k df as a z-score
        pooled = None if mapping is None else {'mapping': mapping, 'values': raw_values}
        scored.append((z, -len(vals), path, {v: fails[v] / runs[v] for v in runs}, pooled))
    if not scored: return None
    scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
    score, _, path, levels, pooled = scored[0]
    return path, levels, score, [(p, s) for s, _, p, _, _ in scored[1:]], pooled


def _strata_why(path, counts, total, levels, considered, score, pooled=None):
    shares = ', '.join(f'{k} {counts[k] / total:.0%}' for k, _ in counts.most_common())
    lo, hi = min(levels.values()), max(levels.values())
    if pooled:
        other = next(v for v in pooled['mapping'].values() if v.startswith('other'))
        n_other = sum(1 for v in pooled['mapping'].values() if v == other)
        s = (f'Split by {path} ({len(counts)} kinds of case: {shares}). No case-level field has 2-6 values; {path} has {pooled["values"]}, so the '
             f'{len(counts) - 1} values with at least {OTHER_SHARE:.0%} of cases each keep their name and the other {n_other} are pooled as "{other}". '
             f'Of the fields that are the same for a case in every configuration, its kinds of case differ most in how often runs fail '
             f'(from {lo:.1%} to {hi:.1%} of runs by kind).')
    else:
        s = (f'Split by {path} ({len(counts)} kinds of case: {shares}). Of the case-level fields with 2-6 values that are the same for a '
             f'case in every configuration, its values differ most in how often runs fail (from {lo:.1%} to {hi:.1%} of runs by value).')
    if considered: s += ' Also considered: ' + ', '.join(f'{p} (z {sc:.1f})' for p, sc in considered[:3]) + f' against {path}\'s z of {score:.1f} (chi-square of failure by value, as a z-score).'
    return s


# ----------------------------------------------------------------------------- the model-backed reader (interface)

class LLMReader:
    """Interface for a model-backed reader (hosted only). Contract for any implementation:

    describe(records) -> {'purpose': str, 'deployment': {'mode': 'normal'|'agentic', 'why': str}}
    propose_strata(records, labels=None) -> {stratum: {'weight', 'failure_limit': None, 'how'}}
    propose_taxonomy(records, standard) -> taxonomy whose kinds each carry a `standard` parent; keys ASCII labels.
    label(record, taxonomy) -> {'mistake': key or None, 'rationale': str}; None when unsure (treated as unknown).
    The prompt is LLM_READER_SYSTEM_PROMPT; the model must return LLM_READER_RESPONSE_SCHEMA. The reader never receives
    price_usd or tolerance_per_month (they are stripped before the call), never writes raw content into a row, and its
    labels enter a decision only through audit_sample/apply_audit.
    """
    name = 'llm'
    version = 'interface'
    system_prompt = LLM_READER_SYSTEM_PROMPT
    response_schema = LLM_READER_RESPONSE_SCHEMA

    def describe(self, records): raise NotImplementedError('wire a provider in the hosted service')
    def propose_strata(self, records, labels=None): return HeuristicReader().propose_strata(records, labels)
    def propose_taxonomy(self, records, standard=None): raise NotImplementedError('wire a provider in the hosted service')
    def label(self, record, taxonomy): raise NotImplementedError('wire a provider in the hosted service')


class MockLLMReader(LLMReader):
    """TEST DOUBLE. Same interface as LLMReader; answers from hidden truth labels with injected noise, deterministically.

    fn_rate: chance a true mistake is labelled fine (a missed failure); fp_rate: chance a fine run is labelled with a
    mistake kind (a false flag; always a `mistake`-severity kind). Purpose, deployment and strata come from the heuristic
    reader, reworded. It exists so tests and demos can exercise the two-phase audit on labels that are wrong in both
    directions. It must never be the default and is refused by the CLI without --allow-test-double.
    """
    name = 'mock-llm-test-double'
    version = '1'

    def __init__(self, truth, fn_rate=.02, fp_rate=.04, seed=11):
        need(isinstance(truth, list) and truth, 'invalid_reader', 'reader', 'The mock reader is a test double and needs the hidden truth labels.')
        need(0 <= fn_rate < .5 and 0 <= fp_rate < .5, 'invalid_reader', 'reader', 'Noise rates are between 0 and 0.5.')
        self.truth = {(t['candidate_id'], t['case_id']): t for t in truth}
        self.fn_rate, self.fp_rate, self.seed = fn_rate, fp_rate, seed
        self._heuristic = HeuristicReader()

    def _draw(self, cand, case):
        h = int(digest({'mock': self.seed, 'c': cand, 'k': case})[:8], 16)
        rnd = mulberry32(h)
        return rnd(), rnd()

    def describe(self, records):
        d = self._heuristic.describe(records)
        return {'purpose': d['purpose'], 'deployment': {'mode': d['deployment']['mode'], 'why': 'From the tool calls and review fields in the sample: ' + d['deployment']['why'] + '.'}, 'summary': d.get('summary', '')}

    def propose_strata(self, records, labels=None):
        out = self._heuristic.propose_strata(records, labels)
        self.strata_rationale = getattr(self._heuristic, 'strata_rationale', '')
        return out

    def propose_taxonomy(self, records, standard=None):
        records = normalise_logs(records) if records and 'raw' not in records[0] else records
        seen = sorted({self.truth[(r['candidate_id'], r['case_id'])]['mistake'] for r in records if (r['candidate_id'], r['case_id']) in self.truth} - {None})
        kinds = {}
        for k in seen:
            if k in KIND_CATALOG: lab, sev, std = KIND_CATALOG[k]
            elif k.startswith('unwanted_'): lab, sev, std = _unwanted_kind(k.replace('unwanted_', '') + '.issue')[1], 'incident', 'unwanted_action'
            else: lab, sev, std = k.replace('_', ' ').capitalize(), 'mistake', None
            kinds[k] = {'label': lab, 'severity': sev, 'price_usd': None, 'tolerance_per_month': None, 'standard': std}
        return {'version': (standard or standard_taxonomy())['version'], 'kinds': kinds}

    def label(self, record, taxonomy):
        t = self.truth.get((record.get('candidate_id'), record.get('case_id')))
        if t is None: return {'mistake': None, 'rationale': 'not in the sample the model saw'}
        a, b = self._draw(record['candidate_id'], record['case_id'])
        kinds = taxonomy['kinds']
        if t['mistake'] is not None:
            if a < self.fn_rate: return {'mistake': None, 'rationale': 'looks completed (model missed the failure)'}
            std = KIND_CATALOG.get(t['mistake'], (None, None, None))[2]
            return {'mistake': _map_kind(t['mistake'], std, kinds), 'rationale': f"model: {kinds.get(t['mistake'], {}).get('label', t['mistake'])}"}
        if a < self.fp_rate:
            pool = sorted(k for k, v in kinds.items() if v['severity'] == 'mistake')
            if pool: return {'mistake': pool[int(b * len(pool))], 'rationale': 'model flagged a problem (a false flag)'}
        return {'mistake': None, 'rationale': 'completed'}


# ----------------------------------------------------------------------------- proposals

def _normal(v):
    if isinstance(v, str): return re.sub(r'\s+', ' ', v).strip().lower()
    if isinstance(v, dict): return {k: _normal(x) for k, x in sorted(v.items())}
    if isinstance(v, list): return [_normal(x) for x in v]
    return v


def blind_taxonomy(taxonomy):
    """What a reader is allowed to see: labels and severities, never prices or tolerances."""
    return {'version': taxonomy['version'], 'kinds': {k: {'label': v['label'], 'severity': v['severity'], 'standard': v['standard']} for k, v in taxonomy['kinds'].items()}}


def _engine_audit_sample(rows, n_fine, n_flagged, seed):
    from aver_policy import decision_v2 as v2
    return v2.audit_sample(rows, n_fine, n_flagged, seed)


def audit_size(rows, spec=None, n_fine=None, n_flagged=None):
    """How many AI-"fine" and AI-flagged rows a person must check (docs/130 §8.1). With a spec, the fine-row count comes
    from a quick plan's audit.fine_rows_needed (the size at which a pass becomes possible at the spec's failure limit);
    otherwise min(AI-fine rows, 400). Flagged rows: min(AI-flagged rows, 100)."""
    ai = [r for r in rows if r.get('origin') == 'ai-labeled']
    fine = sum(r['completed'] is True and r['incident'] is not True for r in ai)
    flagged = sum(r['completed'] is False or r['incident'] is True for r in ai)
    needed, limit, sized_from, for_limit = None, None, 'default', 'the failure limit'
    if spec is not None:
        from aver_policy import decision_v2 as v2
        limit = spec['requirements']['failure_limit']
        try:
            plan = v2.make_plan(spec, rows, len(spec['candidates']), 1, 0, None, 'clause', .01, 4)
            needed = (plan.get('audit') or {}).get('fine_rows_needed')
            key = (plan.get('audit') or {}).get('for_limit', 'average:failure')
            raw = v2.raw_limits(spec); limit = raw.get(key, limit)
            for_limit = ('the failure limit' if key == 'average:failure' else f"the tolerance for {key[len('tolerance:'):]}" if key.startswith('tolerance:')
                         else f"the limit for {key.split(':')[0]} cases")
            sized_from = 'plan' if needed is not None else 'default'
        except (DecisionError, KeyError, TypeError, ValueError):
            needed = None
    nf = n_fine if n_fine is not None else min(fine, needed if needed is not None else 400)
    nh = n_flagged if n_flagged is not None else min(flagged, 100)
    nf, nh = min(nf, fine), min(nh, flagged)
    if needed is not None:
        why = (f'A person must re-label {nf} runs the reader called fine and {nh} it flagged. The tightest limit is {for_limit} ({limit:.1%} of runs): about {needed} '
               f'fine-labelled runs must be checked with no missed failures for a pass to be possible at all; fewer than that and no evidence can pass. '
               f'The flagged runs measure how often the reader cries wolf.')
        if nf < needed: why += f' The logs hold only {fine} fine-labelled runs, short of {needed}: collect more before the audit can support a pass.'
    else:
        why = (f'A person must re-label {nf} runs the reader called fine and {nh} it flagged. This is not a formality: the fine-labelled runs bound how '
               f'many failures the reader misses, and at a few-percent failure limit a pass needs a few hundred of them checked with no misses '
               f'(about 324 at 3.2%). Set the failure limit in a spec to get the exact number.')
    return {'fine_rows': nf, 'flagged_rows': nh, 'fine_rows_needed': needed, 'ai_fine_rows': fine, 'ai_flagged_rows': flagged, 'sized_from': sized_from, 'why': why}


def read_logs(records, reader=None, taxonomy=None, audit_n=None, seed=7, spec=None, audit_flagged=None, prices=None, log_format=None):
    """docs/129 §5.1, docs/132 §2, docs/141. Returns an aver-log-proposal/v1.

    records: logged runs in any format log_formats reads (OpenTelemetry, run-record exports, observation-trace exports, Agents SDK, flat
    records, or CSV/JSON/JSONL text), detected from their structure.
    audit_n: the number of AI-"fine" rows to sample for the audit (default: sized by audit_size); audit_flagged likewise.
    spec: an accepted spec for the same rows, used only to size the audit from the plan.
    prices: {model: ($ per 1k input tokens, $ per 1k output tokens)} or 'model=in,out;...', for runs that log tokens but
    no cost. log_format: override detection (one of log_formats.FORMATS)."""
    reader = reader or HeuristicReader()
    hint = None
    if isinstance(records, str):
        need(bool(records.strip()), 'invalid_logs', 'logs', 'The export is empty.')
        records, hint = log_formats.load_text(records)
    need(isinstance(records, (list, dict)) and records, 'invalid_logs', 'logs', 'Supply a list of logged cases or an OTLP/JSON export.')
    raw_items = records if isinstance(records, list) else [records]
    need(all(isinstance(r, dict) for r in raw_items), 'invalid_logs', 'logs', 'Supply a list of logged cases.')
    adapted = log_formats.adapt(records, log_format, prices, hint)
    recs = normalise_logs(adapted['records'])
    need(bool(recs), 'invalid_logs', 'logs', 'The export holds no runs.')
    for i, r in enumerate(recs):
        need(valid_label(str(r.get('case_id') or '')) and valid_label(str(r.get('candidate_id') or '')), 'invalid_logs', f'logs[{i}]',
             'Every logged case needs a case id (case_id, ticket_id, task_id, document_id, ...) and a candidate (candidate_id, variant, config, model_id, ...).')
    seen_pairs = set()
    for i, r in enumerate(recs):
        need((r['candidate_id'], r['case_id']) not in seen_pairs, 'invalid_logs', f'logs[{i}]', 'One run per case and candidate; repeated runs cannot be read as independent evidence.')
        seen_pairs.add((r['candidate_id'], r['case_id']))
    if hasattr(reader, 'prepare'): reader.prepare(recs)
    described = reader.describe(recs)
    full = reader.propose_taxonomy(recs, taxonomy or standard_taxonomy())
    seen = blind_taxonomy(full)
    labels, rationales, unknown = {}, {}, set()
    for r in recs:
        lab = reader.label(r, seen)
        m = lab.get('mistake')
        need(m is None or m in full['kinds'], 'invalid_label', f'logs[{r["case_id"]}]', 'The reader returned a kind outside the taxonomy.')
        labels[(r['candidate_id'], r['case_id'])] = m
        if lab.get('unsure'): unknown.add((r['candidate_id'], r['case_id']))
        rationales[f"{r['candidate_id']}|{r['case_id']}"] = lab.get('rationale', '')
    try: strata = reader.propose_strata(recs, labels)
    except TypeError: strata = reader.propose_strata(recs)
    path = next(iter(strata.values()))['how']
    stratum_of = _stratum_assigner(recs, strata)
    origin, outcome_field = reader.labels_origin(recs) if hasattr(reader, 'labels_origin') else ('ai-labeled', None)
    case_digest = {}
    for r in sorted(recs, key=lambda r: (r['case_id'], r['candidate_id'])):
        case_digest.setdefault(r['case_id'], digest({'case': r['case_id'], 'input': _jsonable(r.get('input')), 'expected': _jsonable(r.get('expected'))}))
    rows = []
    for r in recs:
        m = labels[(r['candidate_id'], r['case_id'])]
        sev = full['kinds'][m]['severity'] if m else None
        rows.append({'schema': 'aver-outcomes/v2', 'case_id': r['case_id'], 'case_sha256': case_digest[r['case_id']],
                     'candidate_id': r['candidate_id'], 'configuration_sha256': digest({'configuration': r['candidate_id']}), 'stratum': stratum_of(r), 'phase': 'pilot',
                     'origin': origin, 'completed': None if (r['candidate_id'], r['case_id']) in unknown else m is None, 'incident': sev == 'incident', 'cost_usd': r.get('cost_usd'), 'mistake': m})
    if origin == 'customer':
        thresholds = sorted({k for k in full['kinds'] if any(reader.finding(r).get('basis') == 'threshold' and labels[(r['candidate_id'], r['case_id'])] == k for r in recs)})
        labels_from = {'origin': 'customer', 'field': outcome_field,
                       'why': (f'Every run carries a pass/fail the harness recorded ({outcome_field}); a failed run is labelled from it'
                               + (f", and {', '.join(thresholds)} comes from a fixed threshold on logged numbers (stated in each label's rationale)" if thresholds else '')
                               + '. No label rests on the reader\'s judgement, so the labels are the team\'s own (origin customer) and need no audit.')}
        size = {'fine_rows': 0, 'flagged_rows': 0, 'fine_rows_needed': None, 'ai_fine_rows': 0, 'ai_flagged_rows': 0, 'sized_from': 'not_needed',
                'why': 'No audit is needed: the labels are the outcomes the harness recorded, not a reader\'s reading of the runs.'}
        sample = []
    else:
        labels_from = {'origin': 'ai-labeled', 'field': None, 'why': 'The labels are the reader\'s reading of the runs; a person audits a random sample before any pass.'}
        size = audit_size(rows, spec, audit_n, audit_flagged)
        sample = _engine_audit_sample(rows, size['fine_rows'], size['flagged_rows'], seed)
    mode = described['deployment']['mode']
    defaults = {k: DEPLOYMENT_DEFAULTS[mode].get(full['kinds'][k]['standard'] or k, 'price') for k in full['kinds']}
    evidence = {}
    for row in sorted(rows, key=lambda x: (x['candidate_id'], x['case_id'])):
        m = row['mistake']
        if not m: continue
        e = evidence.setdefault(m, {'count': 0, 'by_candidate': {}, 'examples': []})
        e['count'] += 1; e['by_candidate'][row['candidate_id']] = e['by_candidate'].get(row['candidate_id'], 0) + 1
        if len(e['examples']) < 5 and row['case_id'] not in e['examples']: e['examples'].append(row['case_id'])
    for m in full['kinds']: evidence.setdefault(m, {'count': 0, 'by_candidate': {}, 'examples': []})
    strata_why = getattr(reader, 'strata_rationale', None) or f'Strata from {path}.'
    return {'schema': 'aver-log-proposal/v1', 'purpose': described['purpose'], 'deployment': described['deployment'], 'summary': described.get('summary', ''),
            'strata': strata, 'strata_field': path.split(' as logged')[0], 'taxonomy': full, 'taxonomy_evidence': evidence, 'defaults': defaults,
            'rows': rows, 'audit': size, 'audit_sample': sample, 'labels_from': labels_from,
            'rationales': {'purpose': described.get('purpose_why', 'From the tool names, output fields and case-id field of the logs.'), 'deployment': described['deployment']['why'],
                           'strata': strata_why, 'taxonomy': 'Standard kinds with evidence in the logs, specialised to this agent; each keeps its standard parent.',
                           'labels': rationales},
            'reader': {'name': reader.name, 'version': reader.version, **({'provenance': reader.provenance} if hasattr(reader, 'provenance') else {})},
            **_evidence_summary(recs, adapted, prices),
            'note': ('A proposal. Accept or edit the kinds and strata; the labels are the harness\'s recorded outcomes, so no audit is needed.' if origin == 'customer'
                     else 'A proposal. Accept or edit the kinds and strata; a person must audit the sample before any pass.')}


def _evidence_summary(recs, adapted, prices):
    """What the export says about format, options, tools, cost and latency (docs/141), for the proposal. No content."""
    av = [r.get('aver') or {} for r in recs]
    tools = sorted({t['name'] for r in recs for t in (r.get('tool_calls') or [])})
    fields = sorted({f for x in av for f in (x.get('cost_fields') or [])} | ({k for r in recs for k in COST_KEYS if isinstance((r.get('raw') or {}).get(k), (int, float))} if not any(av) else set()))
    known = sum(r.get('cost_usd') is not None for r in recs)
    priced = sum(x.get('cost_basis') == 'priced' for x in av)
    unpriced = sum(x.get('cost_basis') == 'tokens_unpriced' for x in av)
    if known == len(recs): why = f'Cost is known for every run ({priced} priced from tokens with your price table).' if priced else 'Cost is logged for every run.'
    else:
        why = f'Cost is unknown for {len(recs) - known} of {len(recs)} runs'
        why += (f': {unpriced} log tokens but ' + ('no price table was given (--prices model=$/1k_in,$/1k_out)' if not prices else 'the price table has no entry for their model')) if unpriced else ': nothing logged gives a cost or token count'
        why += '.' + (f' {priced} were priced from tokens.' if priced else '')
    lat = [x['latency_s'] for x in av if x.get('latency_s') is not None]
    lfield = Counter(x['latency_field'] for x in av if x.get('latency_field'))
    calls = [x.get('llm_calls') or 0 for x in av]
    out_rule = adapted.get('outcome_rule')
    return {'format': {'name': adapted['format'], 'reason': adapted['reason']},
            'candidates_rule': adapted.get('candidates_rule') or 'The case-id and configuration fields of each record.',
            'tools': tools,
            'cost': {'fields': fields, 'known_runs': known, 'priced_runs': priced, 'unknown_runs': len(recs) - known, 'why': why},
            'latency': {'field': lfield.most_common(1)[0][0] if lfield else None, 'median_s': _median(lat), 'runs': len(lat)},
            'calls': {'llm_calls_median': _median(calls) if any(calls) else None, 'tool_calls_median': _median([len(r.get('tool_calls') or []) for r in recs])},
            'outcome_fields': {'used': adapted.get('outcome_fields') or [], 'skipped': adapted.get('skipped_scores') or [], 'rule': out_rule}}


def _jsonable(x):
    try:
        digest(x); return x
    except DecisionError:
        return json.loads(json.dumps(x, default=str))


def stratum_from_proposal(proposal):
    """record -> kind of case, as the proposal assigns it. For a harness that must draw new cases of each kind (docs/137)."""
    return _stratum_assigner(None, proposal['strata'])


def _stratum_assigner(records, strata):
    how = next(iter(strata.values()))['how']
    if how.endswith(' as logged' + POOLED):
        path = how[:-len(' as logged' + POOLED)]
        other = 'other (pooled)' if 'other (pooled)' in strata else 'other'
        return lambda r: (lambda v: v if v in strata and v != other else other)(_strat_value(r, path))
    if how.endswith(' as logged') and how != 'meta.kind as logged':
        path = how[:-len(' as logged')]
        return lambda r: _strat_value(r, path)
    if how == 'meta.kind as logged':
        return lambda r: (r.get('meta') or {}).get('kind')
    if how.startswith('input length tertile'):
        t1, t2 = (int(x) for x in re.findall(r'\d+', how)[:2])
        return lambda r: 'short' if len(str(r.get('input') or '')) < t1 else 'long' if len(str(r.get('input') or '')) >= t2 else 'typical'
    return lambda r: next(iter(strata))


def proposal_to_spec(proposal, scope, requirements, monthly_uses, label_cost_usd, horizon_months=12, candidates=None, baseline=None, budget_runs=None, purpose=None):
    """Turn an accepted proposal into an aver-decision-spec/v3. Candidates come from the logs unless given. `purpose`: the
    person's own sentence when they restate the reader's (it becomes the population statement)."""
    ids = candidates or sorted({r['candidate_id'] for r in proposal['rows']})
    strata = {k: {'weight': v['weight'], 'failure_limit': v.get('failure_limit')} for k, v in proposal['strata'].items()}
    lf = proposal.get('labels_from') or {}
    if lf.get('origin') == 'customer':
        runaway = any(v.get('standard') == 'runaway_cost' for v in proposal['taxonomy']['kinds'].values())
        definitions = {'completion': f"{lf['field']} as the harness recorded it" + ("; a passing run over the reader's fixed runaway threshold counts as a runaway_cost mistake." if runaway else '.'),
                       'incident': 'A mistake kind of severity incident.', 'grader': f"the harness's recorded outcome ({lf['field']}); thresholds by the {proposal['reader']['name']} reader v{proposal['reader']['version']}; no audit"}
    else:
        definitions = {'completion': 'As labelled by the log reader and audited.', 'incident': 'A mistake kind of severity incident.', 'grader': f"{proposal['reader']['name']} reader v{proposal['reader']['version']}, audited"}
    statement = purpose.strip() if isinstance(purpose, str) and purpose.strip() else proposal['purpose']
    spec = {'schema': 'aver-decision-spec/v3', 'scope': scope,
            'candidates': [{'id': c, 'name': c, 'digest': digest({'configuration': c})} for c in ids], 'baseline': baseline or ids[0],
            'strata': strata, 'definitions': definitions,
            'requirements': requirements, 'monthly_uses': monthly_uses, 'consequence_usd': None, 'budget_runs': budget_runs or 4 * len(proposal['rows']),
            'population_statement': statement, 'label_cost_usd': label_cost_usd, 'horizon_months': horizon_months, 'current_failure_rate': None,
            'grader_error': {'false_negative': 0, 'false_positive': 0}, 'taxonomy': {'version': proposal['taxonomy']['version'], 'kinds': {k: dict(v) for k, v in proposal['taxonomy']['kinds'].items()}},
            'deployment': {'mode': proposal['deployment']['mode'], 'description': statement}}
    from aver_policy.decision_v2 import validate_spec_v2
    return validate_spec_v2(spec)


def load_truth(path):
    """Hidden labels for tests and the mock reader only. Readers never call this."""
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]
