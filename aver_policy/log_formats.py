"""Log formats (docs/141): turn the trace and results exports teams actually have into the flat runs the log reader reads.

One adapter per format, chosen from the structure of the export, never from a file name:

  * otel           OpenTelemetry spans with GenAI semantic-convention attributes: an OTLP/JSON export (resourceSpans) or a
                   flat list / JSONL of spans (the SDK console exporter's shape, or one OTLP span per line).
  * runs           a run export from an observability tool (run_type, inputs/outputs, parent_run_id, error, feedback_stats, total_cost, ...).
  * observations   a trace export from an observability tool: traces with observations (the stored {trace, observations} shape, the public API's
                   TraceWithFullDetails, or its {data, meta} page) and scores.
  * agents_sdk  Agents SDK trace exports ({"data": [trace, trace.span, ...]}): agent, function, generation,
                   handoff and guardrail spans.
  * generic        flat result records with a case-id field (the shape the reader has always read); passed through as is.
  * csv            a CSV results table (a header row, one run per row); dotted columns are nested (input.q -> input.q).

Every adapter produces flat records the reader's normaliser understands, plus a private `_aver` block with what the
format can tell (tools, models, agents, handoffs, guardrails, errors, scores, tokens, cost, latency). Nothing in `_aver`
is ever written into an outcome row; the system prompt is kept to its first 200 characters and only read for purpose.

Run, case and option (docs/141 §2):
  * one run per root span (otel), root run (runs), trace (observations, agents_sdk) or row (generic, csv);
  * the option is an explicit configuration attribute when every run has one; otherwise the model (or agent) name, but
    only when the runs share a case key (an example id, a session, a trace name) across different models, so that the
    same case was run by each option; otherwise there is one option ('logged', or the one model every run used).

Recorded outcomes (docs/141 §3): a score or feedback field becomes a pass/fail label only when it is on every run and
its scale and direction are known (values in [0, 1], or pass/fail words; direction from the name: toxicity,
hallucination, ... are lower-is-better). Errors and tripped guardrails are recorded failures, but they say nothing
about runs without them, so on their own they do not make the labels the team's own.

Stdlib only (the CLI's import allow-list, docs/104 §1).
"""
from __future__ import annotations

import csv
import io
import json
import re
from collections import Counter, defaultdict
from datetime import datetime

from aver_policy.decision import DecisionError, need

TRACE_FORMATS = ('otel', 'runs', 'observations', 'agents_sdk')
FORMATS = TRACE_FORMATS + ('generic', 'csv')
# Must match log_reader.CASE_KEYS / CANDIDATE_KEYS (log_reader imports this module, so the lists live here too).
CASE_KEYS = ('case_id', 'example_id', 'ticket_id', 'task_id', 'document_id', 'instance_id', 'sample_id', 'item_id', 'input_id', 'conversation_id')
EXPLICIT_CANDIDATE_KEYS = ('candidate_id', 'variant', 'config', 'configuration', 'agent_config', 'pipeline_config', 'arm', 'candidate', 'agent_id')
SYSTEM_PROMPT_CHARS = 200
LF_TYPES = {'GENERATION', 'SPAN', 'EVENT', 'AGENT', 'TOOL', 'CHAIN', 'RETRIEVER', 'EVALUATOR', 'EMBEDDING', 'GUARDRAIL'}
LLM_OPS = {'chat', 'text_completion', 'generate_content', 'completion', 'responses'}

# Score direction, by name. Checked in this order: harm (an incident), other lower-is-better (a mistake), then
# higher-is-better. A name in none of them has an unknown direction and is not used as an outcome.
SCORE_HARM = ('toxic', 'harm', 'unsafe', 'jailbreak', 'pii', 'bias', 'violat', 'nsfw', 'offens', 'insecure')
SCORE_LOWER = ('hallucinat', 'incorrect', 'error', 'wrong', 'fail', 'mistake')
SCORE_HIGHER = ('correct', 'accura', 'resolv', 'pass', 'success', 'quality', 'helpful', 'relevan', 'faithful', 'ground', 'exact', 'match',
                'f1', 'score', 'complete', 'valid', 'good', 'useful', 'rating', 'thumb', 'user_feedback', 'feedback', 'reward')
# CSV columns read as scores without a feedback. / scores. prefix (booleans such as passed or resolved are read as
# recorded outcomes by the reader itself)
BARE_SCORE_COLUMNS = ('score', 'reward', 'correctness', 'accuracy', 'toxicity', 'hallucination', 'grade')
PASS_WORDS = {'pass', 'passed', 'correct', 'yes', 'true', 'good', 'resolved', 'success', 'thumbs_up', 'up', '1'}
FAIL_WORDS = {'fail', 'failed', 'incorrect', 'no', 'false', 'bad', 'unresolved', 'failure', 'thumbs_down', 'down', '0'}
GENERIC_NAMES = {'agent workflow', 'chatopenai', 'chat', 'response', 'run', 'llm', 'langgraph', 'invoke_agent', 'agent', 'chain', 'runnablesequence',
                 'openai.chat', 'openai.chat.completions', 'chat_completion', 'completion', 'generatecontent', 'generate_content', 'ai.generatetext',
                 'claude.conversation', 'default', 'trace', 'root', 'workflow', 'main'}
PLACEHOLDER = re.compile(r'^(foo|bar|baz|qux|test\w*|some_\w+|my_\w+|example\w*|dummy\w*|sample\w*|tool_?\d*|agent_?\d*|run ?\d+|func\w*|demo\w*)$', re.I)
UUIDISH = re.compile(r'^[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}$|^[0-9a-f]{16,}$', re.I)
DENIED = re.compile(r'denied|not allowed|rejected|permission|unauthori[sz]ed|forbidden|policy|blocked', re.I)


# ----------------------------------------------------------------------------- loading text

def load_text(text):
    """Text of an export -> (object, hint). JSON (object or array) or JSONL first; otherwise a CSV table with a header
    row. The hint is 'csv' for a CSV table, else None. Decided from the text itself, never from a file name."""
    s = text.lstrip('\ufeff')
    st = s.lstrip()
    if st[:1] in '{[':
        try:
            return json.loads(st), None
        except ValueError:
            lines = [l for l in st.splitlines() if l.strip()]
            try:
                return [json.loads(l) for l in lines], None
            except ValueError:
                pass
            raise DecisionError('unreadable_input', 'logs', 'The export starts like JSON but is neither a JSON document nor JSON Lines.')
    lines = [l for l in s.splitlines() if l.strip()]
    if len(lines) >= 2:
        try:
            dialect = csv.Sniffer().sniff(lines[0] + '\n' + lines[1], delimiters=',;\t|')
        except csv.Error:
            dialect = None
        if dialect is not None:
            rows = list(csv.DictReader(io.StringIO('\n'.join(lines)), dialect=dialect))
            if rows and len(rows[0]) >= 2 and all(k is not None and k.strip() for k in rows[0]):
                return rows, 'csv'
    raise DecisionError('unknown_log_format', 'logs', 'The export is neither JSON, JSON Lines nor a CSV table with a header row.')


# ----------------------------------------------------------------------------- detection

def _dicts(x):
    return isinstance(x, list) and bool(x) and all(isinstance(i, dict) for i in x)


def _seen_keys(items):
    c = Counter(k for i in items[:50] if isinstance(i, dict) for k in i)
    return sorted(c, key=lambda k: (-c[k], k))[:20]


def _unwrap(obj):
    """Envelopes -> (items, note). {'data': [...]} (Agents SDK batches, observation-trace pages), {'runs': [...]}."""
    if isinstance(obj, dict):
        if 'resourceSpans' in obj: return [obj], ''
        for k in ('data', 'runs', 'traces', 'spans', 'items'):
            if _dicts(obj.get(k)) and not any(x in obj for x in ('trace', 'observations', 'run_type')):
                return obj[k], f' (inside a "{k}" envelope)'
        return [obj], ''
    return obj, ''


def _is_span(x):
    ctx = x.get('context') if isinstance(x.get('context'), dict) else {}
    ids = ('trace_id' in ctx and 'span_id' in ctx) or (('traceId' in x or 'trace_id' in x) and ('spanId' in x or 'span_id' in x))
    return ids and 'name' in x and 'run_type' not in x and 'object' not in x and ('attributes' in x or 'status' in x)


def detect(obj, hint=None):
    """The format of an export, from its structure: (name, reason). Raises unknown_log_format, naming the keys it saw."""
    if hint == 'csv':
        return 'csv', 'a CSV table with a header row'
    items, env = _unwrap(obj)
    need(_dicts(items), 'unknown_log_format', 'logs', 'Expected a JSON object, an array of objects, JSON Lines or a CSV table.')
    n = len(items)
    if all('resourceSpans' in i for i in items):
        return 'otel', 'an OTLP/JSON export: objects with resourceSpans (OpenTelemetry)' + env
    if all(i.get('object') in ('trace', 'trace.span') for i in items):
        return 'agents_sdk', 'objects with "object": "trace" / "trace.span" and span_data (Agents SDK trace export)' + env
    if all('trace' in i and 'observations' in i for i in items) or all(isinstance(i.get('observations'), list) and 'id' in i and 'run_type' not in i for i in items):
        return 'observations', 'traces with an observations list (a trace export from an observability tool)' + env
    if all(i.get('type') in LF_TYPES and ('traceId' in i) and ('startTime' in i) for i in items):
        return 'observations', 'observations with traceId, type (GENERATION / SPAN / ...) and startTime (an observation export from an observability tool)' + env
    if all('run_type' in i for i in items) and all(any(k in i for k in ('inputs', 'outputs', 'trace_id', 'dotted_order', 'parent_run_id', 'start_time')) for i in items):
        return 'runs', 'runs with run_type and inputs / outputs / trace_id (a run export from an observability tool)' + env
    if all(_is_span(i) for i in items):
        return 'otel', 'a flat list of spans with trace and span ids, a name and attributes (OpenTelemetry)' + env
    with_case = sum(any(i.get(k) is not None for k in CASE_KEYS) for i in items)
    if with_case and with_case >= n / 2:
        key = Counter(k for i in items for k in CASE_KEYS if i.get(k) is not None).most_common(1)[0][0]
        return 'generic', f'flat result records with a case-id field ({key})'
    raise DecisionError('unknown_log_format', 'logs',
                        'Could not tell the log format from its structure. Keys seen: ' + ', '.join(_seen_keys(items)) + '. Known: OpenTelemetry '
                        '(resourceSpans, or spans with trace/span ids and attributes), run exports (run_type), trace exports (observations), '
                        'Agents SDK exports (object: trace / trace.span), result records with a case-id field (case_id, example_id, task_id, ...), '
                        'or a CSV table. Pass --format to choose one.')


# ----------------------------------------------------------------------------- small helpers

def _num(v):
    if isinstance(v, bool): return None
    if isinstance(v, (int, float)): return v
    if isinstance(v, str):
        try: return float(v) if re.search(r'[.eE]', v) else int(v)
        except ValueError: return None
    return None


def _maybe_json(s):
    if isinstance(s, str) and s[:1] in '{[':
        try: return json.loads(s)
        except ValueError: return s
    return s


def _ts(x):
    """ISO string, Unix seconds or nanoseconds (int, str, or {low, high}) -> float seconds."""
    if x is None or x == '': return None
    if isinstance(x, dict) and 'low' in x and 'high' in x:
        return ((x['high'] & 0xffffffff) * 2 ** 32 + (x['low'] & 0xffffffff)) / 1e9
    n = _num(x)
    if n is not None:
        return n / 1e9 if n > 1e14 else n / 1e3 if n > 1e11 else float(n)
    if isinstance(x, str):
        s = x.strip().replace('Z', '+00:00')
        m = re.match(r'^(.*T\d\d:\d\d:\d\d)(\.\d+)?(.*)$', s)
        if m:
            frac = (m.group(2) or '')[:7]
            s = m.group(1) + frac + m.group(3)
        try:
            return datetime.fromisoformat(s).timestamp()
        except ValueError:
            return None
    return None


def _hex(v):
    if isinstance(v, dict) and v.get('type') == 'Buffer' and isinstance(v.get('data'), list):
        return ''.join(f'{b:02x}' for b in v['data'])
    if isinstance(v, str) and v.lower().startswith('0x'):
        return v[2:]
    return v if v is None else str(v)


def _otel_value(v):
    if not isinstance(v, dict): return v
    for k in ('stringValue', 'boolValue', 'doubleValue'):
        if k in v: return v[k]
    if 'intValue' in v:
        iv = v['intValue']
        if isinstance(iv, dict): return ((iv.get('high', 0) << 32) | (iv.get('low', 0) & 0xffffffff))
        return int(iv)
    if 'arrayValue' in v: return [_otel_value(x) for x in (v['arrayValue'] or {}).get('values', [])]
    if 'kvlistValue' in v: return {x['key']: _otel_value(x['value']) for x in (v['kvlistValue'] or {}).get('values', [])}
    return None


def _attrs(a):
    if isinstance(a, dict): return dict(a)
    if isinstance(a, list): return {x['key']: _otel_value(x.get('value')) for x in a if isinstance(x, dict) and 'key' in x}
    return {}


def _first200(text):
    if text is None: return None
    if isinstance(text, (list, dict)):
        parts = []
        def walk(x):
            if isinstance(x, str): parts.append(x)
            elif isinstance(x, dict):
                for k in ('content', 'text'):
                    if isinstance(x.get(k), (str, list, dict)): walk(x[k])
            elif isinstance(x, list):
                for y in x: walk(y)
        walk(text); text = ' '.join(parts)
    text = str(text).strip()
    return text[:SYSTEM_PROMPT_CHARS] or None


def _system_from_messages(msgs):
    """A system message's text from a chat message list in the usual shapes (role/type 'system'; LangChain kwargs)."""
    msgs = _maybe_json(msgs)
    if isinstance(msgs, dict):
        if isinstance(msgs.get('system'), (str, list)): return _first200(msgs['system'])
        for k in ('messages', 'input', 'prompt'):
            if k in msgs: return _system_from_messages(msgs[k])
        return None
    if not isinstance(msgs, list): return None
    flat = []
    for m in msgs:
        if isinstance(m, list): flat.extend(m)
        else: flat.append(m)
    for m in flat:
        if not isinstance(m, dict): continue
        idv = m.get('id')
        role = m.get('role') or m.get('type') or (idv[-1] if isinstance(idv, list) and idv else '')
        if str(role).lower() in ('system', 'systemmessage', 'developer'):
            body = m.get('content') if 'content' in m else m.get('parts') if 'parts' in m else (m.get('kwargs') or {}).get('content')
            return _first200(body)
    return None


def _tool_label(name):
    s = str(name or '').strip()
    for p in ('execute_tool ', 'running tool: ', 'tool: ', 'tool.'):
        if s.lower().startswith(p): s = s[len(p):]
    return s or '?'


def _leafmost(nodes, children, has):
    """Nodes that carry a value (has(n) true) and have no descendant that carries one: avoids counting a parent's
    aggregate and its children's values twice."""
    memo = {}
    def below(i):
        if i not in memo: memo[i] = any(has(nodes[c]) or below(c) for c in children.get(i, ()))
        return memo[i]
    return [n for i, n in nodes.items() if has(n) and not below(i)]


def _descendants(root, children):
    out, stack = [], [root]
    while stack:
        i = stack.pop(); out.append(i)
        stack.extend(children.get(i, ()))
    return out


# ----------------------------------------------------------------------------- a run, independent of format

def _run(**kw):
    base = {'run_id': None, 'explicit_case': None, 'explicit_candidate': None, 'example_id': None, 'session_id': None, 'run_name': None,
            'models': [], 'agents': [], 'tools': [], 'tool_errors': [], 'handoffs': 0, 'guardrails': [], 'llm_calls': 0,
            'tokens': {}, 'cost_logged': None, 'cost_fields': set(), 'token_fields': set(), 'latency_s': None, 'latency_field': None,
            'error': None, 'span_errors': [], 'scores': {}, 'system_prompt': None, 'input': None, 'output': None, 'expected': None,
            'output_logged': False, 'meta': {}}
    base.update(kw)
    return base


def _add_tokens(run, model, tin, tout, fin, fout):
    if tin is None and tout is None: return
    t = run['tokens'].setdefault(model or '?', [0, 0])
    t[0] += int(tin or 0); t[1] += int(tout or 0)
    if tin is not None and fin: run['token_fields'].add(fin)
    if tout is not None and fout: run['token_fields'].add(fout)


# ----------------------------------------------------------------------------- OpenTelemetry

def _otel_spans(obj):
    """Either shape -> [{trace_id, span_id, parent_id, name, start, end, attrs, events, status, status_msg, resource}]."""
    items, _ = _unwrap(obj)
    out = []
    for it in items:
        if 'resourceSpans' in it:
            for rs in it.get('resourceSpans') or []:
                res = _attrs((rs.get('resource') or {}).get('attributes'))
                for ss in rs.get('scopeSpans', rs.get('instrumentationLibrarySpans', [])) or []:
                    for s in ss.get('spans', []) or []:
                        out.append(_span(s, res))
        else:
            out.append(_span(it, _attrs((it.get('resource') or {}).get('attributes')) if isinstance(it.get('resource'), dict) else {}))
    return out


def _span(s, res):
    ctx = s.get('context') if isinstance(s.get('context'), dict) else {}
    st = s.get('status') or {}
    code = st.get('code', st.get('status_code', ''))
    code = 'ERROR' if code in (2, '2', 'STATUS_CODE_ERROR', 'ERROR', 'StatusCode.ERROR') else 'OK' if code in (1, '1', 'STATUS_CODE_OK', 'OK') else 'UNSET'
    events = [{'name': e.get('name'), 'attrs': _attrs(e.get('attributes'))} for e in (s.get('events') or []) if isinstance(e, dict)]
    return {'trace_id': _hex(ctx.get('trace_id') or s.get('traceId') or s.get('trace_id')),
            'span_id': _hex(ctx.get('span_id') or s.get('spanId') or s.get('span_id')),
            'parent_id': _hex(s.get('parentSpanId') or s.get('parent_id') or s.get('parent_span_id')) or None,
            'name': str(s.get('name') or ''), 'start': _ts(s.get('startTimeUnixNano') or s.get('start_time')), 'end': _ts(s.get('endTimeUnixNano') or s.get('end_time')),
            'attrs': _attrs(s.get('attributes')), 'events': events, 'status': code, 'status_msg': st.get('message') or st.get('description') or '', 'resource': res}


def _otel_text(span, which):
    """Prompt or completion text of a span, from any of the conventions in use."""
    a = span['attrs']
    keys = ('input.value', 'gen_ai.input.messages', 'gen_ai.prompt') if which == 'in' else ('output.value', 'gen_ai.output.messages', 'gen_ai.completion')
    for k in keys:
        if a.get(k) not in (None, ''): return _maybe_json(a[k])
    idx = sorted(k for k in a if re.match(rf"gen_ai\.{'prompt' if which == 'in' else 'completion'}\.\d+\.content$", k))
    if idx: return a[idx[-1]] if which == 'out' else a[idx[0]]
    names = ('gen_ai.content.prompt', 'gen_ai.user.message') if which == 'in' else ('gen_ai.content.completion', 'gen_ai.choice')
    for e in span['events']:
        if e['name'] in names:
            ea = e['attrs']
            return _maybe_json(ea.get('gen_ai.prompt') or ea.get('gen_ai.completion') or ea.get('content') or ea.get('message'))
    return None


def _from_otel(obj):
    spans = _otel_spans(obj)
    by_trace = defaultdict(list)
    for s in spans: by_trace[s['trace_id']].append(s)
    runs = []
    for tid in sorted(by_trace, key=str):
        ss = by_trace[tid]
        ids = {s['span_id']: s for s in ss}
        children = defaultdict(list)
        for s in ss:
            if s['parent_id'] and s['parent_id'] in ids: children[s['parent_id']].append(s['span_id'])
        roots = sorted((s for s in ss if not s['parent_id'] or s['parent_id'] not in ids), key=lambda s: (s['start'] or 0, s['span_id'] or ''))
        for root in roots:
            members = {i: ids[i] for i in _descendants(root['span_id'], children)}
            runs.append(_otel_run(tid, root, members, children, len(roots) > 1))
    return runs


def _otel_run(tid, root, members, children, many_roots):
    a = root['attrs']
    run = _run(run_id=f"{tid}:{root['span_id']}" if many_roots else tid, run_name=root['name'])
    run['explicit_case'] = _find_key(a, CASE_KEYS)
    run['explicit_candidate'] = _find_key(a, EXPLICIT_CANDIDATE_KEYS)
    run['session_id'] = a.get('session.id') or a.get('gen_ai.conversation.id')
    run['example_id'] = a.get('example.id') or a.get('dataset.example.id')
    ordered = sorted(members.values(), key=lambda s: (s['start'] or 0, s['span_id'] or ''))
    parent_name = {s['span_id']: members[s['parent_id']]['name'] if s['parent_id'] in members else '' for s in ordered}
    for s in ordered:
        sa = s['attrs']; op = sa.get('gen_ai.operation.name')
        if op == 'execute_tool' or 'gen_ai.tool.name' in sa or s['name'].startswith('execute_tool') or (parent_name[s['span_id']].lower() == 'tools' and not sa.get('gen_ai.request.model')):
            run['tools'].append({'name': _tool_label(sa.get('gen_ai.tool.name') or s['name']), 'args': _maybe_json(sa.get('gen_ai.tool.call.arguments')) or {}})
            if s['status'] == 'ERROR': run['tool_errors'].append({'tool': run['tools'][-1]['name'], 'message': s['status_msg']})
        agent = sa.get('gen_ai.agent.name')
        m = re.match(r'^(?:agent_run|invoke_agent|agent)\s*\[?\s*([^\]]+?)\]?$', s['name']) if not agent else None
        if agent or (m and op != 'chat'): run['agents'].append(str(agent or m.group(1)))
        if op in LLM_OPS or sa.get('gen_ai.request.model') and (op is None or op in LLM_OPS):
            run['llm_calls'] += 1
            mdl = sa.get('gen_ai.request.model') or sa.get('gen_ai.response.model')
            if mdl: run['models'].append(str(mdl))
        sp = sa.get('gen_ai.system_instructions')
        if sp and not run['system_prompt']: run['system_prompt'] = _first200(_maybe_json(sp))
        for e in s['events']:
            if e['name'] == 'gen_ai.system.message' and not run['system_prompt']: run['system_prompt'] = _first200(e['attrs'].get('content'))
            if e['name'] == 'gen_ai.evaluation.result':
                n, v = e['attrs'].get('gen_ai.evaluation.name'), e['attrs'].get('gen_ai.evaluation.score.value', e['attrs'].get('gen_ai.evaluation.score.label'))
                if n is not None and v is not None: run['scores'][str(n)] = v
        if not run['system_prompt'] and sa.get('gen_ai.input.messages'): run['system_prompt'] = _system_from_messages(sa['gen_ai.input.messages'])
        if s is not root and s['status'] == 'ERROR' and (op == 'invoke_agent' or agent) and not run['error']:
            run['error'] = _otel_error(s)
        elif s is not root and s['status'] == 'ERROR':
            run['span_errors'].append(s['name'])
    # tokens and cost on the leaf-most spans that carry them (an agent span's aggregate is not counted again)
    def tok(s):
        x = s['attrs']; return any(k in x for k in ('gen_ai.usage.input_tokens', 'gen_ai.usage.output_tokens', 'gen_ai.usage.prompt_tokens', 'gen_ai.usage.completion_tokens'))
    for s in _leafmost(members, children, tok):
        x = s['attrs']
        fin = 'gen_ai.usage.input_tokens' if 'gen_ai.usage.input_tokens' in x else 'gen_ai.usage.prompt_tokens' if 'gen_ai.usage.prompt_tokens' in x else None
        fout = 'gen_ai.usage.output_tokens' if 'gen_ai.usage.output_tokens' in x else 'gen_ai.usage.completion_tokens' if 'gen_ai.usage.completion_tokens' in x else None
        _add_tokens(run, x.get('gen_ai.request.model') or x.get('gen_ai.response.model') or (run['models'][0] if run['models'] else None),
                    _num(x.get(fin)) if fin else None, _num(x.get(fout)) if fout else None, fin, fout)
    def costk(s): return next((k for k in s['attrs'] if k.endswith('cost_usd') or k == 'gen_ai.usage.cost'), None)
    costed = _leafmost(members, children, lambda s: costk(s) is not None and _num(s['attrs'][costk(s)]) is not None)
    if costed:
        run['cost_logged'] = sum(_num(s['attrs'][costk(s)]) for s in costed)
        run['cost_fields'].update(costk(s) for s in costed)
    if root['start'] is not None and root['end'] is not None:
        run['latency_s'] = max(0., root['end'] - root['start']); run['latency_field'] = 'span start/end time'
    if root['status'] == 'ERROR': run['error'] = _otel_error(root)
    llm = [s for s in ordered if s['attrs'].get('gen_ai.operation.name') in LLM_OPS or s['attrs'].get('gen_ai.request.model')]
    run['input'] = _otel_text(root, 'in') if _otel_text(root, 'in') is not None else next((t for s in llm for t in [_otel_text(s, 'in')] if t is not None), None)
    run['output'] = _otel_text(root, 'out') if _otel_text(root, 'out') is not None else next((t for s in reversed(llm) for t in [_otel_text(s, 'out')] if t is not None), None)
    run['output_logged'] = run['output'] is not None
    ref = next((a[k] for k in ('eval.reference', 'eval.expected', 'eval.expected_output', 'reference', 'expected') if k in a), None)
    run['expected'] = _maybe_json(ref)
    run['meta'] = {k: v for k, v in a.items() if isinstance(v, (str, bool, int, float)) and k not in ('input.value', 'output.value') and not k.startswith(('eval.', 'gen_ai.'))
                   and k not in ('session.id', 'user.id')}
    for k in ('deployment.environment', 'deployment.environment.name', 'service.name'):
        if isinstance(root['resource'].get(k), str): run['meta']['resource.' + k] = root['resource'][k]
    return run


def _find_key(attrs, keys):
    for k in keys:
        for a, v in attrs.items():
            if a == k or a.endswith('.' + k): return v
    return None


def _otel_error(s):
    ev = next((e for e in s['events'] if e['name'] == 'exception'), None)
    ea = ev['attrs'] if ev else {}
    return {'class': str(ea.get('exception.type') or 'error'), 'message': str(ea.get('exception.message') or s['status_msg'] or '')}


# ----------------------------------------------------------------------------- run-record exports

def _ls_model(r):
    ex = r.get('extra') or {}
    md = ex.get('metadata') or {}
    ip = ex.get('invocation_params') or {}
    out = r.get('outputs') if isinstance(r.get('outputs'), dict) else {}
    inp = r.get('inputs') if isinstance(r.get('inputs'), dict) else {}
    return md.get('ls_model_name') or ip.get('model') or ip.get('model_name') or inp.get('model') or out.get('model')


def _ls_usage(r):
    out = r.get('outputs') if isinstance(r.get('outputs'), dict) else {}
    um = out.get('usage_metadata')
    if isinstance(um, dict):
        return um.get('input_tokens'), um.get('output_tokens'), 'outputs.usage_metadata.input_tokens', 'outputs.usage_metadata.output_tokens'
    lo = (out.get('llm_output') or {}).get('token_usage') if isinstance(out.get('llm_output'), dict) else None
    if isinstance(lo, dict):
        return lo.get('prompt_tokens'), lo.get('completion_tokens'), 'outputs.llm_output.token_usage.prompt_tokens', 'outputs.llm_output.token_usage.completion_tokens'
    us = out.get('usage')
    if isinstance(us, dict) and ('prompt_tokens' in us or 'input_tokens' in us):
        return us.get('prompt_tokens', us.get('input_tokens')), us.get('completion_tokens', us.get('output_tokens')), 'outputs.usage.prompt_tokens', 'outputs.usage.completion_tokens'
    if r.get('prompt_tokens') is not None or r.get('completion_tokens') is not None:
        return r.get('prompt_tokens'), r.get('completion_tokens'), 'prompt_tokens', 'completion_tokens'
    return None, None, None, None


def _from_runs(obj):
    items, _ = _unwrap(obj)
    byid = {str(r.get('id')): r for r in items}
    children = defaultdict(list)
    for r in items:
        p = r.get('parent_run_id')
        if p is not None and str(p) in byid: children[str(p)].append(str(r.get('id')))
    runs = []
    roots = [r for r in items if r.get('parent_run_id') is None or str(r.get('parent_run_id')) not in byid]
    for root in sorted(roots, key=lambda r: (str(r.get('start_time') or ''), str(r.get('id')))):
        rid = str(root.get('id'))
        members = {i: byid[i] for i in _descendants(rid, children)}
        md = ((root.get('extra') or {}).get('metadata') or {})
        run = _run(run_id=str(root.get('trace_id') or rid), run_name=root.get('name'))
        run['explicit_case'] = _find_key({k: v for k, v in md.items()}, CASE_KEYS)
        run['explicit_candidate'] = _find_key(md, EXPLICIT_CANDIDATE_KEYS)
        run['example_id'] = root.get('reference_example_id')
        run['session_id'] = md.get('thread_id') or md.get('session_id') or md.get('conversation_id')
        for i, r in sorted(members.items(), key=lambda kv: (str(kv[1].get('start_time') or ''), kv[0])):
            rt = str(r.get('run_type') or '').lower()
            if rt == 'tool':
                run['tools'].append({'name': _tool_label(r.get('name')), 'args': r.get('inputs') or {}})
                if r.get('error'): run['tool_errors'].append({'tool': run['tools'][-1]['name'], 'message': str(r['error'])[:200]})
            elif rt == 'llm':
                run['llm_calls'] += 1
                if _ls_model(r): run['models'].append(str(_ls_model(r)))
                if not run['system_prompt']: run['system_prompt'] = _system_from_messages((r.get('inputs') or {}).get('messages') if isinstance(r.get('inputs'), dict) else None)
            if r is not root and r.get('error') and rt != 'tool': run['span_errors'].append(str(r.get('name')))
        # tokens: the server's per-run totals on the root when present, else the LLM runs' usage
        if root.get('prompt_tokens') is not None or root.get('completion_tokens') is not None:
            _add_tokens(run, run['models'][0] if len(set(run['models'])) == 1 else '?', root.get('prompt_tokens'), root.get('completion_tokens'), 'prompt_tokens', 'completion_tokens')
        else:
            for r in members.values():
                if str(r.get('run_type') or '').lower() != 'llm': continue
                tin, tout, fin, fout = _ls_usage(r)
                _add_tokens(run, _ls_model(r), _num(tin), _num(tout), fin, fout)
        if _num(root.get('total_cost')) is not None:
            run['cost_logged'] = _num(root['total_cost']); run['cost_fields'].add('total_cost')
        else:
            cs = [_num(r.get('total_cost')) for r in members.values() if _num(r.get('total_cost')) is not None and str(r.get('run_type') or '').lower() == 'llm']
            if cs: run['cost_logged'] = sum(cs); run['cost_fields'].add('total_cost')
        s, e = _ts(root.get('start_time')), _ts(root.get('end_time'))
        if s is not None and e is not None: run['latency_s'] = max(0., e - s); run['latency_field'] = 'start_time/end_time'
        if root.get('error') or str(root.get('status') or '').lower() == 'error':
            msg = str(root.get('error') or 'status=error')
            run['error'] = {'class': msg.split(':')[0].split('\n')[0][:80] or 'error', 'message': msg[:500]}
        for k, v in (root.get('feedback_stats') or {}).items():
            if isinstance(v, dict):
                val = v.get('avg')
                if val is None and isinstance(v.get('values'), dict) and len(v['values']) == 1: val = next(iter(v['values']))
                if val is not None: run['scores'][str(k)] = val
            elif v is not None: run['scores'][str(k)] = v
        run['input'] = root.get('inputs'); run['output'] = root.get('outputs')
        run['output_logged'] = any(r.get('outputs') not in (None, {}) for r in members.values())
        tags = root.get('tags') or []
        run['meta'] = {k: v for k, v in md.items() if isinstance(v, (str, bool)) and not k.startswith('ls_') and k not in ('revision_id', 'thread_id', 'session_id', 'conversation_id')}
        if tags: run['meta']['tags'] = ','.join(sorted(str(t) for t in tags))
        runs.append(run)
    return runs


# ----------------------------------------------------------------------------- observation-trace exports

def _lf_traces(obj):
    items, _ = _unwrap(obj)
    out = []
    if all(i.get('type') in LF_TYPES and 'traceId' in i for i in items):
        g = defaultdict(list)
        for o in items: g[o['traceId']].append(o)
        return [({'id': t}, obs, 'observations without traces') for t, obs in sorted(g.items())]
    for i in items:
        if 'trace' in i and 'observations' in i: out.append((i['trace'], i['observations'] or [], 'stored'))
        else: out.append((i, i.get('observations') or [], 'api'))
    return out


def _from_observations(obj):
    runs = []
    for t, obs, shape in _lf_traces(obj):
        obs = [o for o in obs if isinstance(o, dict)]
        ids = {o['id']: o for o in obs if 'id' in o}
        children = defaultdict(list)
        for o in obs:
            if o.get('parentObservationId') in ids: children[o['parentObservationId']].append(o['id'])
        run = _run(run_id=str(t.get('id')), run_name=t.get('name'))
        meta = _maybe_json(t.get('metadata'))
        meta = meta if isinstance(meta, dict) else {}
        run['session_id'] = t.get('sessionId')
        run['explicit_case'] = _find_key({k: v for k, v in meta.items() if not isinstance(v, dict)}, CASE_KEYS)
        run['explicit_candidate'] = _find_key({k: v for k, v in meta.items() if not isinstance(v, dict)}, EXPLICIT_CANDIDATE_KEYS)
        name_of = {i: (o.get('name') or '') for i, o in ids.items()}
        for o in sorted(obs, key=lambda o: (str(o.get('startTime') or ''), str(o.get('id')))):
            typ = str(o.get('type') or '').upper()
            om = _maybe_json(o.get('metadata')); om = om if isinstance(om, dict) else {}
            oa = om.get('attributes') if isinstance(om.get('attributes'), dict) else {}
            parent = name_of.get(o.get('parentObservationId'), '')
            if typ == 'TOOL' or oa.get('gen_ai.tool.name') or (typ == 'SPAN' and parent.lower() == 'tools'):
                run['tools'].append({'name': _tool_label(oa.get('gen_ai.tool.name') or o.get('name')), 'args': {}})
                if o.get('level') == 'ERROR': run['tool_errors'].append({'tool': run['tools'][-1]['name'], 'message': str(o.get('statusMessage') or '')[:200]})
            if typ == 'AGENT' or oa.get('gen_ai.agent.name'):
                n = oa.get('gen_ai.agent.name') or o.get('name')
                if n: run['agents'].append(str(n))
            if typ == 'GUARDRAIL':
                out = _maybe_json(o.get('output'))
                trig = (isinstance(out, dict) and (out.get('triggered') is True or out.get('tripwire_triggered') is True)) or o.get('level') == 'ERROR'
                run['guardrails'].append({'name': o.get('name'), 'triggered': bool(trig)})
            if typ == 'GENERATION':
                run['llm_calls'] += 1
                if o.get('model'): run['models'].append(str(o['model']))
                if not run['system_prompt']: run['system_prompt'] = _system_from_messages(o.get('input'))
            if not run['system_prompt'] and oa.get('gen_ai.system_instructions'): run['system_prompt'] = _first200(_maybe_json(oa['gen_ai.system_instructions']))
            if o.get('level') == 'ERROR' and typ not in ('TOOL', 'GUARDRAIL'):
                if o.get('parentObservationId') not in ids or typ == 'AGENT':
                    run['error'] = run['error'] or {'class': 'error', 'message': str(o.get('statusMessage') or 'level=ERROR')[:500]}
                else:
                    run['span_errors'].append(str(o.get('name')))
        def usage(o):
            u = o.get('usageDetails') if isinstance(o.get('usageDetails'), dict) and o.get('usageDetails') else o.get('usage') if isinstance(o.get('usage'), dict) else {}
            return u
        def has_tok(o): u = usage(o); return bool(_num(u.get('input')) or _num(u.get('output')) or _num(u.get('promptTokens')) or _num(u.get('completionTokens')))
        for o in _leafmost(ids, children, has_tok):
            u = usage(o); src = 'usageDetails' if o.get('usageDetails') else 'usage'
            tin = _num(u.get('input', u.get('promptTokens'))); tout = _num(u.get('output', u.get('completionTokens')))
            _add_tokens(run, o.get('model'), tin, tout, f'observations[].{src}.input', f'observations[].{src}.output')
        def costf(o):
            for k in ('calculatedTotalCost', 'totalCost'):
                if _num(o.get(k)): return k
            if isinstance(o.get('costDetails'), dict) and _num(o['costDetails'].get('total')): return 'costDetails.total'
            return None
        costed = _leafmost(ids, children, lambda o: costf(o) is not None)
        if costed:
            vals = []
            for o in costed:
                k = costf(o); vals.append(_num(o['costDetails']['total']) if k == 'costDetails.total' else _num(o[k]))
                run['cost_fields'].add(f'observations[].{k}')
            run['cost_logged'] = sum(vals)
        elif _num(t.get('totalCost')):
            run['cost_logged'] = _num(t['totalCost']); run['cost_fields'].add('totalCost')
        if _num(t.get('latency')) is not None:
            run['latency_s'] = float(_num(t['latency'])); run['latency_field'] = 'trace.latency' if shape == 'stored' else 'latency'
        else:
            st = [_ts(o.get('startTime')) for o in obs if _ts(o.get('startTime')) is not None]
            en = [_ts(o.get('endTime')) for o in obs if _ts(o.get('endTime')) is not None]
            if st and en: run['latency_s'] = max(0., max(en) - min(st)); run['latency_field'] = 'observations startTime/endTime'
        scores = t.get('scores') or []
        for s in scores:
            if not isinstance(s, dict) or s.get('name') is None: continue
            v = s.get('value') if s.get('value') is not None else s.get('stringValue')
            if s.get('dataType') == 'CATEGORICAL' and s.get('stringValue') is not None: v = s['stringValue']
            if v is not None: run['scores'][str(s['name'])] = v
        if scores and not any(isinstance(s, dict) for s in scores): run['meta']['_scores_as_ids'] = True
        tin = _maybe_json(t.get('input'))
        if not run['system_prompt']: run['system_prompt'] = _system_from_messages(tin)
        if not run['system_prompt'] and isinstance(meta.get('attributes'), dict) and meta['attributes'].get('gen_ai.system_instructions'):
            run['system_prompt'] = _first200(_maybe_json(meta['attributes']['gen_ai.system_instructions']))
        run['input'] = tin; run['output'] = _maybe_json(t.get('output'))
        run['output_logged'] = run['output'] is not None or any(o.get('output') is not None for o in obs)
        m = {k: v for k, v in meta.items() if isinstance(v, (str, bool))}
        for k in ('environment', 'release', 'version'):
            if isinstance(t.get(k), str): m[k] = t[k]
        if t.get('tags'): m['tags'] = ','.join(sorted(str(x) for x in t['tags']))
        run['meta'] = m
        runs.append(run)
    return runs


# ----------------------------------------------------------------------------- Agents SDK

def _from_agents_sdk(obj):
    items, _ = _unwrap(obj)
    traces = {i['id']: i for i in items if i.get('object') == 'trace'}
    spans = defaultdict(list)
    for i in items:
        if i.get('object') == 'trace.span': spans[i.get('trace_id')].append(i)
    runs = []
    for tid in list(traces) + sorted(t for t in spans if t not in traces):
        tr = traces.get(tid, {'id': tid})
        ss = sorted(spans.get(tid, []), key=lambda s: (str(s.get('started_at') or ''), str(s.get('id'))))
        meta = tr.get('metadata') if isinstance(tr.get('metadata'), dict) else {}
        run = _run(run_id=str(tid), run_name=tr.get('workflow_name'), session_id=tr.get('group_id'))
        run['explicit_case'] = _find_key(meta, CASE_KEYS); run['explicit_candidate'] = _find_key(meta, EXPLICIT_CANDIDATE_KEYS)
        run['example_id'] = meta.get('example_id')
        ids = {s.get('id') for s in ss}
        for s in ss:
            d = s.get('span_data') or {}
            typ = d.get('type')
            err = s.get('error')
            if typ == 'custom' and isinstance(d.get('data'), dict): typ = d['data'].get('sdk_span_type', 'custom')
            if typ == 'function':
                run['tools'].append({'name': _tool_label(d.get('name')), 'args': _maybe_json(d.get('input')) or {}})
                if err: run['tool_errors'].append({'tool': run['tools'][-1]['name'], 'message': str(err.get('message', err))[:200]})
            elif typ == 'agent':
                if d.get('name'): run['agents'].append(str(d['name']))
                if err and not run['error']:
                    data = err.get('data') if isinstance(err.get('data'), dict) else {}
                    run['error'] = {'class': str(data.get('name') or err.get('message') or 'error')[:80], 'message': str(err.get('message') or '') + (f": {data.get('error')}" if data.get('error') else '')}
            elif typ in ('generation', 'response'):
                run['llm_calls'] += 1
                if d.get('model'): run['models'].append(str(d['model']))
                u = d.get('usage') if isinstance(d.get('usage'), dict) else {}
                tin = u.get('input_tokens', u.get('prompt_tokens')); tout = u.get('output_tokens', u.get('completion_tokens'))
                _add_tokens(run, d.get('model'), _num(tin), _num(tout), 'span_data.usage.input_tokens' if tin is not None else None, 'span_data.usage.output_tokens' if tout is not None else None)
                if d.get('output') is not None: run['output'] = d['output']; run['output_logged'] = True
                if d.get('input') is not None and run['input'] is None: run['input'] = d['input']
                if not run['system_prompt'] and d.get('input') is not None: run['system_prompt'] = _system_from_messages(d['input'])
            elif typ == 'handoff':
                run['handoffs'] += 1
                if err: run['span_errors'].append('handoff: ' + str(err.get('message', ''))[:80])
            elif typ == 'guardrail':
                run['guardrails'].append({'name': d.get('name'), 'triggered': d.get('triggered') is True})
            if err and s.get('parent_id') not in ids and typ not in ('agent',) and not run['error']:
                run['error'] = {'class': str(err.get('message') or 'error')[:80], 'message': str(err.get('message') or '')}
        st = [_ts(s.get('started_at')) for s in ss if _ts(s.get('started_at')) is not None]
        en = [_ts(s.get('ended_at')) for s in ss if _ts(s.get('ended_at')) is not None]
        if st and en: run['latency_s'] = max(0., max(en) - min(st)); run['latency_field'] = 'started_at/ended_at'
        run['meta'] = {k: v for k, v in meta.items() if isinstance(v, (str, bool))}
        runs.append(run)
    return runs


# ----------------------------------------------------------------------------- CSV

def _unflatten_row(row):
    out = {}
    for k, v in row.items():
        if k is None: continue
        k = k.strip()
        if isinstance(v, str):
            s = v.strip()
            v = None if s == '' else True if s.lower() == 'true' else False if s.lower() == 'false' else (_num(s) if _num(s) is not None and re.fullmatch(r'-?\d+(\.\d+)?([eE]-?\d+)?', s) else s)
        parts = k.split('.')
        head = {'inputs': 'input', 'outputs': 'output', 'reference': 'expected', 'references': 'expected'}.get(parts[0], parts[0])
        if len(parts) == 1:
            out[head] = v; continue
        cur = out
        for p in [head] + parts[1:-1]:
            if not isinstance(cur.get(p), dict): cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = v
    return out


def _from_csv(rows):
    recs = []
    for row in rows:
        r = _unflatten_row(row)
        fb = r.pop('feedback', None) if isinstance(r.get('feedback'), dict) else None
        sc = r.pop('scores', None) if isinstance(r.get('scores'), dict) else None
        bare = {k: r[k] for k in BARE_SCORE_COLUMNS if k in r and not isinstance(r[k], dict)}
        scores = {**bare, **(fb or {}), **(sc or {})}
        if not any(r.get(k) is not None for k in CASE_KEYS):
            r['case_id'] = r.get('id') if r.get('id') is not None else r.get('run_id')
        lat = next(((k, _num(r[k])) for k in ('execution_time', 'latency', 'latency_s', 'duration', 'duration_s') if _num(r.get(k)) is not None), None)
        err = r.get('error')
        if err in (None, ''): r.pop('error', None)
        aver = {'format': 'csv', 'scores': scores, 'latency_s': lat[1] if lat else None, 'latency_field': lat[0] if lat else None,
                'output_logged': any(r.get(k) is not None for k in ('output', 'prediction', 'response', 'answer', 'completion', 'result'))}
        r['_aver'] = aver
        recs.append(r)
    if recs and not any(any(r.get(k) is not None for k in EXPLICIT_CANDIDATE_KEYS + ('model_id', 'model', 'agent')) for r in recs):
        for r in recs: r['candidate_id'] = 'logged'
    return recs


# ----------------------------------------------------------------------------- cases, options, outcomes, cost

def _model_or_agent(run):
    if run['models']: return Counter(run['models']).most_common(1)[0][0]
    named = [a for a in run['agents'] if not UUIDISH.match(a)]
    return named[0] if named else None


def _assign_cases(runs):
    """Case and option per run (docs/141 §2). Returns the rule that was applied, in words."""
    if runs and all(r['explicit_candidate'] is not None for r in runs):
        for r in runs:
            r['case_id'] = str(r['explicit_case'] or r['session_id'] or r['run_id']); r['candidate_id'] = str(r['explicit_candidate'])
        return 'Each run names its configuration (an explicit candidate attribute); the case is its case-id attribute, else its session, else its trace.'
    moa = [_model_or_agent(r) for r in runs]
    for key, label in (('explicit_case', 'case-id attribute'), ('example_id', 'example id'), ('session_id', 'session'), ('run_name', 'trace name')):
        vals = [r[key] for r in runs]
        if any(v is None for v in vals): continue
        groups = defaultdict(list)
        for r, m, v in zip(runs, moa, vals): groups[str(v)].append(m)
        if any(len(set(ms)) > 1 for ms in groups.values()) and all(None not in ms and len(ms) == len(set(ms)) for ms in groups.values()):
            for r, m, v in zip(runs, moa, vals): r['case_id'] = str(v); r['candidate_id'] = str(m)
            return f'Runs share a {label} across different models or agents, so each {label} is a case and each model or agent is an option.'
    single = moa[0] if moa and len(set(moa)) == 1 and moa[0] else 'logged'
    for r in runs:
        r['case_id'] = str(r['explicit_case'] if r['explicit_case'] is not None and len({x['explicit_case'] for x in runs}) == len(runs) else r['run_id'])
        r['candidate_id'] = single
    return (f'No case key (example id, session, trace name) is shared across different models, so no two runs are the same case run by '
            f'two options: one option ({single}).' + ('' if single != 'logged' else ' The models seen: ' + (', '.join(sorted({m for m in moa if m})) or 'none') + '.'))


def score_direction(name):
    n = str(name).lower()
    if any(w in n for w in SCORE_HARM): return 'harm'
    if any(w in n for w in SCORE_LOWER): return 'lower'
    if any(w in n for w in SCORE_HIGHER): return 'higher'
    return None


def _score01(v):
    if isinstance(v, bool): return 1. if v else 0.
    n = _num(v)
    if n is not None: return float(n)
    if isinstance(v, str):
        s = v.strip().lower().replace(' ', '_')
        if s in PASS_WORDS: return 1.
        if s in FAIL_WORDS: return 0.
    return None


def _kind_for(name, direction):
    key = re.sub(r'[^a-z0-9_]', '_', str(name).lower()).strip('_')[:56] or 'score'
    if direction == 'harm':
        return key, f'Scored as {name} (score at least 0.5)', 'incident', 'harmful_output'
    if direction == 'lower':
        return key, f'Scored as {name} (score at least 0.5)', 'mistake', 'wrong_output'
    return f'low_{key}', f'Scored low on {name} (below 0.5)', 'mistake', 'wrong_output'


def select_outcomes(scores_by_run):
    """Which score fields label runs (docs/141 §3). A field is used when it is on every run, its values are numbers in
    [0, 1] or pass/fail words, and its direction is known from its name. Returns (fields, rule text, skipped)."""
    n = len(scores_by_run)
    names = sorted({k for s in scores_by_run for k in s})
    used, skipped = [], []
    for name in names:
        on = sum(name in s and s[name] is not None for s in scores_by_run)
        vals = [_score01(s[name]) for s in scores_by_run if name in s and s[name] is not None]
        d = score_direction(name)
        if on < n: skipped.append(f'{name} (on {on} of {n} runs)'); continue
        if any(v is None or v < 0 or v > 1 for v in vals): skipped.append(f'{name} (scale unknown: not all values in [0, 1] or pass/fail words)'); continue
        if d is None: skipped.append(f'{name} (direction unknown from its name)'); continue
        used.append((name, d))
    rule = ('A score or feedback field labels runs when it is on every run, its values lie in [0, 1] (or are pass/fail words) and its name '
            'says which way is good. Higher-is-better fields fail below 0.5; lower-is-better fields (hallucination, ...) fail at 0.5 or more; '
            'harm fields (toxicity, ...) at 0.5 or more are incidents.')
    return used, rule, skipped


def parse_prices(spec):
    """'model-a=0.00015,0.0006;model-b=$0.003,$0.012' (or a list of such) -> {model: (usd per 1k input, per 1k output)}."""
    if not spec: return {}
    parts = spec if isinstance(spec, (list, tuple)) else [spec]
    out = {}
    for p in parts:
        if isinstance(p, dict):
            out.update({str(k): (float(v[0]), float(v[1])) for k, v in p.items()}); continue
        for item in re.split(r';', str(p)):
            item = item.strip()
            if not item: continue
            m = re.fullmatch(r'([^=]+)=\s*\$?([0-9.eE+-]+)\s*,\s*\$?([0-9.eE+-]+)', item)
            need(m is not None, 'invalid_arguments', 'prices', f'Prices are model=$/1k_input,$/1k_output (for example model-a=0.00015,0.0006)')
            out[m.group(1).strip()] = (float(m.group(2)), float(m.group(3)))
    return out


def _price_for(model, prices):
    if model in prices: return prices[model]
    # a dated or versioned name of a priced model (model-a-2024-07-18, model-b-0125); never another model that
    # merely shares a prefix (model-a-mini is not model-a)
    best = max((k for k in prices if model and re.match(re.escape(k) + r'-\d', str(model))), key=len, default=None)
    return prices[best] if best else None


def _apply_cost(run, prices):
    if run['cost_logged'] is not None:
        return run['cost_logged'], 'logged'
    if not run['tokens']: return None, 'none'
    if not prices: return None, 'tokens_unpriced'
    total = 0.
    for model, (tin, tout) in run['tokens'].items():
        p = _price_for(model if model != '?' else (run['models'][0] if run['models'] else None), prices)
        if p is None: return None, 'tokens_unpriced'
        total += tin / 1000 * p[0] + tout / 1000 * p[1]
    return round(total, 10), 'priced'


def _to_record(run, fmt, used, prices):
    cost, basis = _apply_cost(run, prices)
    fails = []
    for name, d in used:
        v = _score01(run['scores'][name])
        failed = v < .5 if d == 'higher' else v >= .5
        if failed: fails.append({'field': name, 'value': run['scores'][name], **dict(zip(('kind', 'label', 'severity', 'standard'), _kind_for(name, d)))})
    aver = {'format': fmt, 'tools': [t['name'] for t in run['tools']], 'tool_errors': run['tool_errors'], 'handoffs': run['handoffs'],
            'guardrails': run['guardrails'], 'agents': sorted(set(run['agents'])), 'models': sorted(set(run['models'])), 'llm_calls': run['llm_calls'],
            'tokens': {m: list(t) for m, t in run['tokens'].items()}, 'cost_basis': basis, 'cost_fields': sorted(run['cost_fields'] | run['token_fields']),
            'latency_s': run['latency_s'], 'latency_field': run['latency_field'], 'span_errors': run['span_errors'], 'scores': run['scores'],
            'system_prompt': run['system_prompt'], 'run_name': run['run_name'], 'output_logged': run['output_logged'], 'run_id': run['run_id']}
    if used:
        aver['recorded_outcome'] = {'field': 'scores: ' + ', '.join(n for n, _ in used), 'passed': not fails}
        aver['score_fails'] = fails
    rec = {'case_id': run['case_id'], 'candidate_id': run['candidate_id'], 'tool_calls': run['tools'], 'cost_usd': cost,
           'meta': {**run['meta'], **({'run_name': run['run_name']} if run['run_name'] else {})}, '_aver': aver}
    if run['input'] is not None: rec['input'] = run['input']
    if run['output'] is not None: rec['output'] = run['output']
    if run['expected'] is not None: rec['expected'] = run['expected']
    if run['error']: rec['error'] = run['error']
    return rec


ADAPTERS = {'otel': _from_otel, 'runs': _from_runs, 'observations': _from_observations, 'agents_sdk': _from_agents_sdk}


def adapt(obj, fmt=None, prices=None, hint=None):
    """An export -> {'format', 'reason', 'records', 'candidates_rule', 'outcome_rule', 'outcome_fields', 'skipped_scores'}.
    `fmt` overrides detection (--format); `prices` is a price table (parse_prices) for runs that log tokens but no cost."""
    if isinstance(obj, str):
        obj, hint = load_text(obj)
    if fmt:
        need(fmt in FORMATS, 'unknown_log_format', 'format', f'Unknown format "{fmt}". Known: {", ".join(FORMATS)}.')
        reason = 'chosen with --format'
    else:
        fmt, reason = detect(obj, hint)
    prices = parse_prices(prices) if not isinstance(prices, dict) else prices
    out = {'format': fmt, 'reason': reason, 'candidates_rule': None, 'outcome_rule': None, 'outcome_fields': [], 'skipped_scores': []}
    if fmt == 'generic':
        out['records'] = obj if isinstance(obj, list) else [obj]
        return out
    if fmt == 'csv':
        rows = obj if isinstance(obj, list) else [obj]
        need(_dicts(rows), 'unknown_log_format', 'logs', 'A CSV export needs a header row and at least one data row.')
        recs = _from_csv(rows)
        used, rule, skipped = select_outcomes([r['_aver']['scores'] for r in recs])
        for r in recs:
            fails = []
            for name, d in used:
                v = _score01(r['_aver']['scores'][name]); failed = v < .5 if d == 'higher' else v >= .5
                if failed: fails.append({'field': name, 'value': r['_aver']['scores'][name], **dict(zip(('kind', 'label', 'severity', 'standard'), _kind_for(name, d)))})
            if used:
                r['_aver']['recorded_outcome'] = {'field': 'scores: ' + ', '.join(n for n, _ in used), 'passed': not fails}
                r['_aver']['score_fails'] = fails
            r['_aver']['cost_fields'] = [k for k in ('cost_usd', 'total_cost_usd', 'cost', 'total_cost') if r.get(k) is not None]
            r['_aver']['cost_basis'] = 'logged' if r['_aver']['cost_fields'] else 'none'
        out.update(records=recs, outcome_rule=rule, outcome_fields=[n for n, _ in used], skipped_scores=skipped,
                   candidates_rule='No configuration column: one option (logged).' if all(r.get('candidate_id') == 'logged' for r in recs) else 'The configuration column names the option.')
        return out
    try:
        runs = ADAPTERS[fmt](obj)
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        if isinstance(exc, DecisionError): raise
        raise DecisionError('unknown_log_format', 'logs', f'The export does not have the structure of a {fmt} export ({type(exc).__name__}). Check --format.') from exc
    need(bool(runs), 'unknown_log_format', 'logs', f'No runs found in the structure of a {fmt} export. Check --format.')
    need(all(r['run_id'] and not re.search(r'(^|:)None($|:)', str(r['run_id'])) for r in runs), 'unknown_log_format', 'logs',
         f'The export does not have the structure of a {fmt} export: its runs have no trace or run ids. Check --format.')
    out['candidates_rule'] = _assign_cases(runs)
    used, rule, skipped = select_outcomes([r['scores'] for r in runs])
    out.update(records=[_to_record(r, fmt, used, prices) for r in runs], outcome_rule=rule, outcome_fields=[n for n, _ in used], skipped_scores=skipped)
    return out
