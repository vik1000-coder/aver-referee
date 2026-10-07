"""Offline decision command group. No provider calls, uploads or approval powers.

v1 (one frozen look, docs/126) and v2 (planned looks, docs/127) share the files and the group.
A plan made with --looks is a v2 plan; freeze, look, explain and bundle dispatch on the artifact's schema.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from aver_policy.decision import (
    ALPHA, METHOD, DecisionError, demo, digest, evaluate, explore, freeze,
    make_plan, need, validate_rows, validate_spec, verify_result,
)
from aver_policy import decision_v2 as v2
from aver_policy import explain as ex_mod
from aver_policy import log_reader as lr
from aver_policy import log_formats as lf_mod
from aver_policy import label_assistance as la
from aver_policy.decision_memo import decision_memo
from aver_cli import workflow as wf

STATE_WORDS = {'pass': 'PASSES', 'block': 'FAILS', 'continue': 'KEEP TESTING', 'exhausted': 'INCONCLUSIVE', 'incomplete': 'ROUND INCOMPLETE'}
COMMANDS = ['next', 'demo', 'init', 'label-request', 'read-logs', 'accept', 'audit', 'validate', 'explore', 'plan', 'oc', 'freeze', 'cases', 'evaluate', 'look', 'explain', 'memo', 'hosted-request', 'bundle', 'capabilities', 'schema']
EXITS = {'pass': 0, 'refuse': 1, 'review': 1, 'block': 2}


def capabilities():
    return {'method_version': METHOD, 'methods': [METHOD, v2.METHOD, v2.METHOD_V3], 'commands': COMMANDS, 'offline': True,
            'supported': ['independent_binary_cases', 'fixed_traffic_weights', 'one_frozen_look', 'planned_looks_1_to_5', 'clause_aware_allocation', 'exact_bonferroni', 'observed_cost_ranking', 'operating_characteristics', 'net_value_of_checking',
                          'mistake_kinds_with_prices_and_tolerances', 'tolerances_in_occurrences_per_month', 'audited_ai_labels', 'two_phase_audit', 'gamma_bound_for_weighted_clauses', 'heuristic_log_reader',
                          'otlp_trace_logs', 'reliability_explanation', 'agent_next_step', 'accept_from_proposal', 'round_case_lists', 'round_only_or_cumulative_outcomes', 'model_labeling_handoff',
                          'paired_savings_bound', 'separate_limits_and_savings_forecasts', 'unsigned_economic_memo', 'approved_hosted_plan_request_packets'],
            'unsupported': ['dependent_repeats', 'signed_records', 'hosted_upload', 'automatic_collection', 'anytime_valid_stopping', 'model_backed_log_reader_offline'],
            'alpha_selection': ALPHA, 'alpha_violation': ALPHA,
            'release_eligibility': 'not_evaluated',
            'schemas': ['spec', 'evidence', 'plan', 'frozen-plan', 'result', 'bundle'],
            'exit_codes': {'0': 'successful operation or requirements met', '1': 'refusal, continue to the next look, or human review', '2': 'demonstrated violation', '3': 'invalid input or execution error'},
            'next_exit_codes': {'0': 'there is a next command to run', '1': 'a person must decide before continuing', '2': 'the decision is finished (pass, fail or inconclusive)', '3': 'the workspace is inconsistent'},
            'workspace_files': wf.FILES}


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise DecisionError('invalid_arguments', 'arguments', message)


def parser():
    p = Parser(prog='aver decision', description='Plan and evaluate the cost and reliability of a pipeline using local evidence. Start with demo. No record is issued.')
    sub = p.add_subparsers(dest='operation', required=True, parser_class=Parser)
    for name in COMMANDS:
        extra = {}
        if name == 'look':
            extra = dict(formatter_class=argparse.RawDescriptionHelpFormatter, epilog=LOOK_EPILOG, description='Look at the evidence after a round (alias: round). Exit 0 passes, 1 keep testing or inconclusive, 2 fails, 3 error.')
        if name == 'next':
            extra = dict(formatter_class=argparse.RawDescriptionHelpFormatter, epilog=NEXT_EPILOG, description='For agents: read which files exist in a workspace and print exactly the next command, why, what a person must decide, and what must not be done.')
        q = sub.add_parser(name, aliases=['round'] if name == 'look' else [], **extra)
        q.add_argument('--json', action='store_true', help='One structured JSON response on stdout')
        q.add_argument('--out', help='Write the artifact atomically; never replace different content')
        if name == 'next':
            q.add_argument('workspace', nargs='?', default='.', help='The workspace directory (default: the current directory)')
        if name == 'accept':
            q.add_argument('--proposal', required=True, help='proposal.json from read-logs')
            q.add_argument('--prices', help='Dollars per occurrence, as kind=price pairs: wrong_field=2,missed_page=2')
            q.add_argument('--tolerances', help='At most N a month, as kind=N pairs: unwanted_write=450')
            q.add_argument('--failure-limit', help='The most failed runs accepted overall, as a share: 8%% or 0.08')
            q.add_argument('--incident-limit', help='The most incidents accepted in each kind of case (default: the failure limit)')
            q.add_argument('--monthly-uses', help='Runs a month (tolerances are per month)')
            q.add_argument('--price-per-case', help='Dollars to run and check one test case (prices the test)')
            q.add_argument('--candidates', help='Two to four options to compare, comma-separated (default: every option in the logs)')
            q.add_argument('--baseline', help='The option in use now (default: the first option, alphabetically)')
            q.add_argument('--budget-runs', help='Total runs you will pay for, the logged ones included (default: four times the logged runs)')
            q.add_argument('--horizon-months', type=int, default=12)
            q.add_argument('--app', help='A name for the pipeline'); q.add_argument('--checkpoint', help='What counts as one run')
            q.add_argument('--purpose', help="What the agent is for, in your words, when the reader's sentence is not right (default: the proposal's)")
            q.add_argument('--pilot-out', help='Also write the labelled logged runs (pilot rows, JSONL) here: pilot.jsonl')
        if name == 'cases':
            q.add_argument('--plan', required=True, help='frozen.json')
            q.add_argument('--round', type=int, required=True, help='The round to list (1 to the planned rounds)')
        if name == 'demo':
            q.add_argument('--scenario', choices=['document-processing', 'support-routing'], default='document-processing')
            q.add_argument('--outcome', choices=['pass', 'refuse', 'block', 'continue', 'exhausted'], default='pass')
            q.add_argument('--rounds', '--looks', dest='looks', type=int, help='Rounds of testing (1-5): a v2 demo that tests round by round')
            q.add_argument('--out-dir', help='Write editable example spec, pilot, plan and confirmation files')
        if name == 'init':
            q.add_argument('--rounds', '--looks', dest='looks', type=int, help='Write the v2 template (cost per checked case, horizon, grader error)')
            q.add_argument('--deployment', choices=['normal', 'agentic'], help='Write the v3 template with the standard mistake kinds and this deployment profile')
        if name == 'label-request':
            q.add_argument('--logs', required=True)
            q.add_argument('--model', required=True, help='Exact model chosen for your labeling agent; identity is customer-reported, not attested')
            q.add_argument('--include-content', action='store_true', help='Include input, output, expected, tools and metadata after stripping credential fields. Review and redact personal content before sending the messages to a model.')
            q.add_argument('--format', choices=list(lf_mod.FORMATS))
            q.add_argument('--prices', action='append', help='The same model price table used by read-logs, for token-only exports; never sent in the labeling messages')
        if name == 'read-logs':
            q.add_argument('--logs', required=True, help='Your export as it is: OpenTelemetry (OTLP/JSON or JSONL of spans), run or trace exports from LLM observability tools, Agents-SDK trace exports (trace / trace.span objects), JSON/JSONL result records, or CSV. The format is detected from the structure (docs/141)')
            q.add_argument('--format', choices=list(lf_mod.FORMATS), help='Skip detection and read the export as this format')
            q.add_argument('--prices', action='append', help='Model prices for runs that log tokens but no cost: model=$/1k_input,$/1k_output (repeat, or separate models with ;). Without it such runs have unknown cost')
            q.add_argument('--reader', choices=['heuristic', 'model-response', 'mock-llm'], default='heuristic', help='heuristic (offline), model-response (your labeling agent\'s saved response; still needs a human audit), or mock-llm (TEST DOUBLE)')
            q.add_argument('--label-request', help='The request bound to the logs, from label-request')
            q.add_argument('--label-response', help='aver-label-response/v1 envelope from your labeling agent')
            q.add_argument('--allow-test-double', action='store_true', help='Permit --reader mock-llm (tests and demos only)')
            q.add_argument('--truth', help='Hidden truth labels (truth.jsonl) for the mock-llm test double only')
            q.add_argument('--spec', help='An accepted spec for these logs: sizes the audit from its failure limit')
            q.add_argument('--audit-fine', '--audit-n', dest='audit_fine', type=int, help='AI-fine runs to sample for the audit (default: sized from the spec, else min(fine runs, 400))')
            q.add_argument('--audit-flagged', type=int, help='AI-flagged runs to sample (default: min(flagged runs, 100))')
        if name == 'audit':
            q.add_argument('--spec', help='The v3 spec: sizes a new sample, or receives grader_audit when --labels is given')
            q.add_argument('--proposal', help='A proposal from read-logs: draw a fresh two-phase sample from its rows')
            q.add_argument('--fine', type=int, help='AI-fine runs to sample (default: the plan\'s fine_rows_needed, else min(fine runs, 400))')
            q.add_argument('--flagged', type=int, help='AI-flagged runs to sample (default: min(flagged runs, 100))')
            q.add_argument('--seed', type=int, default=7)
            q.add_argument('--sample', help='The audit sample (a proposal, an audit-sample file, or a list) that the labels answer')
            q.add_argument('--labels', help="JSON array or JSONL of the person's labels: {candidate_id, case_id, completed, incident, mistake?}")
        if name in ['validate', 'explore', 'plan']:
            q.add_argument('--spec', required=True)
        if name == 'validate': q.add_argument('--evidence')
        if name == 'explore': q.add_argument('--pilot', required=True)
        if name == 'plan':
            q.add_argument('--exploration', required=True)
            q.add_argument('--finalists', type=int, required=True)
            q.add_argument('--rounds', '--looks', dest='looks', type=int, help='Rounds of testing (1-5). With --rounds the plan is a v2 plan that tests round by round.')
            q.add_argument('--cap', type=int, help='Confirmation cases per finalist (v2); default: the remaining budget')
            q.add_argument('--allocation', choices=['clause', 'equal'], default='clause')
            q.add_argument('--incident-rate', type=float, default=.01, help='Planning incident rate as a fraction (v2)')
            q.add_argument('--spending', help='Comma-separated shares of the error budget per look, summing to one (v2)')
            q.add_argument('--paired', action='store_true', help="Also run every confirmation case on the option in use now (the spec's baseline) and bound the difference: needs requirements.paired_limit_per_month in the spec, or --paired-limit")
            q.add_argument('--paired-limit', type=float, help='Paired test: how many more failures a month than the option in use now you accept (0 or more); sets requirements.paired_limit_per_month and implies --paired')
        if name == 'oc':
            q.add_argument('--plan', required=True, help='A v2 plan or frozen plan')
            q.add_argument('--rate', type=float, help='A single true failure rate to evaluate instead of the three design rates')
            q.add_argument('--exchange', action='store_true', help='Also compute the best cap and the move-the-bar table')
        if name in ['freeze', 'evaluate', 'look']: q.add_argument('--plan', required=True)
        if name in ['evaluate', 'look']:
            q.add_argument('--evidence', required=True)
            q.add_argument('--previous', help='The previous round\'s result (result-N.json). With it, --evidence may hold the new round\'s rows only; earlier rows are taken from it, unchanged')
        if name == 'explain':
            q.add_argument('--spec', help='The decision spec (default: the one bound into the plan)')
            q.add_argument('--plan', help='A plan or frozen plan (default: the one bound into the result)')
            q.add_argument('--result', help='A decision result from look; verified before it is explained')
            q.add_argument('--pilot', help='Pilot rows (default: the plan\'s)')
            q.add_argument('--what-if-price', type=float, help='Price used in "priced instead of tolerated" (default: your highest price)')
        if name == 'bundle': q.add_argument('--result', required=True)
        if name == 'memo':
            q.add_argument('--result', required=True)
            q.add_argument('--monthly-fee', type=float, default=0., help='Monthly subscription cost allocated to this decision; zero excludes fees')
        if name == 'hosted-request':
            q.add_argument('--operation', choices=['submit_evidence', 'run_check', 'read_verdict'], required=True)
            q.add_argument('--app', required=True); q.add_argument('--plan-id', required=True)
            q.add_argument('--round', type=int); q.add_argument('--evidence')
        if name == 'schema': q.add_argument('--name', choices=['spec', 'evidence', 'plan', 'frozen-plan', 'result', 'bundle'], required=True)
    return p


LOOK_EPILOG = """\
Outcomes (--evidence): JSONL (one row per line) or a JSON array of aver-outcomes/v2 rows, one per case in
round-N-cases.jsonl and option. Either shape is accepted:
  * the new round's rows only, with --previous result-(N-1).json (earlier rows are taken from it), or
  * every row so far (rounds 1 to N), earlier rows unchanged.
Round 1 needs no --previous. A file that repeats only some earlier rows, or holds a later round, is refused.

One row:
  {"schema": "aver-outcomes/v2", "case_id": "confirmation-scanned-0", "case_sha256": "<64 hex: sha256 of the case content>",
   "candidate_id": "guarded", "configuration_sha256": "<from round-N-cases.jsonl>", "stratum": "scanned",
   "phase": "confirmation", "origin": "ai-labeled", "completed": false, "incident": false,
   "cost_usd": 0.0043, "mistake": "wrong_field"}
Rules: completed false or incident true -> mistake names a kind from the spec (an incident kind needs incident true);
completed true and incident false -> mistake null; completed null (unknown) -> mistake null. origin must match the
pilot's (ai-labeled when the logs were read by read-logs). A case_id is a NEW case of that stratum, never a logged one;
the same case_id is the same content for every option.

Example: aver decision look --plan frozen.json --evidence round-2.jsonl --previous result-1.json --out result-2.json
"""

NEXT_EPILOG = """\
Exit codes: 0 there is a next command (run it, then call next again); 1 a person must decide first (prices and
tolerances, the audit labels, an infeasible plan); 2 the decision is finished (pass, fail or inconclusive) and
explained; 3 the workspace is inconsistent (a file changed or is out of order).

Files: logs.jsonl, proposal.json, spec.json, pilot.jsonl, audit-sample.json, audit-labels.jsonl, audited-spec.json,
exploration.json, plan.json, frozen.json, round-N-cases.jsonl, round-N.jsonl, result-N.json, explanation.json,
bundle.json, and optionally harness-command.txt (your harness with {cases}, {out}, {round}). See docs/136.

Agent loop:  while next exits 0: run the printed command.  On 1 ask the person; on 2 report; on 3 stop.
"""


def read(path, rows=False):
    try:
        text = Path(path).read_text(encoding='utf-8')
        need(len(text) <= 32_000_000, 'file_too_large', path, 'Files must be at most 32 MB.')
        if rows and text.lstrip() and not text.lstrip().startswith('['):
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        return json.loads(text)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DecisionError('unreadable_input', str(path), 'Could not read the requested JSON/JSONL input.') from exc


def write_jsonl(path, rows):
    """JSONL with the same rules as write: atomic, and never replace different content."""
    path = Path(path)
    content = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n' for r in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        need(path.read_text(encoding='utf-8') == content, 'output_exists', str(path), 'Different output already exists. Choose a new path.')
        return str(path)
    fd, tmp = tempfile.mkstemp(prefix='.aver-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(content); stream.flush(); os.fsync(stream.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            need(path.read_text(encoding='utf-8') == content, 'output_exists', str(path), 'A different output was written concurrently.')
    finally:
        os.unlink(tmp)
    return str(path)


def spec_text(spec):
    """Human output of accept: what the person decided, kind by kind."""
    w = ex_mod.wrap
    V = spec['monthly_uses']; L = spec['requirements']['failure_limit']
    out = [w(f"Accepted: {spec['population_statement']}"), w(f"Options: {', '.join(c['id'] for c in spec['candidates'])} (in use now: {spec['baseline']}). "
           f"Volume: {V:,} runs a month. Failure limit: {L:.1%} (about {L * V:,.0f} failed runs a month). Checking one case: ${spec['label_cost_usd']:g}."), 'Kinds of mistake:']
    for k, v in sorted(spec['taxonomy']['kinds'].items()):
        terms = [f"${v['price_usd']:g} each" if v['price_usd'] is not None else None, f"at most {v['tolerance_per_month']:g} a month" if v['tolerance_per_month'] is not None else None]
        out.append(w(f"{k} [{v['severity']}]: {v['label']}: " + (', '.join(t for t in terms if t) or 'no price, no tolerance (counts as a failure only)'), '  - '))
    return '\n'.join(out) + '\n'


def write(path, value):
    path = Path(path)
    content = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        need(path.read_text(encoding='utf-8') == content, 'output_exists', str(path), 'Different output already exists. Choose a new path.')
        return str(path)
    fd, tmp = tempfile.mkstemp(prefix='.aver-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            need(path.read_text(encoding='utf-8') == content, 'output_exists', str(path), 'A different output was written concurrently.')
    finally:
        os.unlink(tmp)
    return str(path)


def is_v2(obj):
    return isinstance(obj, dict) and (str(obj.get('schema', '')).endswith('/v2') or str(obj.get('schema', '')).endswith('/v3'))


def clause_words(binding, report):
    """'unwanted_write tolerance' -> 'the tolerance for Wrote to the ledger… (450 a month)'; 'ambiguous incident limit' -> as is."""
    if not binding: return ''
    if binding == 'paired limit':
        for c in report.get('candidates', []):
            p = c.get('paired')
            if p: return f"the paired limit (no more than {p['limit_per_month']:g} more failures a month than the option in use now)"
    if binding.endswith(' tolerance'):
        kind = binding[:-len(' tolerance')]
        for c in report.get('candidates', []):
            row = (c.get('by_mistake') or {}).get(kind)
            if row: return f"the tolerance for {kind} ({row['tolerance_per_month']:g} a month)"
    return binding


def schema_of(obj):
    return obj.get('schema') if isinstance(obj, dict) else None


def proposal_text(p):
    """Human output of read-logs: purpose, deployment, strata, the specialised kinds with evidence counts, the audit."""
    from aver_policy.explain import wrap as _wrap   # the CLI imports only aver_policy, aver_cass and the stdlib allow-list (docs/104 §1)
    w = lambda t, ind='': _wrap(t, ind)
    out = [w('What it is for: ' + p['purpose']), w(f"Deployment: {p['deployment']['mode']} ({p['deployment']['why']})."), w(p.get('summary', ''))]
    out.append(f"Kinds of case ({p.get('strata_field', '')}):")
    for k, v in sorted(p['strata'].items(), key=lambda kv: -kv[1]['weight']): out.append(f'  - {k}: {v["weight"]:.0%} of cases')
    out.append(w(p['rationales']['strata'], '    '))
    out.append('Kinds of mistake (proposed; edit before you accept):')
    for k, v in p['taxonomy']['kinds'].items():
        e = p['taxonomy_evidence'].get(k, {})
        by = ', '.join(f'{c} {n}' for c, n in sorted(e.get('by_candidate', {}).items()))
        out.append(w(f"{k} [{v['severity']}, from {v['standard']}]: {v['label']}. Seen {e.get('count', 0)} times ({by or 'none'}); "
                     f"starts as a {p['defaults'].get(k, 'price')}; e.g. {', '.join(e.get('examples', [])[:3]) or '-'}", '  - '))
    a = p['audit']
    lf = p.get('labels_from') or {}
    if lf.get('origin') == 'customer':
        out.append(w(f"Labels: {lf['why']}"))
    else:
        out.append(w(f"Audit before any pass: {a['why']}"))
    out.append(f"Reader: {p['reader']['name']} v{p['reader']['version']}. {len(p['rows'])} labelled runs.")
    return '\n'.join(out) + '\n'


def artifact_text(d):
    """Human output of explore, plan (v2/v3), freeze and bundle: a short summary instead of the whole artifact (the
    artifact itself is in --out or --json). Added after the first real export (docs/137 §3): explore and plan printed
    8,000-10,000 lines of JSON, every pilot row included."""
    s = schema_of(d); w = ex_mod.wrap; out = []
    if s == 'aver-exploration/v1':
        V = (d.get('spec') or {}).get('monthly_uses')
        kappa = next((round((r['score_usd'] - r['execution_cost_usd']) / r['failure_rate'], 2) for r in d['ranking']
                      if r.get('score_usd') is not None and r.get('execution_cost_usd') is not None and r['failure_rate']), None)
        out.append(w(f"Options on the {len(d.get('pilot') or [])} logged runs, ranked by score: running cost plus failures at one price per failure pooled over "
                     f"the options" + (f" (${kappa:,.2f})" if kappa is not None else '') + ". explain gives each option's own all-in cost, at its own mix of mistakes."))
        for r in d['ranking']:
            run_cost = 'unknown' if r.get('execution_cost_usd') is None else f"${r['execution_cost_usd']:.4f}"
            score = 'unknown' if r.get('score_usd') is None else (f"${r['score_usd'] * V:,.0f} a month" if V else f"${r['score_usd']:.4f} a use")
            out.append(w(f"{r['candidate_id']}: score {score}; {run_cost} a run to execute; {r['failure_rate']:.1%} of logged runs failed"
                         + ('' if r.get('observed_eligible') else ' (above the failure limit on the logs)'), '  - '))
    elif s in ('aver-test-plan/v2', 'aver-test-plan/v3'):
        oc = d['operating_characteristics']; sc = oc['scenarios']; al = d['allocation']
        per = ', '.join(f"{k} {v}" for k, v in sorted(al['final'].items()))
        out.append(w(f"Test {', '.join(d['selected'])} on up to {al['total']:,} new cases each, in {d['looks']} round(s) ({per})."
                     + f" Feasible: {'yes' if d['feasible'] else 'no (cannot pass even with no mistakes)'}."))
        paired = d.get('paired') is True
        if paired:
            floor = d.get('saving_floor')
            out.append(w(f"Paired: each case also runs on the option in use now, {d['confirmation_executions']:,} runs in all, so the result can bound the difference "
                         f"between the two (no more than {d['economics_inputs']['paired_limit_per_month']:g} more failures a month)"
                         + (f"; with no case where the two differ, that needs at least {floor:,} cases." if floor is not None else '.')))
        if oc.get('per_case_usd') is not None:
            # a paired plan prices each case on both options at once (docs/140 §5.3)
            priced = al['total'] * len(d['selected']) if paired else d['confirmation_executions']
            audit_usd = oc.get('audit_cost_usd') or 0
            out.append(w(f"Price of the test: about ${oc['per_case_usd']:.2f} a case{' (both options)' if paired else ''}; at most ${oc['per_case_usd'] * priced + audit_usd:,.0f} if every round runs "
                         f"(expected about ${sc['central']['expected_spend_usd']:,.0f} at the central rate"
                         + (f", including about ${audit_usd:,.0f} for a person to check the audit sample" if audit_usd else '') + ")."))
        out.append(w('Chance of a clear pass: ' + ', '.join(f"{sc[k]['pass']:.0%} {k} (failure rate {oc['design_rates'][k]:.1%})" for k in ('optimistic', 'central', 'pessimistic')) + '.'))
        central = sc['central']; shown = central.get('p_saving_shown')
        if 'p_limits_met' in central:
            out.append(w(f"Forecast if the sample holds: {central['p_limits_met']:.0%} chance of meeting your limits; "
                         + (f"{shown:.0%} chance of showing positive savings" if shown is not None else 'savings chance unavailable until costs and prices are known')
                         + f"; {central['p_clear_answer']:.0%} chance of a pass or fail. "
                         + ('Both options run on the same cases.' if paired else "Savings are compared with today's logged rate.")
                         + ' Planning estimates, not promised outcomes.'))
            out.append(w('Value counts savings only after a pass with a positive saving shown, assuming you then switch. Subscription fees and unlogged setup work are extra.'))
    elif s in ('aver-frozen-plan/v2', 'aver-frozen-plan/v3'):
        p = d['plan']
        out.append(w(f"Frozen: {', '.join(p['selected'])} on up to {p['allocation']['total']:,} new cases each in {p['looks']} round(s). Plan digest {d['plan_digest']}. Nothing upstream may change now."))
    elif s == 'aver-decision-bundle/v1':
        rep = d['result']['report']
        out.append(w(f"Bundle of the verified result after round {rep.get('look')} of {rep.get('looks')}: {rep['verdict']['state']} ({rep['verdict']['reason_code']}), "
                     f"choice {rep.get('choice') or 'none'}; {len(d['result']['evidence'])} outcome rows. {d['sensitivity']} Authority: {d['authority']}."))
    else:
        return None
    return '\n'.join(out) + '\n'


def run(argv, stream):
    json_mode = '--json' in argv
    command = argv[0] if argv else 'unknown'
    response = {'schema_version': 'aver-decision-response/v1', 'command': command, 'status': 'completed', 'artifacts': [], 'warnings': [], 'errors': [], 'next_actions': []}
    code = 0
    try:
        args = parser().parse_args(argv)
        command = 'look' if args.operation == 'round' else args.operation
        value = None
        if command == 'capabilities': value = capabilities()
        elif command == 'next':
            next_code, value = wf.next_step(args.workspace)
        elif command == 'accept':
            value, pilot_rows, notes = wf.accept(read(args.proposal), args.prices, args.tolerances, args.monthly_uses, args.price_per_case, args.failure_limit, args.incident_limit,
                                                 args.app, args.checkpoint, args.candidates, args.baseline, args.budget_runs, args.horizon_months, args.purpose)
            response['warnings'].extend(notes)
            if args.pilot_out: response['artifacts'].append(write_jsonl(args.pilot_out, pilot_rows))
        elif command == 'cases':
            value = wf.round_cases(read(args.plan), args.round)
        elif command == 'schema':
            value = read(Path(__file__).resolve().parents[1] / 'aver_policy' / 'schemas' / f'decision-{args.name}.json')
        elif command == 'demo':
            if args.looks or args.outcome in ('continue', 'exhausted'):
                data = v2.demo(args.scenario, 'pass' if args.outcome == 'refuse' else args.outcome, args.looks or 3)
            else:
                data = demo(args.scenario, args.outcome)
            value = data['result']
            if args.out_dir:
                for key, filename in [('spec', 'decision-spec.json'), ('pilot', 'pilot.json'), ('plan', 'test-plan.json'), ('frozen_plan', 'frozen-plan.json'), ('confirmation', 'confirmation.json'), ('result', 'decision.json')]:
                    response['artifacts'].append(write(Path(args.out_dir) / filename, data[key]))
        elif command == 'init':
            value = {'schema': 'aver-decision-spec/v1', 'scope': {'app': '', 'checkpoint': ''}, 'candidates': [], 'baseline': '', 'strata': {}, 'definitions': {'completion': '', 'incident': '', 'grader': ''}, 'requirements': {'failure_limit': None, 'incident_limit': None}, 'monthly_uses': None, 'consequence_usd': None, 'budget_runs': None, 'population_statement': ''}
            if args.looks or args.deployment:
                value.update({'schema': v2.SPEC_SCHEMA, 'label_cost_usd': None, 'horizon_months': 12, 'current_failure_rate': None, 'grader_error': {'false_negative': 0, 'false_positive': 0}})
            if args.deployment:
                value.update({'schema': v2.SPEC_SCHEMA_V3, 'taxonomy': lr.standard_taxonomy(), 'deployment': {'mode': args.deployment, 'description': ''}})
                response['warnings'].append('Set a price (price_usd) or a tolerance (tolerance_per_month, occurrences a month) on each mistake kind, or remove the kind.')
            response['warnings'].append('Incomplete template. Choose your requirements; this is not a valid plan yet.')
        elif command == 'hosted-request':
            value = wf.hosted_request(args.operation, args.app, args.plan_id, read(args.evidence, True) if args.evidence else None, args.round)
        elif command == 'memo':
            value = decision_memo(read(args.result), args.monthly_fee)
        elif command == 'label-request':
            try:
                logs = Path(args.logs).read_text(encoding='utf-8')
            except (OSError, UnicodeError) as exc:
                raise DecisionError('unreadable_input', args.logs, 'Could not read the logs file.') from exc
            need(len(logs) <= 32_000_000, 'file_too_large', args.logs, 'Files must be at most 32 MB.')
            value = la.make_request(logs, args.model, args.include_content, args.format, lf_mod.parse_prices(args.prices) if args.prices else None)
            response['warnings'].append('Review the request before giving only its messages and response_schema to your labeling agent. No model was called. Return schema aver-label-response/v1, request_digest, model and response. A person must audit the proposed labels.')
        elif command == 'read-logs':
            need(args.reader == 'model-response' or not (args.label_request or args.label_response), 'invalid_arguments', 'reader', 'Use --reader model-response when supplying a labeling request or response.')
            if args.reader == 'mock-llm':
                need(args.allow_test_double, 'test_double_refused', 'reader', 'mock-llm is a test double that reads hidden truth labels. Pass --allow-test-double (tests and demos only).')
                need(bool(args.truth), 'test_double_refused', 'truth', 'mock-llm needs --truth (the hidden labels it answers from).')
                reader = lr.MockLLMReader(read(args.truth, True))
                response['warnings'].append('TEST DOUBLE: labels came from hidden truth with injected noise, not from reading the logs.')
            elif args.reader == 'model-response':
                need(bool(args.label_request) and bool(args.label_response), 'invalid_arguments', 'reader', 'model-response needs --label-request and --label-response.')
                reader = la.ResponseReader(read(args.label_request), read(args.label_response))
                response['warnings'].append('Model identity is reported by your labeling agent. These are proposed AI labels, never human audit labels.')
            else:
                reader = lr.HeuristicReader()
            spec = v2.validate_spec_v2(read(args.spec)) if args.spec else None
            try:
                text = Path(args.logs).read_text(encoding='utf-8')
            except (OSError, UnicodeError) as exc:
                raise DecisionError('unreadable_input', str(args.logs), 'Could not read the logs file.') from exc
            need(len(text) <= 32_000_000, 'file_too_large', args.logs, 'Files must be at most 32 MB.')
            need(bool(text.strip()), 'invalid_logs', args.logs, 'The logs file is empty.')
            value = lr.read_logs(text, reader, audit_n=args.audit_fine, spec=spec, audit_flagged=args.audit_flagged,
                                 prices=lf_mod.parse_prices(args.prices) if args.prices else None, log_format=args.format)
            fmt = value.get('format') or {}
            response['warnings'].append(f"Format: {fmt.get('name')} ({'chosen with --format' if args.format else 'detected from the structure: ' + str(fmt.get('reason'))}).")
            if (value.get('cost') or {}).get('unknown_runs'):
                response['warnings'].append(value['cost']['why'])
            if (value.get('labels_from') or {}).get('origin') == 'customer':
                response['warnings'].append('A proposal. Accept or edit the kinds and strata. The labels are the outcomes your harness recorded, so no audit is needed.')
            else:
                response['warnings'].append('A proposal. Accept or edit the kinds and strata, then a person audits the sample before any pass.')
        elif command == 'audit':
            if args.labels:
                need(bool(args.spec) and bool(args.sample), 'invalid_arguments', 'audit', 'Ingesting labels needs --spec and --sample.')
                spec = read(args.spec); sample = read(args.sample)
                if isinstance(sample, dict): sample = sample.get('audit_sample', sample.get('sample'))
                labels = read(args.labels, True)
                person = {}
                for l in labels:
                    lab = {'completed': l.get('completed'), 'incident': l.get('incident')}
                    if 'mistake' in l: lab['mistake'] = l['mistake']
                    person[(l['candidate_id'], l['case_id'])] = lab
                value = v2.apply_audit(v2.validate_spec_v2(spec), sample, person)
                g = value['grader_audit']
                if 'sampled_fine' in g:
                    response['warnings'].append(f"Audit recorded: {g['sampled_fine']} AI-fine runs checked ({g['missed_failures']} missed failures), {g['sampled_flagged']} AI-flagged runs checked ({g['false_flags']} false flags). Plans made from this spec compare the AI labels with audited limits.")
                else:
                    response['warnings'].append(f"Audit recorded: {g}.")
            else:
                need(bool(args.proposal), 'invalid_arguments', 'audit', 'Draw a sample with --proposal, or ingest labels with --spec --sample --labels.')
                prop = read(args.proposal)
                rows = prop['rows'] if isinstance(prop, dict) else prop
                spec = v2.validate_spec_v2(read(args.spec)) if args.spec else None
                size = lr.audit_size(rows, spec, args.fine, args.flagged)
                sample = v2.audit_sample(rows, size['fine_rows'], size['flagged_rows'], args.seed)
                value = {'schema': 'aver-audit-sample/v1', 'audit': size, 'audit_sample': sample, 'seed': args.seed,
                         'instructions': 'Re-label every case: {candidate_id, case_id, completed, incident, mistake} (mistake: a kind key or null). Do not look at the AI label first.'}
                response['warnings'].append(size['why'])
                if spec is not None and v2.is_v3(spec):
                    tight = wf.audit_plan(spec, rows)
                    if tight['fine_rows_needed_tightest'] is None or size['fine_rows'] < tight['fine_rows_needed_tightest']:
                        response['warnings'].append(f"This sample is too small for a pass on {tight['tightest']} ({tight['tightest_limit']:.2%}): that needs about "
                                                    + (f"{tight['fine_rows_needed_tightest']:,}" if tight['fine_rows_needed_tightest'] else 'more than any number of')
                                                    + f" AI-fine runs checked with no missed failures (draw them with --fine; the logs hold {tight['ai_fine_rows']:,}). A fail is still possible.")
        elif command == 'validate':
            spec = read(args.spec)
            spec = v2.validate_spec_v2(spec) if is_v2(spec) else validate_spec(spec)
            count = None
            if args.evidence:
                rows = read(args.evidence, True)
                if v2.is_v3(spec): v2.validate_mistakes(spec, rows); rows = v2.rows_v1(rows)
                count = len(validate_rows(v2.v1_view(spec) if is_v2(spec) else spec, rows, 'pilot'))
            value = {'valid': True, 'method_version': v2.method_of(spec) if is_v2(spec) else METHOD, 'pilot_rows': count}
        elif command == 'explore':
            spec = read(args.spec)
            pilot = read(args.pilot, True)
            if v2.is_v3(spec):
                spec = v2.validate_spec_v2(spec); v2.validate_mistakes(spec, pilot)
                value = explore(v2.v1_view(spec, v2.pooled_kappa(spec, pilot)), v2.rows_v1(pilot)); value['spec'] = v2.v1_view(spec); value['pilot'] = pilot
            else:
                value = explore(v2.v1_view(v2.validate_spec_v2(spec)) if is_v2(spec) else spec, pilot)
        elif command == 'plan':
            spec, ex = read(args.spec), read(args.exploration)
            spec_v1 = v2.v1_view(v2.validate_spec_v2(spec)) if is_v2(spec) else spec
            need(isinstance(ex, dict) and ex.get('schema') == 'aver-exploration/v1' and digest(spec_v1) == digest(ex.get('spec')), 'exploration_changed', 'exploration', 'Exploration belongs to a different specification. Run explore again.')
            if v2.is_v3(spec):
                again = explore(v2.v1_view(spec, v2.pooled_kappa(spec, ex.get('pilot'))), v2.rows_v1(ex.get('pilot'))); again['spec'] = spec_v1; again['pilot'] = ex.get('pilot')
                need(digest(again) == digest(ex), 'exploration_changed', 'exploration', 'Exploration ranking changed; recompute it.')
            else:
                need(digest(explore(spec_v1, ex.get('pilot'))) == digest(ex), 'exploration_changed', 'exploration', 'Exploration ranking changed; recompute it.')
            if args.looks or is_v2(spec):
                need(is_v2(spec), 'unsupported_schema', 'spec.schema', 'Planned looks need an aver-decision-spec/v2 specification (label cost, horizon, grader error). Run init --rounds 3.')
                spending = [float(x) for x in args.spending.split(',')] if args.spending else None
                paired = bool(args.paired or args.paired_limit is not None)
                if args.paired_limit is not None:
                    spec = dict(spec, requirements=dict(spec['requirements'], paired_limit_per_month=args.paired_limit))
                value = v2.make_plan(spec, ex['pilot'], args.finalists, args.looks or 3, args.cap, spending, args.allocation, args.incident_rate, paired=paired)
                oc = value['operating_characteristics']['scenarios']
                response['warnings'].append(f"Chance of a clear PASS at the final look: {oc['optimistic']['pass']:.0%} optimistic, {oc['central']['pass']:.0%} central, {oc['pessimistic']['pass']:.0%} pessimistic.")
                if paired:
                    response['warnings'].append(f"Paired test: every case also runs on {spec['baseline']}, so the test takes {value['confirmation_executions']:,} runs in all "
                                                f"for {value['allocation']['total']:,} cases (docs/140).")
            else:
                value = make_plan(spec, ex['pilot'], args.finalists)
            if not value['feasible']: response['warnings'].append('Infeasible even with zero failures. This plan cannot be frozen.')
        elif command == 'oc':
            plan = read(args.plan)
            if schema_of(plan) in ('aver-frozen-plan/v2', 'aver-frozen-plan/v3'): plan = plan['plan']
            need(schema_of(plan) in ('aver-test-plan/v2', 'aver-test-plan/v3'), 'unsupported_schema', 'plan', 'Operating characteristics need a v2 or v3 plan (made with --rounds).')
            spec = v2.spec_to_v2(plan['exploration']['spec'], plan); ex = plan['exploration']; cand = plan['selected'][0]
            if args.rate is not None:
                value = {'rate': args.rate, 'result': v2.simulate(spec, plan['allocation'], plan['looks'], plan['spending'], args.rate, plan['planning_incident_rate'], 800, 1)}
            else:
                value = {'operating_characteristics': plan['operating_characteristics']}
            if args.exchange:
                value['best_cap'] = v2.best_cap(spec, ex, cand, plan['looks'], plan['spending'], plan['planning_incident_rate'], plan['allocation']['mode'])
                value['move_the_bar'] = v2.move_the_bar(spec, ex, cand, plan['looks'], plan['spending'], plan['planning_incident_rate'], plan['allocation']['mode'])
        elif command == 'freeze':
            plan = read(args.plan)
            value = v2.freeze(plan) if is_v2(plan) else freeze(plan)
        elif command in ('evaluate', 'look'):
            frozen = read(args.plan)
            previous = read(args.previous) if args.previous else None
            if is_v2(frozen):
                rows = read(args.evidence, True)
                need(isinstance(rows, list), 'invalid_evidence', 'evidence', 'Outcomes are JSONL rows or a JSON array of rows (see look --help).')
                if previous is not None:
                    need(isinstance(previous, dict) and isinstance(previous.get('report'), dict) and previous['report'].get('stopped') == 'continue' and isinstance(previous.get('evidence'), list),
                         'plan_stopped', 'previous', 'The previous result did not say keep testing, so this plan takes no more rounds (or it is not a round result).')
                rows, note = wf.merge_round_rows(frozen, rows, previous)
                if note: response['warnings'].append(note)
                value = v2.look(frozen, rows, previous)
            else:
                need(command == 'evaluate', 'unsupported_schema', 'plan', 'look needs a v2 frozen plan; use evaluate for v1 plans.')
                value = evaluate(frozen, read(args.evidence, True), previous)
        elif command == 'explain':
            result = read(args.result) if args.result else None
            if result is not None and not is_v2(result):
                value = verify_result(result)
            else:
                if result is not None: result = v2.verify_result(result)
                plan = read(args.plan) if args.plan else (result['frozen_plan'] if result else None)
                if isinstance(plan, dict) and schema_of(plan) in ('aver-frozen-plan/v2', 'aver-frozen-plan/v3'): plan = plan['plan']
                need(bool(args.spec) or plan is not None, 'invalid_arguments', 'explain', 'Explain needs --spec, --plan or --result.')
                spec = read(args.spec) if args.spec else v2.spec_to_v2(plan['exploration']['spec'], plan)
                pilot = read(args.pilot, True) if args.pilot else None
                value = ex_mod.explain(spec, plan, result['report'] if result else None, pilot, args.what_if_price)
        elif command == 'bundle':
            result = read(args.result)
            value = v2.verify_result(result) if is_v2(result) else verify_result(result)
            value = {'schema': 'aver-decision-bundle/v1', 'result': value, 'sensitivity': 'Case identifiers, hashes, labels and costs; no raw prompts in this format.', 'authority': 'local_assessment_only'}
        if isinstance(value, dict) and schema_of(value) in ('aver-decision-result/v1', 'aver-decision-result/v2', 'aver-decision-result/v3'):
            report = value['report']
            response.update({k: report[k] for k in ['verdict', 'savings', 'release_eligibility', 'lineage', 'next_actions']})
            if 'stopped' in report: response['look'] = {'look': report['look'], 'looks': report['looks'], 'stopped': report['stopped'], 'evidence_spent': report['evidence_spent']}
            code = EXITS[report['verdict']['state']] if command in ['evaluate', 'look', 'demo'] else 0
        if args.out:
            if command == 'cases' and str(args.out).endswith('.jsonl'): response['artifacts'].append(write_jsonl(args.out, value['cases']))
            elif command != 'next': response['artifacts'].append(write(args.out, value))
        response['data'] = value
        if command == 'next':
            code = next_code; response['next_actions'] = [{'code': value['step']['stage'], 'actor': value['step']['actor'], 'automatic_retry': False}]
            if next_code == 3: response['status'] = 'inconsistent'
    except SystemExit as exc:
        return 0 if not exc.code else 3
    except (DecisionError, OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        response.update(status='error', errors=[{'code': getattr(exc, 'code', 'invalid_input'), 'path': getattr(exc, 'path', '$'), 'message': str(exc) if isinstance(exc, DecisionError) else 'Invalid input or file operation failed. Check the schema and paths.'}])
        code = 3
    if json_mode:
        json.dump(response, stream, ensure_ascii=False, allow_nan=False, sort_keys=True)
        stream.write('\n')
    else:
        stream.write('Aver decision — local assessment\n')
        if response['errors']:
            for e in response['errors']: stream.write(f"{e['code']} at {e['path']}: {e['message']}\n")
        elif 'verdict' in response:
            report = response['data']['report']
            if 'stopped' in report:
                stream.write(f"Round {report['look'] or '?'} of {report['looks']}: {STATE_WORDS.get(report['stopped'], report['stopped'].upper())} — {report['verdict']['reason_code']}\n")
            else:
                stream.write(f"{report['verdict']['state'].upper()}: {report['verdict']['reason_code']}\n")
            sv = report.get('savings') or {}
            savings = (f"at least ${sv['lower_usd_per_month']:,.0f} a month for {sv['candidate_id']} against the option in use now (a bound from the same cases; running costs are observed averages)"
                       if sv.get('status') == 'bounded' else
                       (f"shown: at least ${sv['lower_usd_per_month']:,.0f} a month for {sv['candidate_id']} (about ${sv['point_usd_per_month']:,.0f} at the rate seen; against the pilot rate of the option in use now)"
                        if sv['lower_usd_per_month'] > 0 else
                        f"likely, not shown: about ${sv['point_usd_per_month']:,.0f} a month for {sv['candidate_id']}, but at the highest failure rate this test allows it would be ${sv['lower_usd_per_month']:,.0f}")
                       if sv.get('status') == 'estimated' else 'not evaluated')
            stream.write(f"Choice: {report['choice'] or 'none'} | Savings: {savings} | Release: not evaluated\n")
            if report.get('audit', {}).get('status') == 'required': stream.write('Audit required: AI-labelled evidence cannot pass until a person relabels the audit sample (aver decision audit).\n')
            for c in report['candidates']:
                cost = c['modeled_usd_per_month']
                amount = 'unknown' if cost is None else f'${cost:,.2f}'
                extra = f" | holding it: {clause_words(c['binding'], report)}" if c.get('binding') else ''
                stream.write(f"  {c['candidate_id']}: {c['state']} | modeled monthly cost: {amount}{extra}\n")
                p = c.get('paired')
                if p:
                    up = p['per_month_upper']
                    most = f"at least {-up:,.0f} fewer a month" if up <= 0 else f"at most {up:,.0f} more a month"
                    stream.write(f"    against the option in use now, on the same {p['cases']:,} cases: {p['candidate_only']} failed only here, {p['current_only']} only there; "
                                 f"fails {most} (bound), paired limit {p['limit_per_month']:g} more a month: {p['state']}\n")
                for m, row in sorted((c.get('by_mistake') or {}).items()):
                    tol = f" | tolerance {row['tolerance_per_month']:g} a month: {row.get('state', '')}" if row['tolerance_per_month'] is not None else ''
                    price = f" | ${row['price_usd']:g} each" if row['price_usd'] is not None else ''
                    stream.write(f"    {m}: about {row['per_month']:.0f} a month, up to {row['per_month_upper']:.0f}{price}{tol}\n")
            for a in response['next_actions']: stream.write('Next: ' + a['code'] + '\n')
        elif schema_of(response.get('data')) == ex_mod.SCHEMA:
            stream.write(ex_mod.render_text(response['data']))
        elif schema_of(response.get('data')) == 'aver-log-proposal/v1':
            stream.write(proposal_text(response['data']))
        elif schema_of(response.get('data')) == wf.NEXT_SCHEMA:
            stream.write(wf.next_text(response['data'], code, ex_mod.wrap))
        elif schema_of(response.get('data')) == wf.CASES_SCHEMA:
            d = response['data']
            stream.write(f"Round {d['round']} of {d['rounds']}: {len(d['cases'])} runs ({len({c['case_id'] for c in d['cases']})} new cases x {', '.join(d['options'])}).\n")
            stream.write(ex_mod.wrap(d['how']) + '\n')
        elif schema_of(response.get('data')) == 'aver-decision-spec/v3' and command == 'accept':
            stream.write(spec_text(response['data']))
        elif schema_of(response.get('data')) == 'aver-audit-sample/v1':
            d = response['data']
            stream.write(f"Audit sample: {sum(s['phase'] == 'fine' for s in d['audit_sample'])} runs the reader called fine, {sum(s['phase'] == 'flagged' for s in d['audit_sample'])} it flagged.\n")
        elif artifact_text(response.get('data')) is not None:
            stream.write(artifact_text(response['data']))
        else:
            json.dump(response.get('data'), stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
        for w in response['warnings']: stream.write('Note: ' + w + '\n')
        for path in response['artifacts']: stream.write('Saved: ' + path + '\n')
    return code
