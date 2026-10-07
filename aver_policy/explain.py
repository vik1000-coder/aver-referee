"""Explain reliability in the user's own terms (docs/132 §4).

explain(spec, plan=None, report=None, pilot_rows=None) -> dict
render_text(explanation) -> str (plain text, lines of at most 80 columns)

A pure, deterministic narrative of what the evidence says: what the agent is for, what was tested, where each option
stands mistake by mistake (seen -> rate -> N a month at the user's volume, with the bound at the error limit next to the
user's price or tolerance), what decides, the money all-in per month, what would change the answer, the audit, and what
the evidence does not show.

Every number is taken from the engine (decision_v2: the plan, the look report, decide, economics, effective_kappa,
tolerance_floor) or is exact arithmetic on those numbers (x per use times volume; a break-even solved from the same
formula economics uses). Nothing is re-estimated. Words: a pass is a bound at an error limit, never a stronger word; no option
is said to pass unless the report stopped at 'pass'.
"""
from __future__ import annotations

import copy
import textwrap

from aver_policy import decision_v2 as v2

SCHEMA = 'aver-explanation/v1'
WIDTH = 78


# ----------------------------------------------------------------------------- small helpers

def _n(x):
    """A count a month, rounded the way a person says it."""
    if x is None: return '?'
    if x >= 100: return f'{x:,.0f}'
    if x >= 10: return f'{x:.0f}'
    if x >= 1: return f'{x:.1f}'.rstrip('0').rstrip('.')
    return f'{x:.2f}'.rstrip('0').rstrip('.') if x > 0 else '0'


def _usd(x):
    if x is None: return 'unknown'
    return f'${x:,.0f}' if abs(x) >= 100 else f'${x:,.2f}'


def _pct(x):
    if x is None: return '?'
    return f'{x:.2%}' if x < .01 else f'{x:.1%}'


def _kinds(spec):
    return (spec.get('taxonomy') or {}).get('kinds') or {}


def _label(spec, m):
    return _kinds(spec).get(m, {}).get('label', m)


def _ordered_candidates(spec):
    ids = [c['id'] for c in spec['candidates']]
    return [spec['baseline']] + [c for c in ids if c != spec['baseline']]


def _clause_words(spec, key):
    V = spec['monthly_uses']
    if key == 'average:failure':
        L = spec['requirements']['failure_limit']
        return f'the overall failure limit ({_pct(L)}, about {_n(L * V)} failed tasks a month)'
    if key.startswith('tolerance:'):
        m = key.split(':', 1)[1]
        return f"the tolerance for \"{_label(spec, m)}\" ({_n(_kinds(spec)[m]['tolerance_per_month'])} a month)"
    s, metric = key.rsplit(':', 1)
    if metric in ('incident',):
        return f"the incident limit for {s} cases ({_pct(spec['requirements']['incident_limit'])})"
    return f"the failure limit for {s} cases ({_pct(spec['strata'][s]['failure_limit'])})"


def _binding_key(binding):
    """decide()'s binding string -> clause key."""
    if not binding or binding == 'no evidence': return None
    if binding == 'average failure limit': return 'average:failure'
    if binding.endswith(' tolerance'): return 'tolerance:' + binding[:-len(' tolerance')]
    if binding.endswith(' incident limit'): return binding[:-len(' incident limit')] + ':incident'
    if binding.endswith(' failure limit'): return binding[:-len(' failure limit')] + ':failed'
    return None


def _pilot_counts(spec, rows, candidate_id):
    counts = {}
    kinds = sorted(_kinds(spec))
    for s in sorted(spec['strata']):
        rs = [r for r in rows if r['candidate_id'] == candidate_id and r['stratum'] == s]
        counts[s] = {'n': len(rs), 'failed': sum(r['completed'] is False for r in rs), 'incident': sum(r['incident'] is True for r in rs),
                     'unknown_failed': sum(r['completed'] is None for r in rs), 'unknown_incident': sum(r['incident'] is None for r in rs),
                     'by_mistake': {m: sum(r.get('mistake') == m for r in rs) for m in kinds}}
    return counts


# ----------------------------------------------------------------------------- the narrative

def explain(spec, plan=None, report=None, pilot_rows=None, what_if_price=None):
    """See the module docstring. `report` is look(...)['report'] (or a whole result); `pilot_rows` default to the plan's."""
    spec = v2.validate_spec_v2(copy.deepcopy(spec))
    if isinstance(report, dict) and 'report' in report and 'frozen_plan' in report: report = report['report']
    if plan is not None and plan.get('schema', '').startswith('aver-frozen-plan'): plan = plan['plan']
    if pilot_rows is None and plan is not None: pilot_rows = plan['exploration']['pilot']
    pilot_rows = pilot_rows or []
    V = spec['monthly_uses']
    stopped = report['stopped'] if report else None
    out = {'schema': SCHEMA, 'method_version': (report or {}).get('method_version') or (plan or {}).get('method_version') or v2.method_of(spec),
           'monthly_uses': V, 'stopped': stopped}
    out['purpose'] = _purpose(spec)
    out['what_was_tested'] = _tested(spec, plan, report, pilot_rows)
    out['where_it_stands'] = _stands(spec, plan, report, pilot_rows)
    out['what_decides'] = _decides(spec, plan, report)
    out['money'] = _money(spec, plan, report)
    out['what_would_change_the_answer'] = _changes(spec, plan, report, pilot_rows, what_if_price)
    out['audit'] = _audit(spec, plan, report, pilot_rows)
    out['caveats'] = _caveats(spec, plan, report, pilot_rows)
    out['headline'] = _headline(spec, plan, report, out)
    return out


def _purpose(spec):
    dep = spec.get('deployment') or {'mode': 'normal', 'description': ''}
    text = dep.get('description') or spec.get('population_statement') or ''
    mode = dep['mode']
    how = ('It acts on its own (agentic): a mistake can take an action before anyone sees it, so actions are usually limited by a '
           'tolerance (at most N a month) rather than priced.' if mode == 'agentic' else
           'A person sees its output before it matters (normal deployment): most mistakes cost review time and are priced.')
    return {'text': text, 'deployment': mode, 'deployment_text': how}


def _tested(spec, plan, report, pilot_rows):
    V = spec['monthly_uses']
    cands = _ordered_candidates(spec)
    pilot_cases = len({r['case_id'] for r in pilot_rows})
    strata = [{'name': s, 'weight': spec['strata'][s]['weight'], 'pilot_cases': len({r['case_id'] for r in pilot_rows if r['stratum'] == s})} for s in sorted(spec['strata'], key=lambda s: -spec['strata'][s]['weight'])]
    origins = sorted({r['origin'] for r in pilot_rows})
    parts = [f'{pilot_cases} logged cases were run on each of {len(cands)} options ({", ".join(cands)}; the one in use now: {spec["baseline"]})' if pilot_cases else f'{len(cands)} options ({", ".join(cands)})']
    parts.append('split into ' + ', '.join(f"{x['name']} ({x['weight']:.0%} of traffic)" for x in strata))
    t = '; '.join(parts) + f'. Your volume: {V:,} uses a month.'
    rounds = None
    if plan is not None:
        rounds = {'planned': plan['looks'], 'cases_per_option': plan['allocation']['total'], 'options_tested': plan['selected'],
                  'round': report['look'] if report else None, 'confirmation_cases': report['evidence_spent']['cases'] if report else 0}
        t += (f" The test plan checks {', '.join(plan['selected'])} on up to {plan['allocation']['total']:,} new cases in {plan['looks']} rounds.")
        if report:
            t += f" Round {report['look'] or '?'} of {plan['looks']} is in: {report['evidence_spent']['cases']:,} confirmation cases so far" + (f" across {len(plan['selected'])} options." if len(plan['selected']) > 1 else '.')
        else:
            t += ' No round has been run yet.'
    return {'text': t, 'pilot_cases': pilot_cases, 'candidates': cands, 'baseline': spec['baseline'], 'strata': strata, 'origins': origins, 'rounds': rounds}


def _kind_line(spec, m, row, source, stopped, audited):
    """'Unwanted writes: 2 seen in 600 cases -> about 100 a month; at most 450 a month at the error limit; your tolerance is
    450 -> within it.' The bound is the engine's per_month_upper."""
    k = _kinds(spec).get(m, {})
    lab = _label(spec, m)
    seen, cases = row['count'], row['cases']
    words = f"{lab}: {seen} seen in {cases:,} cases -> about {_n(row['per_month'])} a month"
    words += f"; at most {_n(row['per_month_upper'])} a month at the error limit" if source == 'confirmation' else f" (pilot estimate; it could be up to {_n(row['per_month_upper'])})"
    tol_state = None
    if k.get('tolerance_per_month') is not None:
        tol = k['tolerance_per_month']
        lim = audited.get(f'tolerance:{m}') if audited else None
        if source == 'confirmation':
            tol_state = row.get('state')
            verdict = {'pass': 'passes' if stopped == 'pass' else 'within it', 'block': 'over it: fails', 'refuse': 'not settled yet'}.get(tol_state, 'not settled yet')
            words += f'; your tolerance is {_n(tol)} -> {verdict}'
            if audited and f'tolerance:{m}' in audited:
                words += (f' (compared on AI labels with {_n(lim * spec["monthly_uses"])} a month after the audit)' if lim is not None else ' (the audit is too small for this tolerance to pass)')
        else:
            words += f'; your tolerance is {_n(tol)} (tested only in the rounds)'
    if k.get('price_usd') is not None and row['per_month'] > 0:
        words += f"; at ${k['price_usd']:g} each, about {_usd(k['price_usd'] * row['per_month'])} a month"
    if k.get('price_usd') is None and k.get('tolerance_per_month') is None:
        words += '; not priced, not limited (shown so you can decide)'
    return {'kind': m, 'label': lab, 'severity': k.get('severity'), 'seen': seen, 'cases': cases, 'rate': row['rate'], 'per_month': row['per_month'],
            'per_month_upper': row['per_month_upper'], 'price_usd': k.get('price_usd'), 'tolerance_per_month': k.get('tolerance_per_month'),
            'tolerance_state': tol_state, 'text': words + '.'}


def _stands(spec, plan, report, pilot_rows):
    out = []
    summaries = {c['candidate_id']: c for c in (report or {}).get('candidates', []) if c.get('clauses')}
    audited = ((report or {}).get('audit') or {}).get('audited_limit') or {}
    for c in _ordered_candidates(spec):
        if c in summaries and summaries[c].get('by_mistake') is not None:
            src, rows, state = 'confirmation', summaries[c]['by_mistake'], summaries[c]['state']
        elif pilot_rows and any(r['candidate_id'] == c for r in pilot_rows):
            d = v2.decide(_pilot_counts(spec, pilot_rows, c), spec, 1, 1, 1, [1.])
            src, rows, state = 'pilot', d.get('by_mistake') or {}, None
        else:
            continue
        kinds = [_kind_line(spec, m, rows[m], src, (report or {}).get('stopped'), audited if src == 'confirmation' else None) for m in sorted(rows, key=lambda m: (_kinds(spec)[m]['severity'] != 'incident', m))]
        head = (f'{c} (confirmation evidence, round {report["look"]}):' if src == 'confirmation' else f'{c} (pilot only: estimates, not a test):')
        out.append({'candidate_id': c, 'source': src, 'state': state, 'kinds': kinds, 'text': head})
    return out


def _decides(spec, plan, report):
    if not report:
        floors = (plan or {}).get('tolerance_floors') or {}
        bits = [f"\"{_label(spec, m)}\" needs about {f:,} cases with none seen to be shown under {_n(_kinds(spec)[m]['tolerance_per_month'])} a month"
                for m, f in sorted(floors.items()) if f is not None]
        t = 'Nothing is decided until the first round is in.' + (' ' + '; '.join(bits) + '.' if bits else '')
        return {'text': t, 'by_candidate': []}
    V = spec['monthly_uses']; out = []
    audit = report.get('audit') or {}
    for c in report['candidates']:
        if not c.get('clauses'):
            out.append({'candidate_id': c['candidate_id'], 'clause_key': None, 'state': c['state'], 'text': f"{c['candidate_id']}: this round is incomplete."}); continue
        blocking = [x for x in c['clauses'] if x['state'] == 'block']
        if c['state'] == 'block' and blocking:
            x = max(blocking, key=lambda x: x['lower'] / x['limit'] if x['limit'] else 0)
            t = (f"{c['candidate_id']} fails on {_clause_words(spec, x['key'])}: even the low end of the evidence ({_pct(x['lower'])}, about {_n(x['lower'] * V)} a month) "
                 f"is over the limit it is compared with ({_pct(x['limit'])}). More cases will not change this; change the option or the requirement.")
            out.append({'candidate_id': c['candidate_id'], 'clause_key': x['key'], 'state': 'block', 'upper': x['upper'], 'lower': x['lower'], 'limit': x['limit'], 'text': t}); continue
        if c['state'] == 'pass':
            tight = max(c['clauses'], key=lambda x: (x['upper'] / x['limit']) if x['limit'] else 0)
            if report['stopped'] == 'pass':
                t = (f"{c['candidate_id']}: every clause holds at the error limit. Closest to its limit: {_clause_words(spec, tight['key'])} "
                     f"(bound {_pct(tight['upper'])} against {_pct(tight['limit'])}).")
            else:
                why = 'a person has audited the AI labels' if audit.get('status') == 'required' else 'the audit is large enough'
                t = (f"{c['candidate_id']}: every clause holds on the labels so far, but it cannot pass until {why}. Closest to its limit: "
                     f"{_clause_words(spec, tight['key'])} (bound {_pct(tight['upper'])} against {_pct(tight['limit'])}).")
            out.append({'candidate_id': c['candidate_id'], 'clause_key': tight['key'], 'state': 'pass', 'upper': tight['upper'], 'limit': tight['limit'], 'text': t}); continue
        key = _binding_key(c.get('binding'))
        x = next((x for x in c['clauses'] if x['key'] == key), None) or max((x for x in c['clauses'] if x['state'] != 'pass'), key=lambda x: (x['upper'] / x['limit']) if x['limit'] else 9e9)
        lim = x['limit']
        if lim is None:
            t = f"{c['candidate_id']}: held by {_clause_words(spec, x['key'])}: with this audit no evidence can pass it."
        else:
            t = (f"{c['candidate_id']}: held by {_clause_words(spec, x['key'])}. The evidence can show at most {_pct(x['upper'])} "
                 f"(about {_n(x['upper'] * V)} a month); it must get to {_pct(lim)} (about {_n(lim * V)} a month).")
        move = []
        if report.get('next_look_executions'): move.append(f"round {report['look'] + 1} brings the total to {report['next_look_executions']:,} cases, which tightens the bound if the rate holds")
        if x['key'].startswith('tolerance:'): move.append('loosening the tolerance, or pricing this mistake instead, removes or relaxes this clause')
        elif x['key'].endswith(':incident'): move.append(f"a higher incident limit for {x['key'].split(':')[0]} cases, or more of those cases")
        elif x['key'] == 'average:failure': move.append('a looser overall failure limit')
        if move: t += ' What would move it: ' + '; '.join(move) + '.'
        out.append({'candidate_id': c['candidate_id'], 'clause_key': x['key'], 'state': c['state'], 'upper': x['upper'], 'limit': lim, 'text': t})
    return {'text': ' '.join(o['text'] for o in out), 'by_candidate': out}


def _all_in(spec, ranking_row, kappa):
    """V * (execution per use + kappa * failure rate): the same terms decision_v2.economics uses."""
    if ranking_row['execution_cost_usd'] is None or kappa is None: return None
    return spec['monthly_uses'] * (ranking_row['execution_cost_usd'] + kappa * ranking_row['failure_rate'])


def _money(spec, plan, report):
    if plan is None:
        return {'text': 'Money needs a plan (pilot costs and failure rates).', 'options': [], 'baseline': spec['baseline'], 'break_even': [], 'saving': None}
    ex = plan['exploration']; V = spec['monthly_uses']
    rank = {r['candidate_id']: r for r in ex['ranking']}
    kappa = plan.get('kappa') or {c: v2.effective_kappa(spec, ex['pilot'], c) for c in rank}
    base = spec['baseline']; opts = []
    base_all = _all_in(spec, rank[base], kappa[base])
    confirmed = {c['candidate_id']: c.get('modeled_usd_per_month') for c in (report or {}).get('candidates', [])}
    for c in _ordered_candidates(spec):
        r = rank[c]
        saving = v2.economics(spec, ex, c, r['failure_rate'])
        opts.append({'candidate_id': c, 'execution_usd_per_use': r['execution_cost_usd'], 'execution_usd_per_month': None if r['execution_cost_usd'] is None else r['execution_cost_usd'] * V,
                     'kappa_usd': kappa[c], 'failure_rate': r['failure_rate'], 'mistakes_usd_per_month': None if kappa[c] is None else kappa[c] * r['failure_rate'] * V,
                     'all_in_usd_per_month': _all_in(spec, r, kappa[c]), 'saving_vs_baseline_usd_per_month': saving,
                     'confirmed_all_in_usd_per_month': confirmed.get(c)})
    lines = []
    for o in opts:
        s = (f"{o['candidate_id']}: {_usd(o['all_in_usd_per_month'])} a month all-in = {_usd(o['execution_usd_per_month'])} to run "
             f"+ {_usd(o['mistakes_usd_per_month'])} in priced mistakes ({_pct(o['failure_rate'])} of pilot runs failed; {_usd(o['kappa_usd'])} per failure on average)")
        if o['candidate_id'] != base and o['saving_vs_baseline_usd_per_month'] is not None:
            sv = o['saving_vs_baseline_usd_per_month']
            s += f"; {'saves' if sv >= 0 else 'costs'} {_usd(abs(sv))} a month against {base}"
        if o['confirmed_all_in_usd_per_month'] is not None:
            s += f"; on the confirmation rounds so far: {_usd(o['confirmed_all_in_usd_per_month'])} a month"
        lines.append(s + '.')
    be = []
    for o in opts:
        c = o['candidate_id']
        if c == base or base_all is None or o['execution_usd_per_use'] is None: continue
        if kappa[c]:
            r_star = (base_all / V - o['execution_usd_per_use']) / kappa[c]
            if r_star >= 1:
                be.append({'candidate_id': c, 'failure_rate': None, 'failed_per_month': None,
                           'text': f'{c} stays cheaper all-in than {base} at any failure rate: almost none of its mistakes carry a price (tolerances still apply).'})
            elif r_star <= 0:
                be.append({'candidate_id': c, 'failure_rate': None, 'failed_per_month': None, 'text': f'{c} costs more than {base} to run alone, so it cannot save money at any failure rate.'})
            else:
                be.append({'candidate_id': c, 'failure_rate': r_star, 'failed_per_month': r_star * V,
                           'text': f"{c} stays cheaper all-in than {base} while its failure rate stays below {_pct(r_star)} (about {_n(r_star * V)} failed tasks a month, at its mix of mistakes); in the pilot it was {_pct(o['failure_rate'])}."})
        else:
            diff = base_all - o['execution_usd_per_use'] * V
            be.append({'candidate_id': c, 'failure_rate': None, 'failed_per_month': None,
                       'text': f"{c}'s mistakes carry no price, so only running cost counts: it is {_usd(abs(diff))} a month {'cheaper' if diff >= 0 else 'dearer'} than {base} all-in whatever its failure rate (tolerances still apply)."})
    sv = _saving_words(spec, report)
    text = ' '.join(lines) + (f" {sv['text']}" if sv else '') + (f" {report['limitations'][-1]}" if sv else ' Uses pilot costs and failure rates with your prices; savings are not confirmed.')
    return {'text': text, 'options': opts, 'baseline': base, 'break_even': be, 'saving': sv}


def _saving_words(spec, report):
    """report.savings in words (docs/140 §5 for a bound; docs/148 for the hedged estimate): None when not evaluated."""
    sv = (report or {}).get('savings') or {}
    if sv.get('status') not in ('bounded', 'estimated'): return None
    cid, base = sv['candidate_id'], spec['baseline']; lo, pt = sv['lower_usd_per_month'], sv.get('point_usd_per_month')
    if sv['status'] == 'bounded':
        text = (f"{cid} saves at least {_usd(lo)} a month against {base} (a result you can rely on: both ran on the same cases"
                + (f"; about {_usd(pt)} as an estimate)." if pt is not None else ').') if lo > 0 else
                f"On the same cases the saving is not shown: {cid} could cost up to {_usd(-lo)} a month more than {base}." if lo < 0 else
                f"On the same cases the saving is not shown: it could be nothing.")
        rec = (f"Switch to {cid}: saves at least {_usd(lo)} a month (a result you can rely on)." if lo > 0 else
               f"{cid} meets your limits, but the test does not show it saving money — keep {base}.")
        return {'status': 'bounded', 'candidate_id': cid, 'shown': lo > 0, 'lower_usd_per_month': lo, 'point_usd_per_month': pt, 'text': text, 'recommendation': rec}
    if lo > 0:
        text = (f"{cid} saves at least {_usd(lo)} a month against {base}, even at the highest failure rate this test allows (about {_usd(pt)} at the rate seen). "
                f"{base}'s rate comes from the pilot, not from this test.")
        rec = f"Switch to {cid}: saves at least {_usd(lo)} a month (shown by the test, against {base}'s logged rate)."
    elif pt > 0:
        text = (f"{cid} would likely save about {_usd(pt)} a month against {base}, but a test this size cannot show the saving is real: at the highest failure "
                f"rate it allows, " + (f"{cid} would cost about {_usd(-lo)} a month more." if lo < 0 else 'the saving could be nothing.'))
        rec = (f"{cid} meets your limits. It would likely save about {_usd(pt)} a month, but a test this size cannot show the saving is real — keep {base} unless "
               f"running cost matters more. Aver says 'switch' only when the saving is shown, not estimated.")
    else:
        text = f"At these rates {cid} would not save money against {base}: about {_usd(-pt)} a month more, mistakes included."
        rec = f"{cid} meets your limits, but at these prices it would not save money — keep {base}."
    return {'status': 'estimated', 'candidate_id': cid, 'shown': lo > 0, 'lower_usd_per_month': lo, 'point_usd_per_month': pt, 'level': sv['level'], 'text': text, 'recommendation': rec}


def _changes(spec, plan, report, pilot_rows, what_if_price):
    out = []
    kinds = _kinds(spec)
    tolerated = sorted(m for m, k in kinds.items() if k['tolerance_per_month'] is not None)
    priced = [k['price_usd'] for k in kinds.values() if k['price_usd']]
    P = what_if_price if what_if_price is not None else (max(priced) if priced else 10.)
    if plan is not None and tolerated:
        ex = plan['exploration']; rank = {r['candidate_id']: r for r in ex['ranking']}
        for m in tolerated:
            s2 = copy.deepcopy(spec); s2['taxonomy']['kinds'][m]['price_usd'] = float(P); s2['taxonomy']['kinds'][m]['tolerance_per_month'] = None
            now, then = {}, {}
            for c in _ordered_candidates(spec):
                now[c] = _all_in(spec, rank[c], (plan.get('kappa') or {}).get(c, v2.effective_kappa(spec, ex['pilot'], c)))
                then[c] = _all_in(spec, rank[c], v2.effective_kappa(s2, ex['pilot'], c))
            if all(abs((then[c] or 0) - (now[c] or 0)) < .005 for c in now):
                text = (f"If \"{_label(spec, m)}\" were priced at ${P:g} each instead of tolerated, the all-in estimates would not move: none was seen in "
                        f"the pilot. What would change is the test: without the tolerance there is no clause to prove, and no minimum number of cases for it.")
            else:
                text = (f"If \"{_label(spec, m)}\" were priced at ${P:g} each instead of tolerated, the all-in cost a month would be "
                        + ', '.join(f'{c} {_usd(then[c])} (now {_usd(now[c])})' for c in now) + '.')
            out.append({'change': 'price_instead_of_tolerance', 'kind': m, 'price_usd': float(P), 'all_in_now_usd_per_month': now, 'all_in_then_usd_per_month': then, 'text': text})
            floor_now = (plan.get('tolerance_floors') or {}).get(m)
            if floor_now is not None:
                tol = kinds[m]['tolerance_per_month']; looser = tol * 1.5
                s3 = copy.deepcopy(spec); s3['taxonomy']['kinds'][m]['tolerance_per_month'] = looser
                f3 = v2.tolerance_floor(s3, m, plan['looks'], plan['spending'], len(plan['selected']), plan['allocation']['mode'])
                if f3 is not None:
                    out.append({'change': 'loosen_tolerance', 'kind': m, 'tolerance_per_month': tol, 'looser_per_month': looser, 'cases_now': floor_now, 'cases_then': f3,
                                'text': f"Loosening the tolerance for \"{_label(spec, m)}\" from {_n(tol)} to {_n(looser)} a month would cut the cases needed to show it "
                                        f"with none seen from about {floor_now:,} to {f3:,} ({floor_now - f3:,} fewer)."})
                s4 = copy.deepcopy(spec); s4['monthly_uses'] = spec['monthly_uses'] * 2
                f4 = v2.tolerance_floor(s4, m, plan['looks'], plan['spending'], len(plan['selected']), plan['allocation']['mode'])
                out.append({'change': 'double_volume', 'kind': m, 'monthly_uses': s4['monthly_uses'], 'cases_now': floor_now, 'cases_then': f4,
                            'text': f"At twice the volume ({s4['monthly_uses']:,} uses a month) the same {_n(kinds[m]['tolerance_per_month'])} a month is half the rate: "
                                    + (f'showing it would need about {f4:,} cases instead of {floor_now:,}.' if f4 is not None else 'it could not be shown within 200,000 cases.')})
    if report and report.get('next_look_executions') and report['stopped'] == 'continue':
        out.append({'change': 'next_round', 'cases_then': report['next_look_executions'],
                    'text': f"Running round {report['look'] + 1} ({report['next_look_executions']:,} cases in total) is the next step; the answer can change with it."})
    if not out:
        out.append({'change': 'none', 'text': 'Set a price or a tolerance on a mistake kind to see what would change the answer.'})
    return out


def _audit(spec, plan, report, pilot_rows):
    ai = any(r.get('origin') == 'ai-labeled' for r in pilot_rows)
    L = spec['requirements']['failure_limit']; V = spec['monthly_uses']
    plan_audit = (plan or {}).get('audit') or {}
    need_fine = plan_audit.get('fine_rows_needed')
    rep = (report or {}).get('audit')
    g = spec.get('grader_audit')
    if rep is None:
        status = 'applied' if g else ('required' if ai else 'not_needed')
        rep = {'status': status, 'applied': bool(g)}
        if g: rep['counts'] = dict(g)
    out = {'status': rep['status'], 'applied': rep.get('applied', False), 'fine_rows_needed': rep.get('fine_rows_needed', need_fine),
           'flagged_rows_suggested': plan_audit.get('flagged_rows_suggested')}
    if rep['status'] == 'not_needed':
        out['text'] = 'The labels did not come from an AI reader, so no audit is needed.'
        return out
    if rep['status'] == 'required':
        out['still_needed_fine'] = need_fine
        out['text'] = ('The labels were proposed by an AI log reader, so no option can pass until a person re-labels a random sample: '
                       + (f'about {need_fine:,} runs the reader called fine, with no missed failures, ' if need_fine else 'a few hundred runs the reader called fine, ')
                       + (f'and up to {plan_audit["flagged_rows_suggested"]:,} it flagged. ' if plan_audit.get('flagged_rows_suggested') is not None else 'and the runs it flagged. ')
                       + (f'That is the size at which a pass becomes possible at your {_pct(L)} failure limit; it is not a formality.' if need_fine else ''))
        return out
    counts = rep.get('counts') or g or {}
    m, a, h, b = counts.get('sampled_fine'), counts.get('missed_failures'), counts.get('sampled_flagged'), counts.get('false_flags')
    out.update({'checked_fine': m, 'missed_failures': a, 'checked_flagged': h, 'false_flags': b})
    used = rep.get('audited_limit') or {}
    if not used and g and 'sampled_fine' in g:
        used = v2.audited_limits(spec)
    avg = used.get('average:failure')
    out['limit'] = L; out['audited_limit'] = avg
    t = f'A person checked {m} runs the reader called fine and found {a} it missed, and {h} runs it flagged, of which {b} were fine. '
    if avg is not None:
        why = ('the failures it may have missed outweigh its false flags' if avg < L else
               'it wrongly flags enough fine runs that its failure counts run high')
        t += (f'Allowing for both, the AI labels are compared with a failure limit of {_pct(avg)} instead of your {_pct(L)} '
              f'(about {_n(avg * V)} instead of {_n(L * V)} failed tasks a month): {"lower" if avg < L else "higher"}, because {why}.')
    else:
        t += 'That audit is too small: with it, no evidence can pass your failure limit.'
    still = None
    if need_fine and m is not None and m < need_fine: still = need_fine - m
    if rep.get('fine_rows_needed') and m is not None and rep['fine_rows_needed'] > m: still = rep['fine_rows_needed'] - m
    if still:
        t += f' Check about {still:,} more fine-labelled runs (with no missed failures) to make a pass possible.'
    out['still_needed_fine'] = still
    out['text'] = t
    return out


def _caveats(spec, plan, report, pilot_rows):
    c = []
    strata = ', '.join(sorted(spec['strata']))
    n = len({r['case_id'] for r in pilot_rows})
    c.append(f'These numbers describe traffic like the {n:,} logged cases, in the kinds of case {strata}, at the stated mix. '
             'Cases unlike these, or a different mix, are not covered.' if n else f'These numbers describe traffic in the kinds of case {strata} at the stated mix.')
    c.append('If the agent changes (model version, prompt, tools) or the traffic shifts, the bound no longer applies until it is re-tested.')
    origins = sorted({r['origin'] for r in pilot_rows})
    if 'ai-labeled' in origins:
        c.append(f"Outcomes were labelled by an AI log reader ({spec['definitions']['grader']}); they count only as far as the audit supports them.")
    if 'synthetic' in origins:
        c.append('The evidence is synthetic: it shows how the method behaves, not how your agent does.')
    if 'external' in origins:
        c.append('The evidence comes from public benchmark data, not your own traffic.')
    if report is None:
        c.append('Without a confirmation round these are pilot estimates from a small sample, not a test.')
    sv = (report or {}).get('savings') or {}
    c.append(report['limitations'][-1] if sv.get('status') in ('bounded', 'estimated') else 'Money uses pilot costs and failure rates with your prices; savings have not been confirmed against current.')
    if not report or report['stopped'] != 'pass':
        c.append('No option has passed yet.')
    c.append('A pass is a bound at a 5% error limit on this evidence, not a promise about every future case.')
    return c


def _headline(spec, plan, report, ex):
    if not report:
        return {'state': 'planned' if plan else 'pilot', 'text': ('Planned, not yet tested: ' if plan else 'Pilot only: ') + 'nothing has passed or failed.'}
    s = report['stopped']; reason = report['verdict']['reason_code']
    if s == 'pass':
        tested = [c['candidate_id'] for c in report['candidates']]
        t = (f"{report['choice']} passes: every clause holds at the error limit" + (', and it is the cheapest of the tested options that do.' if len(tested) > 1 else '.'))
        sv = ex['money'].get('saving')
        if sv: t += ' ' + sv['recommendation']
    elif s == 'block':
        t = 'Fails: the evidence shows ' + ' and '.join(d['candidate_id'] for d in ex['what_decides']['by_candidate'] if d['state'] == 'block') + ' over a limit.'
    elif reason == 'audit_required':
        ok = [c['candidate_id'] for c in report['candidates'] if c['state'] == 'pass']
        bad = [c['candidate_id'] for c in report['candidates'] if c['state'] == 'block']
        t = (f"Not decided: {' and '.join(ok)} {'holds' if len(ok) == 1 else 'hold'} every clause on the AI labels, but no option can pass until a person "
             f"audits a sample (round {report['look']} of {report['looks']}).") + (f" {' and '.join(bad)} already {'fails' if len(bad) == 1 else 'fail'} a limit." if bad else '')
    elif reason == 'audit_too_small':
        t = f"Not decided: the audit is too small for a pass at these limits (round {report['look']} of {report['looks']})."
    elif s == 'continue':
        t = f"Not decided yet: keep testing (round {report['look']} of {report['looks']})."
    elif s == 'exhausted':
        t = f"Inconclusive after all {report['looks']} rounds: keep current, plan a new test, or change the requirement."
    else:
        t = 'This round is incomplete.'
    return {'state': s, 'reason_code': reason, 'text': t}


# ----------------------------------------------------------------------------- plain text

def wrap(text, indent='', width=None):
    """Plain-text wrapping shared with the CLI (which may import only aver_policy, aver_cass and an allow-list of the stdlib)."""
    return textwrap.fill(text, width=width or WIDTH, initial_indent=indent, subsequent_indent=' ' * len(indent), break_long_words=True, break_on_hyphens=False)


_wrap = wrap


def render_text(e):
    """Plain text, no tables, every line at most 80 columns."""
    out = ['AVER: RELIABILITY IN YOUR TERMS', _wrap(e['headline']['text']), '']
    out += ['WHAT THE AGENT IS FOR', _wrap(e['purpose']['text'] or '(no description given)'), _wrap(e['purpose']['deployment_text']), '']
    out += ['WHAT WAS TESTED', _wrap(e['what_was_tested']['text']), '']
    out.append('WHERE EACH OPTION STANDS')
    for c in e['where_it_stands']:
        out.append(_wrap(c['text']))
        quiet = []
        for k in c['kinds']:
            if c['source'] == 'pilot' and k['seen'] == 0 and k['tolerance_per_month'] is None: quiet.append(k['label'])
            else: out.append(_wrap(k['text'], '  - '))
        if quiet: out.append(_wrap('Not seen in the pilot: ' + '; '.join(quiet) + '.', '  - '))
    out += ['', 'WHAT DECIDES']
    if e['what_decides']['by_candidate']:
        for d in e['what_decides']['by_candidate']: out.append(_wrap(d['text'], '  - '))
    else:
        out.append(_wrap(e['what_decides']['text']))
    out += ['', 'MONEY (ALL-IN PER MONTH)']
    out.append(_wrap(e['money']['text']))
    for b in e['money']['break_even']: out.append(_wrap(b['text'], '  - '))
    out += ['', 'WHAT WOULD CHANGE THE ANSWER']
    for x in e['what_would_change_the_answer']: out.append(_wrap(x['text'], '  - '))
    out += ['', 'AUDIT OF AI LABELS', _wrap(e['audit']['text']), '', 'WHAT THIS DOES NOT SHOW']
    for c in e['caveats']: out.append(_wrap(c, '  - '))
    text = '\n'.join(out) + '\n'
    assert all(len(l) <= 80 for l in text.splitlines())
    return text
