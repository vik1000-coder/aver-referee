"""Unsigned economic handover. Signed hosted records intentionally contain no prices or savings."""
from __future__ import annotations

import math

from aver_policy.decision import need
from aver_policy import decision_v2 as v2


def decision_memo(result, monthly_fee_usd=0.):
    need(type(monthly_fee_usd) in (int, float) and math.isfinite(monthly_fee_usd) and monthly_fee_usd >= 0,
         'invalid_fee', 'monthly_fee_usd', 'Use a finite monthly fee of zero or more.')
    result = v2.verify_result(result)
    plan, report = result['frozen_plan']['plan'], result['report']
    spec = v2.spec_to_v2(plan['exploration']['spec'], plan)
    costs = [r['cost_usd'] for r in result['evidence']]
    audit = spec.get('grader_audit') or {}
    checks = audit.get('sampled_fine', 0) + audit.get('sampled_flagged', 0)
    spend = None if spec['label_cost_usd'] is None or any(x is None for x in costs) else v2.sequential_sum(costs) + (len(costs) + checks) * spec['label_cost_usd']
    saving = report['savings']
    lower = saving.get('lower_usd_per_month') if report['stopped'] == 'pass' else None
    shown = lower is not None and lower > 0
    net_monthly = None if lower is None else lower - monthly_fee_usd
    return {'schema': 'aver-decision-memo/v1', 'authority': 'unsigned_local_economic_memo',
            'plan_digest': result['frozen_plan']['plan_digest'], 'evidence_digest': report['lineage']['evidence_digest'],
            'scope': spec['scope'], 'state': report['stopped'], 'choice': report['choice'],
            'monthly_uses': spec['monthly_uses'], 'prices_and_limits': spec.get('taxonomy'), 'requirements': spec['requirements'],
            'savings': saving, 'positive_saving_shown': shown,
            'comparison_basis': 'same_cases' if plan['paired'] else 'current_logged_rate',
            'checking_spend_usd': spend, 'monthly_fee_usd': monthly_fee_usd, 'fees_included': monthly_fee_usd > 0,
            'monthly_saving_lower_after_fee_usd': net_monthly,
            'months_to_recover_checking_at_lower_saving': None if spend is None or net_monthly is None or net_monthly <= 0 or not shown else spend / net_monthly,
            'limitations': ['An economic memo is not a signed record or release authorization.',
                            'Costs use logged running spend and your per-case checking price; unlogged setup work is excluded.',
                            'Savings assume the frozen traffic mix, prices, running costs and label-error assumptions hold.',
                            'With current_logged_rate, today\'s error rate is from the pilot and is not bounded by this test.',
                            'A zero fee excludes subscription fees; allocate your actual monthly fee before deciding paid value.']}
