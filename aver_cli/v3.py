"""The bridge to the v3 referee engine (docs/103 §§2-7).

``compute_verdict_v3`` and ``RepeatedCase`` live in ``aver_policy.repeated``,
which is docs/103 T-D1's tranche, not this one. This module is the single place
that knows whether that engine is present, so that exactly one error message
exists for its absence and no verdict logic is ever reimplemented here. The CLI
would rather exit 3 saying the referee is missing than compute a second, slightly
different verdict rule — two verdict rules is the failure docs/101 §0 forbids.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType
from typing import Optional

from aver_cli.evidence import CaseDecisions
from aver_cli.policy import CliPolicy

V3_MISSING_MESSAGE = (
    "aver: this build has no v3 referee engine, so there is no verdict to give.\n"
    "      'aver check' and 'aver demo' need aver_policy.repeated.compute_verdict_v3\n"
    "      (docs/103 T-D1). 'aver import', 'aver floor' and 'aver repeat' work without it."
)


def engine_policy(policy: CliPolicy, module: ModuleType):
    """Build the engine ``Policy`` the v3 entry point takes from the CLI's policy.

    ``r`` is the v1/v2 error-cost asymmetry. Neither Clause A nor Clause B reads
    it, and the CLI's policy file has no asymmetry field, so it is set to 1.0 --
    symmetric, i.e. no claim about relative error cost. ``preference`` does read
    it and is still returned, but docs/102 §3 says preference never sets ``pass``,
    so a neutral ``r`` cannot make a verdict more permissive.
    """

    from aver_policy.engine import METHOD_VERSION_V3, GuardedAction, Policy

    return Policy(
        r=1.0,
        confidence=policy.confidence,
        method_version=METHOD_VERSION_V3,
        decidability_threshold=policy.decidability_threshold,
        max_undecided_fraction=policy.max_undecided_fraction,
        guarded_actions=[
            GuardedAction(
                action=action.name,
                tolerance=action.tolerance,
                harm_tolerance=action.harm_tolerance,
                # docs/103 §5: "not stated" and "deliberately switched off" are
                # different facts. `harm_tolerance: null` in the policy file is the
                # explicit opt-out, and only that may silence Clause B.
                harm_clause_active=action.harm_tolerance is not None,
                min_evidence=action.min_evidence,
            )
            for action in policy.actions
        ],
    )


def repeated_module() -> Optional[ModuleType]:
    """``aver_policy.repeated`` if it is present and carries the v3 entry point."""

    try:
        from aver_policy import repeated  # type: ignore[attr-defined]
    except ImportError:
        return None
    required = ("compute_verdict_v3", "RepeatedCase", "RepeatedRecord")
    if not all(hasattr(repeated, name) for name in required):
        return None
    return repeated


def available() -> bool:
    return repeated_module() is not None


def to_repeated_cases(
    cases: Sequence[CaseDecisions],
    policy: CliPolicy,
    module: ModuleType,
) -> list:
    """Adapt the CLI's ``CaseDecisions`` to the engine's ``RepeatedCase``."""

    record = module.RepeatedRecord
    case_type = module.RepeatedCase
    out = []
    for item in cases:
        out.append(case_type(
            id=item.id,
            slice=item.slice or policy.decision_point,
            action=item.action,
            incumbent=record(
                decisions=tuple(item.incumbent),
                variant="rerun",
                cost_usd=tuple(item.incumbent_cost_usd),
                latency_ms=tuple(item.incumbent_latency_ms),
            ),
            candidate=record(
                decisions=tuple(item.candidate),
                variant="rerun",
                cost_usd=tuple(item.candidate_cost_usd),
                latency_ms=tuple(item.candidate_latency_ms),
            ),
        ))
    return out
