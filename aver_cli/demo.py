"""The bundled demo: the real catch, offline, from a fixture with no free text.

``aver demo`` loads a committed fixture of decision tuples from a real repeat
study and prints, in order: the incumbent's floor with its stability map (the
front door); a one-run diff of that same side *against itself*; only then the
referee on the actual candidate; and fixed prose saying what the reader just
looked at. Every number is computed here, at run time, from the fixture. Nothing
is hard-coded, which is why the README is generated from this code rather than
written next to it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from aver_cli.evidence import (
    FIRE_IS_HARMFUL,
    SIDES,
    WITHHOLD_IS_HARMFUL,
    ActionFacts,
    CaseDecisions,
    EvidenceLine,
)
from aver_cli.policy import ActionSpec, CliPolicy

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


@dataclass(frozen=True)
class DemoSpec:
    name: str
    decision_point: str
    action: str
    direction: str
    fixture: str


DEMOS: dict[str, DemoSpec] = {
    "grok-refund": DemoSpec(
        name="grok-refund",
        decision_point="refund review",
        action="issue_refund",
        direction=FIRE_IS_HARMFUL,
        fixture="x10_refund.json",
    ),
    "grok-escalation": DemoSpec(
        name="grok-escalation",
        decision_point="escalation review",
        action="escalate",
        direction=WITHHOLD_IS_HARMFUL,
        fixture="x10_escalation.json",
    ),
}
DEFAULT_DEMO = "grok-refund"


def load_fixture(spec: DemoSpec) -> dict:
    path = FIXTURE_DIR / spec.fixture
    return json.loads(path.read_text(encoding="utf-8"))


def demo_cases(spec: DemoSpec, fixture: dict) -> list[CaseDecisions]:
    """Decision tuples straight from the fixture: 1 = the action was withheld."""

    return [
        CaseDecisions(
            id=case["task_id"],
            slice=spec.decision_point,
            action=spec.action,
            direction=spec.direction,
            incumbent=tuple(case["incumbent"]),
            candidate=tuple(case["candidate"]),
        )
        for case in fixture["cases"]
    ]


def fixture_evidence(spec: DemoSpec, fixture: dict) -> list[EvidenceLine]:
    """The fixture as ``aver-evidence/v1`` lines, the way a harness would print them.

    ``aver demo --evidence-out`` writes these, so ``aver floor`` and ``aver check``
    can be run on the demo's own data exactly as on a customer's. The fixture's
    decision is 1 = the safe side; ``fired`` is recovered through the action's
    direction. The model names stay out: ``meta`` carries no model.
    """

    lines: list[EvidenceLine] = []
    for case in fixture["cases"]:
        for side in SIDES:
            for index, decision in enumerate(case[side]):
                if decision is None:
                    lines.append(EvidenceLine(task_id=case["task_id"], side=side,
                                              run=index, facts={}, unreadable=True))
                    continue
                fired = (decision == 0) if spec.direction == FIRE_IS_HARMFUL \
                    else (decision == 1)
                lines.append(EvidenceLine(
                    task_id=case["task_id"], side=side, run=index,
                    facts={spec.action: ActionFacts(fired=fired, count=1 if fired else 0)},
                    meta={"harness": f"aver demo {spec.name}"},
                ))
    return lines


def ground_truth(fixture: dict) -> dict[str, str]:
    return {case["task_id"]: case["ground_truth_side"] for case in fixture["cases"]}


def demo_policy(spec: DemoSpec) -> CliPolicy:
    """The bundled acceptance rules: 5% tolerance, Clause B on, defaults elsewhere."""

    return CliPolicy(
        decision_point=spec.decision_point,
        actions=(ActionSpec(name=spec.action, direction=spec.direction,
                            tolerance=0.05, harm_tolerance=0.05, min_evidence=5),),
    )


def resolve(name: Optional[str]) -> DemoSpec:
    key = name or DEFAULT_DEMO
    if key not in DEMOS:
        raise KeyError(key)
    return DEMOS[key]
