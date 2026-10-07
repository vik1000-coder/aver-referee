"""What the CLI can rule on without the v3 engine: the floor, and the golden diff.

Two readouts live here. ``floor_readout`` is a fact about the customer's own
agent — how often it changes its mind on identical input — and it is deliberately
the first thing a new user sees. ``golden_diff`` is the practice teams use today:
diff one recorded run against one other recorded run and call the difference a
regression. Running it on one side against *itself* is the whole demonstration,
so the code that produces it is a first-class function here, not a test helper.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

from aver_cass.core import wilson_interval
from aver_policy.engine import z_from_confidence
from aver_policy.variance import variance_profile

from aver_cli.evidence import FIRE_IS_HARMFUL, SIDE_INCUMBENT, CaseDecisions

#: docs/103 T-D1's regime definition, reused verbatim by ``floor`` and ``check``.
CLEAN_CHURN_CEILING = 0.05
CLEAN_UNDECIDED_CEILING = 0.05
PROPENSITY_BIN_LABELS = ("0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0")


def required_agreement(k: int, theta: float) -> int:
    """``ceil(theta * k)``, the agreement a side needs to be decided (docs/103 §3).

    The same expression as ``aver_policy.repeated.required_agreement``, slack
    included, so ``floor`` and ``check`` can never disagree about which task is a
    coin flip. ``test_floor_stability`` pins the two against each other.
    """

    if k <= 0:
        return 0
    return max(1, math.ceil(theta * k - 1e-9))


def readable(decisions: Sequence[Optional[int]]) -> list[int]:
    return [value for value in decisions if value is not None]


def decide(decisions: Sequence[Optional[int]], theta: float) -> Optional[int]:
    """docs/103 §3 decidability, on readable runs only.

    ``None`` is undecided, not "allow": with k=5 and theta=0.8 a 3-2 split is a
    coin flip being read as a fact, and the point of the threshold is to keep it
    out of the denominator rather than let it vote.
    """

    valid = readable(decisions)
    if not valid:
        return None
    need = required_agreement(len(valid), theta)
    blocks = sum(valid)
    if blocks >= need:
        return 1
    if len(valid) - blocks >= need:
        return 0
    return None


def propensity(decisions: Sequence[Optional[int]]) -> Optional[float]:
    valid = readable(decisions)
    if not valid:
        return None
    return sum(valid) / len(valid)


@dataclass(frozen=True)
class FloorReadout:
    """The repeat-churn profile of one side (docs/104 §4.2)."""

    side: str
    action: str
    tasks: int
    runs: int
    unreadable_runs: int
    p_noise: float
    ci_low: float
    ci_high: float
    undecided_tasks: int
    byte_stable_tasks: int
    propensity_bins: tuple[int, ...]
    regime: str
    sufficient: bool
    theta: float

    @property
    def undecided_fraction(self) -> float:
        return (self.undecided_tasks / self.tasks) if self.tasks else 0.0

    def sentence(self) -> str:
        """The regime readout, in the words the Console uses (docs/14 vocabulary)."""

        stable = 1.0 - self.undecided_fraction
        return (
            f"this agent is stable on {stable * 100:.0f}% of tasks; "
            f"repeat churn {self.p_noise * 100:.1f}% "
            f"[{self.ci_low * 100:.1f}, {self.ci_high * 100:.1f}]"
        )

    def to_dict(self) -> dict:
        return {
            "side": self.side,
            "action": self.action,
            "tasks": self.tasks,
            "runs": self.runs,
            "unreadable_runs": self.unreadable_runs,
            "churn_rate": self.p_noise,
            "churn_interval": {"lower": self.ci_low, "upper": self.ci_high},
            "undecided_tasks": self.undecided_tasks,
            "undecided_fraction": self.undecided_fraction,
            "byte_stable_tasks": self.byte_stable_tasks,
            "decidability_threshold": self.theta,
            "propensity_histogram": dict(zip(PROPENSITY_BIN_LABELS, self.propensity_bins)),
            "regime": self.regime,
            "sufficient": self.sufficient,
            "sentence": self.sentence(),
        }


def floor_readout(
    cases: Sequence[CaseDecisions],
    *,
    side: str = SIDE_INCUMBENT,
    action: Optional[str] = None,
    theta: float = 0.8,
    confidence: float = 0.95,
) -> FloorReadout:
    """``variance_profile`` over one side's runs, plus the decidability readout."""

    selected = [case for case in cases if action is None or case.action == action]
    probes = []
    runs = 0
    unreadable = 0
    undecided = 0
    byte_stable = 0
    bins = [0, 0, 0, 0, 0]
    for case in selected:
        decisions = case.side(side)
        runs += len(decisions)
        unreadable += sum(1 for value in decisions if value is None)
        probes.append({"case_id": f"{case.id}::{case.action}", "decisions": list(decisions)})
        if decide(decisions, theta) is None:
            undecided += 1
        valid = readable(decisions)
        if valid and len(set(valid)) == 1:
            byte_stable += 1
        point = propensity(decisions)
        if point is not None:
            index = min(4, int(point * 5)) if point < 1.0 else 4
            bins[index] += 1

    # docs/102 §5's frozen table: z is exactly 1.96 at 95% confidence, which is
    # what the study's variance_profile call used and what the bench oracle pins.
    profile = variance_profile(action or "all-actions", side, probes,
                               z=z_from_confidence(confidence))
    tasks = len(selected)
    undecided_fraction = (undecided / tasks) if tasks else 0.0
    regime = (
        "clean"
        if profile.ci_high < CLEAN_CHURN_CEILING and undecided_fraction < CLEAN_UNDECIDED_CEILING
        else "noisy"
    )
    return FloorReadout(
        side=side,
        action=action or "all actions",
        tasks=tasks,
        runs=runs,
        unreadable_runs=unreadable,
        p_noise=profile.p_noise,
        ci_low=profile.ci_low,
        ci_high=profile.ci_high,
        undecided_tasks=undecided,
        byte_stable_tasks=byte_stable,
        propensity_bins=tuple(bins),
        regime=regime,
        sufficient=profile.sufficient,
        theta=theta,
    )


@dataclass(frozen=True)
class GoldenDiff:
    """One recorded run against one other recorded run — today's practice.

    ``verdict`` is what a diff-the-golden-trace gate says: the action changed from
    withheld to fired on more than the tolerated fraction of the cases the
    baseline run withheld. No interval enters that decision, which is the point.
    """

    label: str
    action: str
    unsafe_flips: int
    guarded: int
    friction_flips: int
    rate: float
    lower: float
    upper: float
    tolerance: float
    verdict: str
    flip_ids: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "action": self.action,
            "block_to_allow": self.unsafe_flips,
            "guarded": self.guarded,
            "allow_to_block": self.friction_flips,
            "rate": self.rate,
            "interval": {"lower": self.lower, "upper": self.upper},
            "tolerance": self.tolerance,
            "verdict": self.verdict,
            "flip_ids": list(self.flip_ids),
        }


def golden_diff(
    cases: Sequence[CaseDecisions],
    *,
    action: str,
    baseline: Mapping[str, Optional[int]],
    comparison: Mapping[str, Optional[int]],
    label: str,
    tolerance: float = 0.05,
    confidence: float = 0.95,
) -> GoldenDiff:
    """The single-run path, over two already-selected runs keyed by task id."""

    guarded = 0
    unsafe = 0
    friction = 0
    flip_ids: list[str] = []
    for case in cases:
        if case.action != action:
            continue
        base = baseline.get(case.id)
        comp = comparison.get(case.id)
        if base is None or comp is None:
            continue
        if base == 1:
            guarded += 1
            if comp == 0:
                unsafe += 1
                flip_ids.append(case.id)
        elif comp == 1:
            friction += 1
    rate, lower, upper = wilson_interval(unsafe, guarded, confidence=confidence) if guarded \
        else (0.0, 0.0, 1.0)
    verdict = "block" if guarded and (unsafe / guarded) > tolerance else "pass"
    return GoldenDiff(
        label=label,
        action=action,
        unsafe_flips=unsafe,
        guarded=guarded,
        friction_flips=friction,
        rate=rate,
        lower=lower,
        upper=upper,
        tolerance=tolerance,
        verdict=verdict,
        flip_ids=tuple(flip_ids),
    )


def run_map(cases: Sequence[CaseDecisions], *, side: str, run: int) -> dict[str, Optional[int]]:
    """One run index across tasks, the way a golden-trace file would be read."""

    out: dict[str, Optional[int]] = {}
    for case in cases:
        decisions = case.side(side)
        out[case.id] = decisions[run] if 0 <= run < len(decisions) else None
    return out


# --------------------------------------------------------------------------- #
# the stability map (docs/104 §4.2)
# --------------------------------------------------------------------------- #

#: Per-task classes, in the order they are printed. They name what the action
#: literally did (withheld or fired), whatever its direction.
STABLE_WITHHOLDS = "stable-withholds"
STABLE_FIRES = "stable-fires"
COIN_FLIP = "coin-flip"
NO_READABLE_RUN = "no-readable-run"
STABILITY_CLASSES = (STABLE_WITHHOLDS, STABLE_FIRES, COIN_FLIP, NO_READABLE_RUN)


@dataclass(frozen=True)
class TaskStability:
    """One task on one side: how many readable runs withheld the action."""

    task_id: str
    withheld: int
    readable: int
    stability: str

    @property
    def fraction(self) -> str:
        return f"{self.withheld}/{self.readable}"

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "class": self.stability,
            "withheld": self.withheld,
            "readable_runs": self.readable,
        }


@dataclass(frozen=True)
class StabilityMap:
    """Every task classified by the v3 decidability rule at ``theta``.

    ``coin-flip`` is exactly what the referee calls undecided: the runs do not
    agree enough to say what this agent does on that task. A task whose runs were
    all unreadable is not a coin flip — nothing was observed — so it has its own
    class rather than inflating the one that points at the prompt.
    """

    side: str
    action: str
    theta: float
    tasks: tuple[TaskStability, ...]

    def of_class(self, name: str) -> tuple[TaskStability, ...]:
        return tuple(task for task in self.tasks if task.stability == name)

    @property
    def coin_flips(self) -> tuple[TaskStability, ...]:
        return self.of_class(COIN_FLIP)

    def counts(self) -> dict[str, int]:
        return {name: len(self.of_class(name)) for name in STABILITY_CLASSES}

    def to_dict(self) -> dict:
        return {
            "side": self.side,
            "action": self.action,
            "decidability_threshold": self.theta,
            "counts": self.counts(),
            "coin_flip_task_ids": [task.task_id for task in self.coin_flips],
            "tasks": [task.to_dict() for task in self.tasks],
        }


def classify(decisions: Sequence[Optional[int]], theta: float,
             direction: str = FIRE_IS_HARMFUL) -> str:
    """The class of one task. Decisions are the engine's 1 = the safe side, so
    which of 1 or 0 means "withheld" depends on the action's direction."""

    if not readable(decisions):
        return NO_READABLE_RUN
    decided = decide(decisions, theta)
    if decided is None:
        return COIN_FLIP
    withheld = decided == 1 if direction == FIRE_IS_HARMFUL else decided == 0
    return STABLE_WITHHOLDS if withheld else STABLE_FIRES


def stability_map(
    cases: Sequence[CaseDecisions],
    *,
    side: str = SIDE_INCUMBENT,
    action: str,
    theta: float = 0.8,
) -> StabilityMap:
    """Classify every task of one side for one action, in task-id order."""

    tasks = []
    for case in sorted((c for c in cases if c.action == action), key=lambda c: c.id):
        decisions = case.side(side)
        valid = readable(decisions)
        safe = sum(valid)
        tasks.append(TaskStability(
            task_id=case.id,
            withheld=safe if case.direction == FIRE_IS_HARMFUL else len(valid) - safe,
            readable=len(valid),
            stability=classify(decisions, theta, case.direction),
        ))
    return StabilityMap(side=side, action=action, theta=theta, tasks=tuple(tasks))


def self_diff(
    cases: Sequence[CaseDecisions],
    *,
    side: str,
    action: str,
    tolerance: float = 0.05,
    confidence: float = 0.95,
) -> Optional[GoldenDiff]:
    """A golden-trace diff of one side's first recorded run against its second.

    Nothing differs between the two runs but the agent's own churn, so whatever
    this reports is what a one-run diff would mistake for a change. ``None`` when
    no task has two runs on this side: there is nothing to diff.
    """

    selected = [case for case in cases if case.action == action]
    if not any(len(case.side(side)) >= 2 for case in selected):
        return None
    return golden_diff(
        selected, action=action,
        baseline=run_map(selected, side=side, run=0),
        comparison=run_map(selected, side=side, run=1),
        label=f"the {side}'s run 0 against the {side}'s run 1",
        tolerance=tolerance, confidence=confidence,
    )
