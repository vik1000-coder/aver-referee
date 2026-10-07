"""``aver-evidence/v1`` — one JSON object per run of one task on one side.

The whole point of a flat per-run record is that the referee never sees a vendor
field. An importer's only job is to answer, for one run, "did this guarded action
fire?"; everything it could not answer is ``None`` and is reported as such,
never as a violation and never dropped (docs/100's no-silent-row-loss rule
applied to the CLI).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Optional

from aver_cli import EVIDENCE_SCHEMA

SIDE_INCUMBENT = "incumbent"
SIDE_CANDIDATE = "candidate"
SIDES: tuple[str, ...] = (SIDE_INCUMBENT, SIDE_CANDIDATE)

#: A fired action is the harmful event (a refund issued, a record deleted, an
#: email sent). This is the default because the guarded action is normally the
#: consequential one.
FIRE_IS_HARMFUL = "fire_is_harmful"
#: Withholding is the harmful event (an escalation that must happen, a
#: confirmation that must be asked).
WITHHOLD_IS_HARMFUL = "withhold_is_harmful"
DIRECTIONS: tuple[str, ...] = (FIRE_IS_HARMFUL, WITHHOLD_IS_HARMFUL)

DEFAULT_VARIANT = "rerun"


class EvidenceError(ValueError):
    """A line that cannot be read as ``aver-evidence/v1``."""


def _optional_bool(value: object) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in ("true", "1", "yes", "valid"):
            return True
        if token in ("false", "0", "no", "invalid"):
            return False
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    return None


@dataclass(frozen=True)
class ActionFacts:
    """What one run did with one guarded action.

    ``args_schema_valid`` is ``None`` when the export did not say, which is never
    treated as a violation (``ToolCall.schema_valid`` semantics).
    """

    fired: bool
    count: int = 0
    args_schema_valid: Optional[bool] = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ActionFacts":
        fired = _optional_bool(raw.get("fired"))
        if fired is None:
            raise EvidenceError("facts entry has no readable 'fired' value")
        count = raw.get("count")
        try:
            count_int = int(count) if count is not None else (1 if fired else 0)
        except (TypeError, ValueError) as exc:
            raise EvidenceError(f"facts entry has a non-integer 'count': {count!r}") from exc
        return cls(
            fired=fired,
            count=max(0, count_int),
            args_schema_valid=_optional_bool(raw.get("args_schema_valid")),
        )

    def to_dict(self) -> dict:
        return {
            "fired": self.fired,
            "count": self.count,
            "args_schema_valid": self.args_schema_valid,
        }


@dataclass(frozen=True)
class EvidenceLine:
    """One run of one task on one side."""

    task_id: str
    side: str
    run: int
    facts: Mapping[str, ActionFacts] = field(default_factory=dict)
    variant: str = DEFAULT_VARIANT
    unreadable: bool = False
    meta: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "EvidenceLine":
        schema = raw.get("schema")
        if schema != EVIDENCE_SCHEMA:
            raise EvidenceError(f"schema is {schema!r}, not {EVIDENCE_SCHEMA!r}")
        task_id = str(raw.get("task_id") or "")
        if not task_id:
            raise EvidenceError("task_id is missing or empty")
        side = str(raw.get("side") or "")
        if side not in SIDES:
            raise EvidenceError(f"side is {side!r}, not one of {SIDES}")
        run_raw = raw.get("run")
        try:
            run = int(run_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise EvidenceError(f"run is not an integer: {run_raw!r}") from exc
        if run < 0:
            raise EvidenceError(f"run is negative: {run}")
        facts_raw = raw.get("facts") or {}
        if not isinstance(facts_raw, Mapping):
            raise EvidenceError("facts is not an object")
        facts = {str(name): ActionFacts.from_dict(entry)
                 for name, entry in facts_raw.items()
                 if isinstance(entry, Mapping)}
        meta = raw.get("meta") or {}
        return cls(
            task_id=task_id,
            side=side,
            run=run,
            facts=facts,
            variant=str(raw.get("variant") or DEFAULT_VARIANT),
            unreadable=bool(raw.get("unreadable")),
            meta=dict(meta) if isinstance(meta, Mapping) else {"value": meta},
        )

    def to_dict(self) -> dict:
        return {
            "schema": EVIDENCE_SCHEMA,
            "task_id": self.task_id,
            "side": self.side,
            "run": self.run,
            "variant": self.variant,
            "facts": {name: facts.to_dict() for name, facts in sorted(self.facts.items())},
            "unreadable": self.unreadable,
            "meta": dict(self.meta),
        }

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json_line(cls, line: str) -> "EvidenceLine":
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvidenceError(f"not JSON: {exc.msg}") from exc
        if not isinstance(raw, Mapping):
            raise EvidenceError("line is not a JSON object")
        return cls.from_dict(raw)

    def decision(self, action: str, direction: str) -> Optional[int]:
        """The engine-convention decision for one action: 1 = the safe side.

        ``None`` means this run said nothing readable about this action — the run
        was unreadable, or the export carried no entry for it. ``None`` is
        preserved all the way into the engine, which drops it from that task's
        ``k`` rather than counting it either way (docs/103 §2).
        """

        if self.unreadable:
            return None
        facts = self.facts.get(action)
        if facts is None:
            return None
        if direction == WITHHOLD_IS_HARMFUL:
            return 1 if facts.fired else 0
        return 0 if facts.fired else 1


@dataclass(frozen=True)
class CaseDecisions:
    """One ``(task, guarded action)`` pair, both sides, runs in run order.

    This is the CLI's own carrier for what docs/103 §2 calls a ``RepeatedCase``.
    It exists so the importers, ``floor`` and the fixture builder do not depend
    on the v3 engine module, and so the adapter to it is one function.
    """

    id: str
    slice: str
    action: str
    direction: str
    incumbent: tuple[Optional[int], ...] = ()
    candidate: tuple[Optional[int], ...] = ()
    incumbent_cost_usd: tuple[Optional[float], ...] = ()
    candidate_cost_usd: tuple[Optional[float], ...] = ()
    incumbent_latency_ms: tuple[Optional[int], ...] = ()
    candidate_latency_ms: tuple[Optional[int], ...] = ()

    def side(self, side: str) -> tuple[Optional[int], ...]:
        return self.incumbent if side == SIDE_INCUMBENT else self.candidate


def _meta_float(meta: Mapping[str, object], key: str) -> Optional[float]:
    value = meta.get(key)
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _meta_int(meta: Mapping[str, object], key: str) -> Optional[int]:
    value = _meta_float(meta, key)
    return None if value is None else int(value)


@dataclass
class GroupReport:
    """What grouping saw, so nothing is lost between the file and the engine."""

    tasks: dict[str, int] = field(default_factory=dict)
    runs_per_task: dict[str, dict[str, int]] = field(default_factory=dict)
    unreadable_runs: int = 0
    duplicate_runs: int = 0
    actions_seen: set[str] = field(default_factory=set)
    #: readable runs that carried no entry for a requested action
    missing_action_runs: dict[str, int] = field(default_factory=dict)
    tasks_one_side_only: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "tasks": dict(sorted(self.tasks.items())),
            "unreadable_runs": self.unreadable_runs,
            "duplicate_runs": self.duplicate_runs,
            "actions_seen": sorted(self.actions_seen),
            "missing_action_runs": dict(sorted(self.missing_action_runs.items())),
            "tasks_one_side_only": sorted(self.tasks_one_side_only),
            "runs_per_task_histogram": {
                side: dict(sorted(hist.items(), key=lambda kv: int(kv[0])))
                for side, hist in sorted(self.runs_per_task.items())
            },
        }


def group_cases(
    lines: Iterable[EvidenceLine],
    *,
    actions: Sequence[tuple[str, str]],
    decision_point: str = "",
) -> tuple[list[CaseDecisions], GroupReport]:
    """Group per-run lines into one case per ``(task, action)``.

    ``actions`` is a sequence of ``(name, direction)`` pairs, normally the policy's
    action list. A task present on only one side is kept in the report and
    excluded from the cases, because a paired clause has nothing to compare it
    against and silently dropping it is how a denominator lies.
    """

    report = GroupReport()
    by_side: dict[str, dict[str, dict[int, EvidenceLine]]] = {side: {} for side in SIDES}
    for line in lines:
        runs = by_side[line.side].setdefault(line.task_id, {})
        if line.run in runs:
            report.duplicate_runs += 1
            continue
        runs[line.run] = line
        if line.unreadable:
            report.unreadable_runs += 1
        report.actions_seen.update(line.facts.keys())

    for side in SIDES:
        report.tasks[side] = len(by_side[side])
        hist: dict[str, int] = {}
        for runs in by_side[side].values():
            key = str(len(runs))
            hist[key] = hist.get(key, 0) + 1
        report.runs_per_task[side] = hist

    inc_tasks, cand_tasks = by_side[SIDE_INCUMBENT], by_side[SIDE_CANDIDATE]
    report.tasks_one_side_only = sorted(set(inc_tasks) ^ set(cand_tasks))
    paired = sorted(set(inc_tasks) & set(cand_tasks))

    cases: list[CaseDecisions] = []
    for name, direction in actions:
        missing = 0
        for task_id in paired:
            per_side: dict[str, list[EvidenceLine]] = {}
            for side, tasks in ((SIDE_INCUMBENT, inc_tasks), (SIDE_CANDIDATE, cand_tasks)):
                per_side[side] = [tasks[task_id][run] for run in sorted(tasks[task_id])]
            decisions = {
                side: tuple(line.decision(name, direction) for line in per_side[side])
                for side in SIDES
            }
            for side in SIDES:
                for line, value in zip(per_side[side], decisions[side]):
                    if value is None and not line.unreadable:
                        missing += 1
            cases.append(
                CaseDecisions(
                    id=task_id,
                    slice=decision_point or name,
                    action=name,
                    direction=direction,
                    incumbent=decisions[SIDE_INCUMBENT],
                    candidate=decisions[SIDE_CANDIDATE],
                    incumbent_cost_usd=tuple(
                        _meta_float(line.meta, "cost_usd") for line in per_side[SIDE_INCUMBENT]
                    ),
                    candidate_cost_usd=tuple(
                        _meta_float(line.meta, "cost_usd") for line in per_side[SIDE_CANDIDATE]
                    ),
                    incumbent_latency_ms=tuple(
                        _meta_int(line.meta, "latency_ms") for line in per_side[SIDE_INCUMBENT]
                    ),
                    candidate_latency_ms=tuple(
                        _meta_int(line.meta, "latency_ms") for line in per_side[SIDE_CANDIDATE]
                    ),
                )
            )
        if missing:
            report.missing_action_runs[name] = missing
    return cases, report


def side_cases(
    lines: Iterable[EvidenceLine],
    *,
    side: str,
    actions: Sequence[tuple[str, str]],
) -> list[CaseDecisions]:
    """One case per ``(task, action)`` for every task on ONE side, paired or not.

    ``floor`` is a fact about one agent, so it must not need the other side: a
    team that has only run ``aver repeat`` on its current agent has a floor to
    read. The other side's tuple is left empty. Duplicate run indices are skipped
    exactly as ``group_cases`` skips them.
    """

    tasks: dict[str, dict[int, EvidenceLine]] = {}
    for line in lines:
        if line.side != side:
            continue
        runs = tasks.setdefault(line.task_id, {})
        if line.run not in runs:
            runs[line.run] = line
    cases: list[CaseDecisions] = []
    for name, direction in actions:
        for task_id in sorted(tasks):
            ordered = [tasks[task_id][run] for run in sorted(tasks[task_id])]
            decisions = tuple(line.decision(name, direction) for line in ordered)
            cases.append(CaseDecisions(
                id=task_id,
                slice=name,
                action=name,
                direction=direction,
                incumbent=decisions if side == SIDE_INCUMBENT else (),
                candidate=decisions if side == SIDE_CANDIDATE else (),
            ))
    return cases


def sides_present(lines: Iterable[EvidenceLine]) -> list[str]:
    seen = {line.side for line in lines}
    return [side for side in SIDES if side in seen]


@dataclass
class ReadReport:
    """Every line the reader saw, and why any of them was not kept."""

    files: list[str] = field(default_factory=list)
    total_lines: int = 0
    blank_lines: int = 0
    kept: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    first_reasons: list[str] = field(default_factory=list)

    def skip(self, reason: str, where: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1
        if len(self.first_reasons) < 10:
            self.first_reasons.append(f"{where}: {reason}")

    @property
    def skipped_total(self) -> int:
        return sum(self.skipped.values())

    def to_dict(self) -> dict:
        return {
            "files": list(self.files),
            "total_lines": self.total_lines,
            "blank_lines": self.blank_lines,
            "kept": self.kept,
            "skipped_total": self.skipped_total,
            "skipped": dict(sorted(self.skipped.items())),
            "examples": list(self.first_reasons),
        }


def read_evidence_text(text: str, *, where: str = "-", report: Optional[ReadReport] = None
                       ) -> tuple[list[EvidenceLine], ReadReport]:
    """Read ``aver-evidence/v1`` JSONL. Every skipped line is counted and named."""

    report = report or ReadReport()
    lines: list[EvidenceLine] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        report.total_lines += 1
        if not raw.strip():
            report.blank_lines += 1
            continue
        try:
            lines.append(EvidenceLine.from_json_line(raw))
        except EvidenceError as exc:
            report.skip(str(exc), f"{where}:{number}")
            continue
        report.kept += 1
    return lines, report


def write_evidence(lines: Iterable[EvidenceLine]) -> str:
    """Serialise in a stable order so ``evidence_digest`` is reproducible."""

    ordered = sorted(lines, key=lambda line: (line.task_id, line.side, line.run, line.variant))
    return "".join(line.to_json_line() + "\n" for line in ordered)


def evidence_digest(lines: Iterable[EvidenceLine]) -> str:
    """sha256 over the sorted input lines (docs/104 §4.3)."""

    payload = sorted(line.to_json_line() for line in lines)
    digest = hashlib.sha256()
    for item in payload:
        digest.update(item.encode("utf-8"))
        digest.update(b"\n")
    return "sha256:" + digest.hexdigest()


def task_id_from_message(message: object) -> str:
    """sha256 of a canonicalised first user message (docs/104 §5.6, last resort)."""

    canonical = json.dumps(message, sort_keys=True, separators=(",", ":"), default=str)
    return "task_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
