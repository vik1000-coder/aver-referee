"""Generic JSONL (docs/104 §5.5).

Either already ``aver-evidence/v1``, or docs/86's "generic paired record"
(``example_id``, ``model_id``, ``decision``, optional ``run``), which maps 1:1
once ``decision`` is inverted per the action's ``direction`` — the engine
convention is 1 = the safe side, so a generic ``decision`` of 1 means the action
was withheld.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

from aver_cli import EVIDENCE_SCHEMA
from aver_cli.evidence import ActionFacts, EvidenceError, EvidenceLine
from aver_cli.importers.base import ImportContext, ImportResult, stable_task_id

TASK_KEYS = ("task_id", "example_id", "case_id", "id")
DECISION_KEYS = ("decision", "block", "blocked")
RUN_KEYS = ("run", "rep", "repetition")


def _rows(payload: object) -> list[dict]:
    if isinstance(payload, Mapping):
        for key in ("rows", "records", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, Mapping)]
        return [dict(payload)]
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    return []


def detect(payload: object) -> bool:
    rows = _rows(payload)
    if not rows:
        return False
    for row in rows:
        if row.get("schema") == EVIDENCE_SCHEMA:
            return True
        if any(key in row for key in TASK_KEYS) and any(key in row for key in DECISION_KEYS):
            return True
    return False


def _decision(row: Mapping) -> Optional[int]:
    for key in DECISION_KEYS:
        if key in row and row[key] is not None:
            value = row[key]
            if isinstance(value, bool):
                return 1 if value else 0
            try:
                return 1 if int(value) == 1 else 0
            except (TypeError, ValueError):
                return None
    return None


def read(payload: object, context: ImportContext) -> ImportResult:
    result = ImportResult(detected_format="jsonl")
    rows = _rows(payload)
    if not rows:
        result.skip("file contained no records")
        return result

    order: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        if row.get("schema") == EVIDENCE_SCHEMA:
            try:
                line = EvidenceLine.from_dict(row)
            except EvidenceError as exc:
                result.skip(str(exc), f"records[{index}]")
                continue
            result.actions_seen.update(line.facts.keys())
            result.lines.append(line)
            continue

        task_id = stable_task_id(
            explicit=next((row[key] for key in TASK_KEYS if row.get(key)), None),
            document=row,
            context=context,
            message=row.get("input"),
        )
        if not task_id:
            result.skip("record carried no task identity", f"records[{index}]")
            continue
        side = context.side_for(attribute=row.get("side"), model=row.get("model_id", row.get("model")))

        valid = row.get("valid")
        decision = _decision(row)
        unreadable = decision is None or valid is False or bool(row.get("provider_error"))

        run_raw = next((row[key] for key in RUN_KEYS if row.get(key) is not None), None)
        try:
            run = int(run_raw) if run_raw is not None else order.get((task_id, side), 0)
        except (TypeError, ValueError):
            run = order.get((task_id, side), 0)
        order[(task_id, side)] = max(order.get((task_id, side), 0), run) + 1

        facts: dict[str, ActionFacts] = {}
        if not unreadable:
            for name in context.actions:
                action = context.direction_of(name)
                # decision 1 = the safe side. Invert back to "did it fire?".
                fired = (decision == 1) if action == "withhold_is_harmful" else (decision == 0)
                facts[name] = ActionFacts(fired=fired, count=1 if fired else 0)
                result.actions_seen.add(name)

        result.lines.append(EvidenceLine(
            task_id=task_id,
            side=side,
            run=run,
            facts=facts,
            unreadable=unreadable,
            meta={key: value for key, value in (
                ("model", row.get("model_id", row.get("model"))),
                ("harness", context.harness or "jsonl"),
            ) if value is not None},
        ))
    return result
