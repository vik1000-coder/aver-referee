"""LangSmith run export (docs/104 §5.2): JSONL of runs.

A trajectory is a root run and its descendants. Tool invocations are runs with
``run_type == "tool"``; the action *fired* when the tool run exists with
``error == null``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

from aver_cli.evidence import EvidenceLine
from aver_cli.importers.base import (
    ImportContext,
    ImportResult,
    facts_for,
    first_user_message,
    stable_task_id,
)


def _runs(payload: object) -> list[dict]:
    if isinstance(payload, Mapping):
        for key in ("runs", "data", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, Mapping)]
        return [dict(payload)]
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    return []


def detect(payload: object) -> bool:
    runs = _runs(payload)
    if not runs:
        return False
    return any("run_type" in run for run in runs)


def _metadata(run: Mapping) -> dict:
    meta = run.get("metadata")
    if not isinstance(meta, Mapping):
        extra = run.get("extra")
        meta = extra.get("metadata") if isinstance(extra, Mapping) else None
    return dict(meta) if isinstance(meta, Mapping) else {}


def _model(run: Mapping) -> Optional[str]:
    meta = _metadata(run)
    for key in ("model", "ls_model_name", "model_name"):
        if meta.get(key):
            return str(meta[key])
    extra = run.get("extra")
    if isinstance(extra, Mapping):
        inner = extra.get("metadata")
        if isinstance(inner, Mapping) and inner.get("model"):
            return str(inner["model"])
    return None


def read(payload: object, context: ImportContext) -> ImportResult:
    result = ImportResult(detected_format="langsmith")
    runs = _runs(payload)
    if not runs:
        result.skip("file contained no runs")
        return result

    by_id = {str(run.get("id")): run for run in runs if run.get("id") is not None}

    def root_of(run_id: str) -> str:
        seen: set[str] = set()
        current = run_id
        while current and current not in seen:
            seen.add(current)
            parent = by_id.get(current, {}).get("parent_run_id")
            parent = str(parent) if parent is not None else ""
            if not parent or parent not in by_id:
                return current
            current = parent
        return current

    children: dict[str, list[str]] = {}
    roots: dict[str, dict] = {}
    for run_id in by_id:
        root = root_of(run_id)
        roots.setdefault(root, by_id[root])
        if run_id != root:
            children.setdefault(root, []).append(run_id)

    trajectories = []
    for root_id, root in roots.items():
        meta = _metadata(root)
        task_id = stable_task_id(
            explicit=meta.get("aver_task_id"),
            document=root,
            context=context,
            message=first_user_message(root.get("inputs")),
        )
        if not task_id:
            result.skip("root run carried no task identity and no readable inputs",
                        f"run {root_id}")
            continue
        side = context.side_for(attribute=meta.get("aver_side"), model=_model(root))
        trajectories.append((task_id, side, root_id, root, meta))

    order: dict[tuple[str, str], int] = {}
    trajectories.sort(key=lambda item: (item[0], item[1],
                                        str(item[3].get("start_time") or ""), item[2]))
    for task_id, side, root_id, root, meta in trajectories:
        fired: dict[str, int] = {}
        schema_valid: dict[str, Optional[bool]] = {}
        for child_id in [root_id] + children.get(root_id, []):
            child = by_id[child_id]
            if str(child.get("run_type") or "") != "tool":
                continue
            name = str(child.get("name") or "")
            if not name:
                continue
            result.actions_seen.add(name)
            if name not in context.actions:
                continue
            if child.get("error") is not None:
                continue
            fired[name] = fired.get(name, 0) + 1
            child_meta = _metadata(child)
            if "aver_schema_valid" in child_meta:
                value = child_meta["aver_schema_valid"]
                schema_valid[name] = None if value is None else bool(value)

        explicit_run = meta.get("aver_run", meta.get("repetition"))
        try:
            run = int(explicit_run) if explicit_run is not None else order.get((task_id, side), 0)
        except (TypeError, ValueError):
            run = order.get((task_id, side), 0)
        order[(task_id, side)] = max(order.get((task_id, side), 0), run) + 1

        unreadable = root.get("error") is not None
        result.lines.append(EvidenceLine(
            task_id=task_id,
            side=side,
            run=run,
            facts={} if unreadable else facts_for(fired, schema_valid, context.actions),
            unreadable=unreadable,
            meta={key: value for key, value in (
                ("model", _model(root)),
                ("harness", context.harness or "langsmith"),
            ) if value is not None},
        ))
    return result
