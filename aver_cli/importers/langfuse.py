"""Langfuse trace export (docs/104 §5.3): a JSON array of traces with observations.

Tool invocations are observations with ``type == "SPAN"`` whose ``name`` matches a
guarded action, or ``type == "TOOL"`` where that type exists in the export
version. Identity and sides come from ``metadata``, as in §5.2.
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

TOOL_TYPES = ("TOOL", "SPAN")


def _traces(payload: object) -> list[dict]:
    if isinstance(payload, Mapping):
        for key in ("traces", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, Mapping)]
        if "observations" in payload:
            return [dict(payload)]
        return []
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    return []


def detect(payload: object) -> bool:
    traces = _traces(payload)
    return bool(traces) and any("observations" in trace for trace in traces)


def _observations(trace: Mapping) -> list[dict]:
    raw = trace.get("observations")
    return [dict(item) for item in raw or [] if isinstance(item, Mapping)]


def _metadata(node: Mapping) -> dict:
    meta = node.get("metadata")
    return dict(meta) if isinstance(meta, Mapping) else {}


def _model(trace: Mapping) -> Optional[str]:
    meta = _metadata(trace)
    if meta.get("model"):
        return str(meta["model"])
    for observation in _observations(trace):
        if observation.get("model"):
            return str(observation["model"])
    return None


def read(payload: object, context: ImportContext) -> ImportResult:
    result = ImportResult(detected_format="langfuse")
    traces = _traces(payload)
    if not traces:
        result.skip("file contained no traces")
        return result

    prepared = []
    for index, trace in enumerate(traces):
        meta = _metadata(trace)
        task_id = stable_task_id(
            explicit=meta.get("aver_task_id"),
            document=trace,
            context=context,
            message=first_user_message(trace.get("input")),
        )
        if not task_id:
            result.skip("trace carried no task identity and no readable input",
                        f"trace {trace.get('id', index)}")
            continue
        side = context.side_for(attribute=meta.get("aver_side"), model=_model(trace))
        prepared.append((task_id, side, index, trace, meta))

    order: dict[tuple[str, str], int] = {}
    prepared.sort(key=lambda item: (item[0], item[1],
                                    str(item[3].get("timestamp") or ""), item[2]))
    for task_id, side, index, trace, meta in prepared:
        fired: dict[str, int] = {}
        schema_valid: dict[str, Optional[bool]] = {}
        failed_observations = 0
        for observation in _observations(trace):
            kind = str(observation.get("type") or "").upper()
            if kind not in TOOL_TYPES:
                continue
            name = str(observation.get("name") or "")
            if not name:
                continue
            result.actions_seen.add(name)
            if name not in context.actions:
                continue
            level = str(observation.get("level") or "").upper()
            if level == "ERROR" or observation.get("statusMessage") and level == "ERROR":
                failed_observations += 1
                continue
            fired[name] = fired.get(name, 0) + 1
            observation_meta = _metadata(observation)
            if "aver_schema_valid" in observation_meta:
                value = observation_meta["aver_schema_valid"]
                schema_valid[name] = None if value is None else bool(value)

        explicit_run = meta.get("aver_run", meta.get("run"))
        try:
            run = int(explicit_run) if explicit_run is not None else order.get((task_id, side), 0)
        except (TypeError, ValueError):
            run = order.get((task_id, side), 0)
        order[(task_id, side)] = max(order.get((task_id, side), 0), run) + 1

        unreadable = str(trace.get("level") or "").upper() == "ERROR"
        result.lines.append(EvidenceLine(
            task_id=task_id,
            side=side,
            run=run,
            facts={} if unreadable else facts_for(fired, schema_valid, context.actions),
            unreadable=unreadable,
            meta={key: value for key, value in (
                ("model", _model(trace)),
                ("harness", context.harness or "langfuse"),
            ) if value is not None},
        ))
    return result
