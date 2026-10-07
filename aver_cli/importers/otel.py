"""OTel GenAI JSON (docs/86 Path 1): what the collector's file exporter writes.

A trajectory is a root span and its descendants. A tool invocation is a span with
``gen_ai.operation.name == "execute_tool"`` or a tool-name attribute from the
alias chain; it *fired* when such a span exists under the task's root with status
not ERROR and, when the attribute is present, ``aver.tool.executed != false``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Optional

from aver_cli.evidence import EvidenceLine
from aver_cli.importers.base import (
    ImportContext,
    ImportResult,
    facts_for,
    first_user_message,
    stable_task_id,
)

#: docs/86's attribute table, in precedence order.
TOOL_NAME_ATTRS = (
    "gen_ai.tool.name",
    "langsmith.tool.name",
    "langfuse.tool.name",
    "tool.name",
)
TASK_ID_ATTRS = ("aver.task_id",)
SIDE_ATTRS = ("aver.side",)
MODEL_ATTRS = ("gen_ai.request.model", "gen_ai.response.model", "llm.model_name")
INPUT_ATTRS = ("gen_ai.input.messages", "gen_ai.prompt", "input.value")
EXECUTED_ATTR = "aver.tool.executed"
SCHEMA_VALID_ATTR = "aver.tool.schema_valid"
RUN_ATTRS = ("aver.run", "aver.rep")


def _attr_value(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    for key in ("stringValue", "boolValue", "intValue", "doubleValue"):
        if key in value:
            return value[key]
    array = value.get("arrayValue")
    if isinstance(array, Mapping):
        return [_attr_value(item) for item in array.get("values") or []]
    kvlist = value.get("kvlistValue")
    if isinstance(kvlist, Mapping):
        return {str(item["key"]): _attr_value(item.get("value"))
                for item in kvlist.get("values") or []
                if isinstance(item, Mapping) and "key" in item}
    return dict(value)


def flatten_attributes(span: Mapping[str, Any]) -> dict[str, Any]:
    """OTLP's attribute list, or an exporter's already-flat map."""

    raw = span.get("attributes")
    if isinstance(raw, Mapping):
        return {str(key): _attr_value(value) for key, value in raw.items()}
    flat: dict[str, Any] = {}
    for item in raw or []:
        if isinstance(item, Mapping) and "key" in item:
            flat[str(item["key"])] = _attr_value(item.get("value"))
    return flat


def _first(attrs: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if attrs.get(name) is not None:
            return attrs[name]
    return None


def iter_spans(payload: object) -> Iterator[dict]:
    """Yield every span in an OTLP export, a bare list, or a ``{"spans": [...]}``."""

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, Mapping):
                if "resourceSpans" in item or "spans" in item:
                    yield from iter_spans(item)
                else:
                    yield dict(item)
        return
    if not isinstance(payload, Mapping):
        return
    if "resourceSpans" in payload:
        for resource in payload.get("resourceSpans") or []:
            if not isinstance(resource, Mapping):
                continue
            for scope_key in ("scopeSpans", "instrumentationLibrarySpans"):
                for scope in resource.get(scope_key) or []:
                    if not isinstance(scope, Mapping):
                        continue
                    for span in scope.get("spans") or []:
                        if isinstance(span, Mapping):
                            yield dict(span)
        return
    for span in payload.get("spans") or []:
        if isinstance(span, Mapping):
            yield dict(span)


def detect(payload: object) -> bool:
    if isinstance(payload, Mapping) and "resourceSpans" in payload:
        return True
    for span in iter_spans(payload):
        if "spanId" in span or "span_id" in span:
            return True
        attrs = flatten_attributes(span)
        if attrs.get("gen_ai.operation.name") is not None:
            return True
        break
    return False


def _span_id(span: Mapping[str, Any]) -> str:
    return str(span.get("spanId") or span.get("span_id") or "")


def _parent_id(span: Mapping[str, Any]) -> str:
    return str(span.get("parentSpanId") or span.get("parent_span_id") or "")


def _status_error(span: Mapping[str, Any]) -> bool:
    status = span.get("status")
    if isinstance(status, Mapping):
        code = str(status.get("code") or status.get("statusCode") or "")
        return code.upper().endswith("ERROR") or code == "2"
    return str(status or "").upper() == "ERROR"


def _start(span: Mapping[str, Any]) -> int:
    for key in ("startTimeUnixNano", "start_time_unix_nano", "startTime", "start_time"):
        value = span.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return 0


def _tool_name(attrs: Mapping[str, Any], span: Mapping[str, Any]) -> Optional[str]:
    name = _first(attrs, TOOL_NAME_ATTRS)
    if name is not None:
        return str(name)
    if str(attrs.get("gen_ai.operation.name") or "") == "execute_tool":
        return str(span.get("name") or "") or None
    return None


def read(payload: object, context: ImportContext) -> ImportResult:
    result = ImportResult(detected_format="otel")
    spans = list(iter_spans(payload))
    if not spans:
        result.skip("file contained no spans")
        return result

    by_id = {_span_id(span): span for span in spans if _span_id(span)}
    attrs_by_id = {span_id: flatten_attributes(span) for span_id, span in by_id.items()}

    def root_of(span_id: str) -> str:
        seen: set[str] = set()
        current = span_id
        while current and current not in seen:
            seen.add(current)
            parent = _parent_id(by_id.get(current, {}))
            if not parent or parent not in by_id:
                return current
            current = parent
        return current

    roots: dict[str, dict] = {}
    children: dict[str, list[str]] = {}
    for span_id in by_id:
        root = root_of(span_id)
        roots.setdefault(root, by_id[root])
        if span_id != root:
            children.setdefault(root, []).append(span_id)

    trajectories: list[tuple[str, str, dict, dict]] = []
    for root_id, root in roots.items():
        root_attrs = attrs_by_id.get(root_id, {})
        task_id = stable_task_id(
            explicit=_first(root_attrs, TASK_ID_ATTRS),
            document=root_attrs,
            context=context,
            message=first_user_message(_first(root_attrs, INPUT_ATTRS)),
        )
        if not task_id:
            result.skip("root span carried no task identity and no readable input",
                        f"span {root_id}")
            continue
        side = context.side_for(
            attribute=_first(root_attrs, SIDE_ATTRS),
            model=_first(root_attrs, MODEL_ATTRS),
        )
        trajectories.append((task_id, side, root, root_attrs))

    order: dict[tuple[str, str], int] = {}
    trajectories.sort(key=lambda item: (item[0], item[1], _start(item[2]), _span_id(item[2])))
    for task_id, side, root, root_attrs in trajectories:
        root_id = _span_id(root)
        fired: dict[str, int] = {}
        schema_valid: dict[str, Optional[bool]] = {}
        for child_id in [root_id] + children.get(root_id, []):
            child = by_id[child_id]
            child_attrs = attrs_by_id.get(child_id, {})
            name = _tool_name(child_attrs, child)
            if not name:
                continue
            result.actions_seen.add(name)
            if name not in context.actions:
                continue
            if _status_error(child):
                continue
            if child_attrs.get(EXECUTED_ATTR) is False:
                continue
            fired[name] = fired.get(name, 0) + 1
            if SCHEMA_VALID_ATTR in child_attrs:
                valid = child_attrs[SCHEMA_VALID_ATTR]
                schema_valid[name] = None if valid is None else bool(valid)

        explicit_run = _first(root_attrs, RUN_ATTRS)
        if explicit_run is not None:
            try:
                run = int(explicit_run)
            except (TypeError, ValueError):
                run = order.get((task_id, side), 0)
        else:
            run = order.get((task_id, side), 0)
        order[(task_id, side)] = max(order.get((task_id, side), 0), run) + 1

        unreadable = _status_error(root)
        result.lines.append(EvidenceLine(
            task_id=task_id,
            side=side,
            run=run,
            facts={} if unreadable else facts_for(fired, schema_valid, context.actions),
            unreadable=unreadable,
            meta={
                key: value for key, value in (
                    ("model", _first(root_attrs, MODEL_ATTRS)),
                    ("harness", context.harness or "otel"),
                    ("span_id", _span_id(root)),
                ) if value is not None
            },
        ))
    return result
