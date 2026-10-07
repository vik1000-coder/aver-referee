"""Promptfoo results (docs/104 §5.4): ``results.json`` from ``promptfoo eval -o``.

One ``results[].vars`` hash is the task id; ``provider.id`` is the side. The action
*fired* from the ``trajectory:tool-used`` assertion's component result when that
assertion is present, else from the response's tool-call array. Repeats come from
promptfoo's ``repeat`` option, so the run index is the result's ordinal within
``(task, provider)``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional

from aver_cli.evidence import EvidenceLine
from aver_cli.importers.base import (
    ImportContext,
    ImportResult,
    facts_for,
    stable_task_id,
)

TOOL_USED_ASSERTION = "trajectory:tool-used"


def _results(payload: object) -> list[dict]:
    node = payload
    if isinstance(node, Mapping):
        inner = node.get("results")
        if isinstance(inner, Mapping):
            node = inner
            inner = node.get("results")
        if isinstance(inner, list):
            return [dict(item) for item in inner if isinstance(item, Mapping)]
    if isinstance(node, list):
        return [dict(item) for item in node if isinstance(item, Mapping)]
    return []


def detect(payload: object) -> bool:
    results = _results(payload)
    if not results:
        return False
    return any("vars" in item and ("provider" in item or "promptId" in item)
               for item in results)


def _provider_id(item: Mapping) -> Optional[str]:
    provider = item.get("provider")
    if isinstance(provider, Mapping):
        for key in ("id", "label", "model"):
            if provider.get(key):
                return str(provider[key])
    if isinstance(provider, str) and provider:
        return provider
    return None


def _tool_calls(item: Mapping) -> list[dict]:
    response = item.get("response")
    if not isinstance(response, Mapping):
        return []
    for key in ("toolCalls", "tool_calls"):
        calls = response.get(key)
        if isinstance(calls, Sequence) and not isinstance(calls, (str, bytes)):
            return [dict(call) for call in calls if isinstance(call, Mapping)]
    output = response.get("output")
    if isinstance(output, Mapping):
        for key in ("toolCalls", "tool_calls"):
            calls = output.get(key)
            if isinstance(calls, Sequence) and not isinstance(calls, (str, bytes)):
                return [dict(call) for call in calls if isinstance(call, Mapping)]
    return []


def _call_name(call: Mapping) -> str:
    function = call.get("function")
    if isinstance(function, Mapping) and function.get("name"):
        return str(function["name"])
    return str(call.get("name") or "")


def _component_results(item: Mapping) -> list[dict]:
    grading = item.get("gradingResult")
    if not isinstance(grading, Mapping):
        return []
    components = grading.get("componentResults")
    if not isinstance(components, Sequence) or isinstance(components, (str, bytes)):
        return []
    return [dict(component) for component in components if isinstance(component, Mapping)]


def read(payload: object, context: ImportContext) -> ImportResult:
    result = ImportResult(detected_format="promptfoo")
    results = _results(payload)
    if not results:
        result.skip("file contained no results")
        return result

    order: dict[tuple[str, str], int] = {}
    for index, item in enumerate(results):
        vars_raw = item.get("vars")
        task_id = stable_task_id(
            explicit=(vars_raw or {}).get("aver_task_id") if isinstance(vars_raw, Mapping) else None,
            document=item,
            context=context,
            message=vars_raw if vars_raw is not None else item.get("testCase"),
        )
        if not task_id:
            result.skip("result carried no vars and no task identity", f"results[{index}]")
            continue
        provider_id = _provider_id(item)
        side = context.side_for(provider=provider_id, model=provider_id)

        fired: dict[str, int] = {}
        schema_valid: dict[str, Optional[bool]] = {}
        saw_assertion = False
        for component in _component_results(item):
            assertion = component.get("assertion")
            if not isinstance(assertion, Mapping):
                continue
            if str(assertion.get("type") or "") != TOOL_USED_ASSERTION:
                continue
            name = str(assertion.get("value") or "")
            if not name:
                continue
            result.actions_seen.add(name)
            saw_assertion = True
            if name not in context.actions:
                continue
            if component.get("pass"):
                fired[name] = fired.get(name, 0) + 1

        if not saw_assertion:
            for call in _tool_calls(item):
                name = _call_name(call)
                if not name:
                    continue
                result.actions_seen.add(name)
                if name not in context.actions:
                    continue
                fired[name] = fired.get(name, 0) + 1

        run = order.get((task_id, side), 0)
        order[(task_id, side)] = run + 1

        unreadable = bool(item.get("error")) or item.get("success") is None and not item.get("response")
        result.lines.append(EvidenceLine(
            task_id=task_id,
            side=side,
            run=run,
            facts={} if unreadable else facts_for(fired, schema_valid, context.actions),
            unreadable=unreadable,
            meta={key: value for key, value in (
                ("model", provider_id),
                ("harness", context.harness or "promptfoo"),
            ) if value is not None},
        ))
    return result
