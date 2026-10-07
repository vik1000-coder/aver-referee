"""Shared importer machinery: identity, sides, and the skip ledger."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

from aver_cli.evidence import (
    FIRE_IS_HARMFUL,
    SIDE_CANDIDATE,
    SIDE_INCUMBENT,
    SIDES,
    ActionFacts,
    EvidenceLine,
    task_id_from_message,
)


class UnknownFormat(ValueError):
    """No detector matched, or a format name that does not exist."""


class SideUndecidable(ValueError):
    """The importer cannot tell the two sides apart, so it refuses to guess.

    docs/104 §5.6: "Never guess a side." A file imported with the sides
    transposed produces a verdict that is exactly backwards, which is the worst
    failure this tool could have.
    """


@dataclass(frozen=True)
class ImportContext:
    """Flags that decide identity and sides for one file."""

    actions: tuple[str, ...]
    side: Optional[str] = None
    incumbent_model: Optional[str] = None
    candidate_model: Optional[str] = None
    incumbent_provider: Optional[str] = None
    candidate_provider: Optional[str] = None
    task_key: Optional[str] = None
    harness: str = ""
    #: ``(action name, direction)`` pairs, so the generic importer can invert a
    #: bare ``decision`` back into "did the action fire?".
    directions: tuple[tuple[str, str], ...] = ()

    def direction_of(self, name: str) -> str:
        for action, direction in self.directions:
            if action == name:
                return direction
        return FIRE_IS_HARMFUL

    def _match(self, name: str, incumbent: Optional[str], candidate: Optional[str],
               label: str) -> Optional[str]:
        inc = bool(incumbent) and incumbent in name
        cand = bool(candidate) and candidate in name
        if inc and cand:
            raise SideUndecidable(
                f"{label} {name!r} matches both sides; two sides could not be told "
                "apart; pass --side per file"
            )
        if inc:
            return SIDE_INCUMBENT
        if cand:
            return SIDE_CANDIDATE
        return None

    def side_for(self, *, attribute: object = None, model: object = None,
                 provider: object = None) -> str:
        """docs/104 §5.6 precedence: explicit attribute, then --side, then a name match.

        Anything short of those refuses. A file imported with the sides
        transposed produces a verdict that is exactly backwards, so "never guess
        a side" is the one place this tool would rather do nothing.
        """

        if isinstance(attribute, str) and attribute.strip() in SIDES:
            return attribute.strip()
        if self.side in SIDES:
            return self.side  # type: ignore[return-value]
        if provider is not None and (self.incumbent_provider or self.candidate_provider):
            matched = self._match(str(provider), self.incumbent_provider,
                                  self.candidate_provider, "provider")
            if matched:
                return matched
        if model is not None and (self.incumbent_model or self.candidate_model):
            matched = self._match(str(model), self.incumbent_model,
                                  self.candidate_model, "model")
            if matched:
                return matched
        raise SideUndecidable("two sides could not be told apart; pass --side per file")


@dataclass
class ImportResult:
    """Lines kept, plus every record that was not kept and why."""

    lines: list[EvidenceLine] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)
    examples: list[str] = field(default_factory=list)
    actions_seen: set[str] = field(default_factory=set)
    detected_format: str = ""

    def skip(self, reason: str, where: str = "") -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1
        if len(self.examples) < 10:
            self.examples.append(f"{where}: {reason}" if where else reason)

    @property
    def skipped_total(self) -> int:
        return sum(self.skipped.values())

    def extend(self, other: "ImportResult") -> None:
        self.lines.extend(other.lines)
        self.actions_seen.update(other.actions_seen)
        for reason, count in other.skipped.items():
            self.skipped[reason] = self.skipped.get(reason, 0) + count
        for example in other.examples:
            if len(self.examples) < 10:
                self.examples.append(example)

    def to_dict(self) -> dict:
        return {
            "detected_format": self.detected_format,
            "kept_runs": len(self.lines),
            "skipped_total": self.skipped_total,
            "skipped": dict(sorted(self.skipped.items())),
            "examples": list(self.examples),
            "actions_seen": sorted(self.actions_seen),
        }


def json_pointer(document: object, pointer: str) -> Any:
    """RFC 6901, enough of it for ``--task-key``."""

    if pointer in ("", "/"):
        return document
    node: Any = document
    for token in pointer.lstrip("/").split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(node, Mapping):
            if token not in node:
                return None
            node = node[token]
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            try:
                node = node[int(token)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return node


def stable_task_id(
    *,
    explicit: object = None,
    document: object = None,
    context: Optional[ImportContext] = None,
    message: object = None,
) -> Optional[str]:
    """docs/104 §5.6 precedence for task identity."""

    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    if context is not None and context.task_key:
        value = json_pointer(document, context.task_key)
        if value is not None:
            return task_id_from_message(value)
    if message is not None:
        return task_id_from_message(message)
    return None


def first_user_message(messages: object) -> object:
    """The canonical identity fallback: the first user turn, whatever its shape."""

    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            return messages
    if isinstance(messages, Mapping):
        for key in ("input", "messages", "question", "prompt", "text"):
            if key in messages:
                return first_user_message(messages[key])
        return {str(k): messages[k] for k in sorted(messages)}
    if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
        for item in messages:
            if isinstance(item, Mapping) and str(item.get("role", "user")).lower() == "user":
                return item.get("content", item)
        return list(messages)[:1] or None
    return messages


def facts_for(
    fired_names: Mapping[str, int],
    schema_valid: Mapping[str, Optional[bool]],
    actions: Sequence[str],
) -> dict[str, ActionFacts]:
    """Build the per-action fact map for one run.

    An action in ``--actions`` that this run never mentions is recorded as *not
    fired*, count 0 — the trajectory is readable and the action is absent from
    it. An action the run mentions but ``--actions`` does not is the importer's
    business to report, not to record.
    """

    return {
        name: ActionFacts(
            fired=fired_names.get(name, 0) > 0,
            count=fired_names.get(name, 0),
            args_schema_valid=schema_valid.get(name),
        )
        for name in actions
    }
