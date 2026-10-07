"""``aver.yaml`` — the acceptance rules a person wrote down, read strictly.

Two rules govern this module, and both are refusals rather than conveniences.

A key the referee does not know is `unsupported_clause`, never silently stored
(docs/102 §5): an owner who wrote a tolerance the engine ignores would be told
their rule was applied when it was not.

The YAML reader understands only the document shape docs/104 §3 specifies —
scalars, one nested block sequence of mappings, comments — and *refuses* on any
line it cannot account for instead of guessing. A wrong-but-plausible parse of
an acceptance rule is worse than no parse. TOML goes through ``tomllib``, which
exists from Python 3.11; on an older interpreter a ``.toml`` policy is a clear
error naming JSON and YAML as the alternatives, not a crash.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from aver_cli import POLICY_SCHEMA
from aver_cli.evidence import DIRECTIONS, FIRE_IS_HARMFUL

try:  # Python 3.11+
    import tomllib as _tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on 3.10 and older
    _tomllib = None  # type: ignore[assignment]

DEFAULT_TOLERANCE = 0.05
DEFAULT_MIN_EVIDENCE = 5
DEFAULT_DECIDABILITY_THRESHOLD = 0.8
DEFAULT_MAX_UNDECIDED_FRACTION = 0.25
DEFAULT_CONFIDENCE = 0.95

_POLICY_KEYS = frozenset({
    "schema", "decision_point", "actions",
    "decidability_threshold", "max_undecided_fraction", "confidence",
})
_ACTION_KEYS = frozenset({"name", "direction", "tolerance", "harm_tolerance", "min_evidence"})

_MISSING = object()


class PolicyError(ValueError):
    """The policy file cannot be read at all — a usage error, not a refusal."""


class UnsupportedClause(ValueError):
    """A key the referee does not implement. ``aver check`` refuses on this."""

    def __init__(self, keys: list[str]) -> None:
        self.keys = keys
        super().__init__("unsupported clause: " + ", ".join(keys))


# --------------------------------------------------------------------------- #
# The strict minimal YAML reader
# --------------------------------------------------------------------------- #

def _strip_comment(line: str) -> str:
    out: list[str] = []
    quote: Optional[str] = None
    for index, char in enumerate(line):
        if quote:
            out.append(char)
            if char == quote:
                quote = None
            continue
        if char in "\"'":
            quote = char
            out.append(char)
            continue
        if char == "#" and (index == 0 or line[index - 1] in " \t"):
            break
        out.append(char)
    return "".join(out).rstrip()


def _scalar(token: str) -> Any:
    token = token.strip()
    if not token:
        return ""
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
        return token[1:-1]
    lowered = token.lower()
    if lowered in ("null", "~"):
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    if token[0] in "[{&*|>":
        raise PolicyError(
            f"this reader does not support the YAML value {token!r}. "
            "Use the plain shape in docs/104 §3, or write the policy as JSON."
        )
    return token


def parse_minimal_yaml(text: str) -> dict:
    """Read the documented policy shape. Refuse anything else, loudly."""

    root: dict[str, Any] = {}
    sequence: Optional[list[dict[str, Any]]] = None
    item: Optional[dict[str, Any]] = None
    item_indent: Optional[int] = None

    for number, raw in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        body = line.strip()
        if "\t" in raw[:indent]:
            raise PolicyError(f"line {number}: tabs are not valid YAML indentation")

        if body.startswith("- "):
            if sequence is None:
                raise PolicyError(f"line {number}: a list item with no key above it")
            entry = body[2:].strip()
            if ":" not in entry:
                raise PolicyError(
                    f"line {number}: this reader supports only lists of 'key: value' mappings"
                )
            key, _, value = entry.partition(":")
            item = {key.strip(): _scalar(value)}
            item_indent = indent + 2
            sequence.append(item)
            continue

        if ":" not in body:
            raise PolicyError(f"line {number}: expected 'key: value', got {body!r}")
        key, _, value = body.partition(":")
        key = key.strip()
        value = value.strip()

        if item is not None and item_indent is not None and indent >= item_indent:
            item[key] = _scalar(value) if value else None
            continue

        sequence, item, item_indent = None, None, None
        if not value:
            sequence = []
            root[key] = sequence
            continue
        if indent != 0:
            raise PolicyError(f"line {number}: unexpected indentation for top-level key {key!r}")
        root[key] = _scalar(value)
    return root


# --------------------------------------------------------------------------- #
# The policy
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ActionSpec:
    name: str
    direction: str = FIRE_IS_HARMFUL
    tolerance: float = DEFAULT_TOLERANCE
    #: ``None`` disables Clause B for this action (docs/103 §5).
    harm_tolerance: Optional[float] = DEFAULT_TOLERANCE
    min_evidence: int = DEFAULT_MIN_EVIDENCE

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "direction": self.direction,
            "tolerance": self.tolerance,
            "harm_tolerance": self.harm_tolerance,
            "min_evidence": self.min_evidence,
        }


@dataclass(frozen=True)
class CliPolicy:
    decision_point: str = ""
    actions: tuple[ActionSpec, ...] = ()
    decidability_threshold: float = DEFAULT_DECIDABILITY_THRESHOLD
    max_undecided_fraction: float = DEFAULT_MAX_UNDECIDED_FRACTION
    confidence: float = DEFAULT_CONFIDENCE
    schema: str = POLICY_SCHEMA

    @property
    def action_pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple((action.name, action.direction) for action in self.actions)

    def action(self, name: str) -> ActionSpec:
        for candidate in self.actions:
            if candidate.name == name:
                return candidate
        raise KeyError(name)

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "decision_point": self.decision_point,
            "actions": [action.to_dict() for action in self.actions],
            "decidability_threshold": self.decidability_threshold,
            "max_undecided_fraction": self.max_undecided_fraction,
            "confidence": self.confidence,
        }

    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _number(raw: dict, key: str, where: str, *, low: float, high: float,
            default: float) -> float:
    value = raw.get(key, _MISSING)
    if value is _MISSING or value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyError(f"{where}: {key} must be a number, got {value!r}")
    value = float(value)
    if not low <= value <= high:
        raise PolicyError(f"{where}: {key} must be between {low} and {high}, got {value}")
    return value


def policy_from_dict(raw: dict) -> CliPolicy:
    """Build a policy, refusing on any clause the referee does not implement."""

    if not isinstance(raw, dict):
        raise PolicyError("the policy file is not a mapping")
    schema = raw.get("schema") or POLICY_SCHEMA
    if schema != POLICY_SCHEMA:
        raise PolicyError(f"schema is {schema!r}; this build reads {POLICY_SCHEMA!r}")

    unknown = sorted(set(raw) - _POLICY_KEYS)
    actions_raw = raw.get("actions") or []
    if not isinstance(actions_raw, list) or not actions_raw:
        raise PolicyError("actions must be a non-empty list of guarded actions")
    for entry in actions_raw:
        if not isinstance(entry, dict):
            raise PolicyError("each action must be a mapping with at least a name")
        unknown.extend(f"actions[{entry.get('name', '?')}].{key}"
                       for key in sorted(set(entry) - _ACTION_KEYS))
    if unknown:
        raise UnsupportedClause(sorted(unknown))

    actions: list[ActionSpec] = []
    seen: set[str] = set()
    for entry in actions_raw:
        name = str(entry.get("name") or "").strip()
        if not name:
            raise PolicyError("an action has no name")
        if name in seen:
            raise PolicyError(f"action {name!r} is listed twice")
        seen.add(name)
        direction = str(entry.get("direction") or FIRE_IS_HARMFUL)
        if direction not in DIRECTIONS:
            raise PolicyError(
                f"action {name!r}: direction must be one of {DIRECTIONS}, got {direction!r}"
            )
        where = f"action {name!r}"
        tolerance = _number(entry, "tolerance", where, low=0.0, high=1.0,
                            default=DEFAULT_TOLERANCE)
        if "harm_tolerance" in entry and entry["harm_tolerance"] is None:
            harm_tolerance: Optional[float] = None
        else:
            harm_tolerance = _number(entry, "harm_tolerance", where, low=0.0, high=1.0,
                                     default=tolerance)
        min_evidence_raw = entry.get("min_evidence", DEFAULT_MIN_EVIDENCE)
        if isinstance(min_evidence_raw, bool) or not isinstance(min_evidence_raw, (int, float)):
            raise PolicyError(f"{where}: min_evidence must be an integer")
        min_evidence = int(min_evidence_raw)
        if min_evidence < 1:
            raise PolicyError(f"{where}: min_evidence must be at least 1")
        actions.append(ActionSpec(
            name=name,
            direction=direction,
            tolerance=tolerance,
            harm_tolerance=harm_tolerance,
            min_evidence=min_evidence,
        ))

    return CliPolicy(
        decision_point=str(raw.get("decision_point") or ""),
        actions=tuple(actions),
        decidability_threshold=_number(raw, "decidability_threshold", "policy",
                                       low=0.5, high=1.0,
                                       default=DEFAULT_DECIDABILITY_THRESHOLD),
        max_undecided_fraction=_number(raw, "max_undecided_fraction", "policy",
                                       low=0.0, high=1.0,
                                       default=DEFAULT_MAX_UNDECIDED_FRACTION),
        confidence=_number(raw, "confidence", "policy", low=0.5, high=0.9999,
                           default=DEFAULT_CONFIDENCE),
        schema=str(schema),
    )


def load_policy_text(text: str, *, suffix: str = ".yaml") -> CliPolicy:
    suffix = suffix.lower()
    if suffix == ".json":
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PolicyError(f"not valid JSON: {exc.msg} (line {exc.lineno})") from exc
    elif suffix == ".toml":
        if _tomllib is None:
            raise PolicyError(
                "TOML policies need Python 3.11 or newer (tomllib). "
                "This interpreter is older: write the policy as aver.yaml or aver.json."
            )
        try:
            raw = _tomllib.loads(text)
        except Exception as exc:  # tomllib.TOMLDecodeError
            raise PolicyError(f"not valid TOML: {exc}") from exc
    else:
        raw = parse_minimal_yaml(text)
    return policy_from_dict(raw)


def load_policy(path: str | Path) -> CliPolicy:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PolicyError(f"cannot read {path}: {exc.strerror}") from exc
    return load_policy_text(text, suffix=path.suffix)
