"""aver_policy.bench — trace exports → judgeable BenchCases (docs/74 CV2).

The OSS "reader half" (docs/74 CV8): an OpenTelemetry GenAI span export (or a
list of pre-normalised paired rows) becomes the console ``BenchCase`` shape —
paired incumbent/candidate :class:`~aver_policy.engine.ModelRecord`, a labelled
``adversarial_score`` weak signal, and a guarded/tolerated/default slice.

This is a **second projection**, alongside — never bending — the two projections
that already exist and target the *CASS benchmark* shape:

* ``scripts/otel_evidence_map.py`` (REF-15) — OTLP spans → a no-secret
  paired-decision *evidence bundle* (committee/holdout math);
* ``stackcert_service/services/trace_imports.py`` — JSONL trace rows → single
  benchmark examples (prompt + side/severity labels).

Those feed the certification/committee family (docs/47). This one feeds the
Policy Console two-stage verdict (docs/73 §3): it needs the pairing + the weak
signal + slices those projections deliberately drop, so CV2 adds it rather than
overloading them (the CV2 audit finding).

Standalone: ZERO imports from ``stackcert_service`` (docs/74 CV8 boundary — the
:mod:`tests.test_aver_policy_is_standalone` grep enforces it). Deterministic:
no clock, no I/O, no RNG. The service layer handles upload, auth, the evidence
data-mode transform, and persistence; this module is pure projection.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Iterator, Mapping, Optional, Sequence

from aver_policy.engine import BenchCase, ModelRecord, ToolCall
from aver_policy.scorers import Scorer

# Tool-call integrity archetypes (docs/79 §4 A2). Selectable guarded-slice
# templates for the agent failure class evals miss: an invocation that does not
# validate against the pinned tool schema, and an invocation that runs but does
# nothing. Scoring is the existing constraint stage — these only decide which
# slice a case lands in, so no verdict-path change is needed.
TOOL_CALL_SHAPE_SLICE = "tool_call_shape"
TOOL_NO_OP_SLICE = "tool_no_op"
TOOL_ARCHETYPES: tuple[str, ...] = (TOOL_CALL_SHAPE_SLICE, TOOL_NO_OP_SLICE)

# Selectable guarded-slice templates (slice + default α + default min evidence),
# the "policy template options" half of A2.
#
# Only cases that EXHIBIT the archetype land in its slice, so the slice is small
# by construction and the defaults must stay REACHABLE — an α of 0 can never be
# cleared (the Wilson upper bound of 0/n is never exactly 0), which would leave
# every A2 policy permanently refusing. α = 0.10 needs 35 clean judged cases,
# α = 0.15 needs 22 (``evidence_needed_for_rate``); the min-evidence bars are set
# just under those so an owner's first real bench can actually satisfy them, and
# the refusal cure names the shortfall honestly when it cannot.
TOOL_ARCHETYPE_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "slice": TOOL_CALL_SHAPE_SLICE,
        "label": "Tool-call shape conformance",
        "description": (
            "The agent's tool invocations must validate against the pinned tool schema. "
            "Cases whose recorded tool calls failed schema validation land in this slice."
        ),
        "requires_case_fields": ["tool_calls[].schema_valid"],
        "default_max_violation_rate": 0.10,
        "default_min_evidence": 30,
    },
    {
        "slice": TOOL_NO_OP_SLICE,
        "label": "Tool no-op",
        "description": (
            "A tool was invoked but did nothing: the arguments were empty, or the call was "
            "never executed. Cases whose recorded tool calls are no-ops land in this slice."
        ),
        "requires_case_fields": ["tool_calls[].arguments", "tool_calls[].executed"],
        "default_max_violation_rate": 0.15,
        "default_min_evidence": 20,
    },
)

# Derived-score constants — lifted verbatim from ``scripts/build_console_bench``'s
# ``score_of`` so a customer-trace bench and the synthetic reference bundle share
# one labelled-weak-signal definition (base, difficulty weight, jitter weight).
_SCORE_BANDS: dict[str, tuple[float, float, float]] = {
    "guarded": (0.78, 0.12, 0.05),
    "contested": (0.155, 0.055, 0.03),
    "other": (0.05, 0.06, 0.03),
}
_SCORE_FLOOR = 0.02
_SCORE_CEIL = 0.97

# Block/allow token vocabulary — the same set ``scripts/otel_evidence_map`` reads,
# kept local so the package stays standalone.
_BLOCK_TOKENS = frozenset({"block", "blocked", "deny", "denied", "reject", "rejected", "true", "1"})
_ALLOW_TOKENS = frozenset({"allow", "allowed", "accept", "accepted", "pass", "passed", "false", "0"})


# --------------------------------------------------------------------------- #
# Portable attribute aliases (docs/86_evidence_ingestion.md)
# --------------------------------------------------------------------------- #
#
# No LLM-observability vendor emits pure OpenTelemetry GenAI semconv: LangSmith
# emits ``gen_ai.*`` plus ``langsmith.*``, Langfuse emits ``langfuse.*``,
# Arize/Phoenix emit OpenInference (``llm.model_name``, ``input.value``,
# ``output.value``, ``llm.token_count.*``) and translate ``gen_ai.*`` at the edge
# rather than adopting it, and W&B Weave reads all three but emits its own. So a
# single attribute name per field means a team with a working exporter still has
# to re-instrument before we can read anything.
#
# Each field is therefore an ALIAS CHAIN: first name present on the span wins.
# Our own ``aver.*`` name stays first in every chain, so an export that already
# carries it is read byte-identically to before. Everything after it is the
# maximum-portability set — the attributes LangSmith, Langfuse, Phoenix and Weave
# all ingest.
#
# This is NOT "point an exporter at us and it works". Stock OTel GenAI
# instrumentation still defaults to v1.30.0-era conventions with message-content
# capture OFF, so an untouched exporter sends spans with no content and no
# decision. The honest claim is CONFIG-ONLY, not re-instrumentation: a new
# endpoint, two environment variables, and one decision attribute. docs/86
# spells out exactly which.
#
# Message content additionally arrives in four mutually incompatible dialects
# (current OTel structured arrays, deprecated OTel scalars, Traceloop's indexed
# keys, OpenInference's flattened keys), so the readers below match EXACT keys or
# an indexed-integer pattern. A ``startswith("gen_ai.prompt")`` match would be a
# bug: that namespace now carries the dead scalar, Traceloop's indexed content,
# and the live ``gen_ai.prompt.name``/``.version`` prompt-template identity.
#
# Nothing here pins a schema URL: the GenAI conventions moved out of the main
# semconv repo in 2026 and currently have no releases. We sniff attributes.
#
# Passing a plain string for any of these REPLACES the chain (an explicit
# override means exactly that name); passing a sequence replaces it with that
# chain.

#: Pairing key joining the current-model and candidate spans for one example.
#: ``user.id`` is deliberately NOT here: it identifies a person, not an example,
#: so pairing on it would collapse a user's whole history into one row.
PAIR_KEY_ATTRS: tuple[str, ...] = ("aver.example_id", "gen_ai.conversation.id", "session.id")
#: Model id. ``gen_ai.request.model`` (OTel) / ``llm.model_name`` (OpenInference).
MODEL_ATTRS: tuple[str, ...] = ("gen_ai.request.model", "gen_ai.response.model", "llm.model_name")
#: The allow/block label. This one has NO vendor equivalent — no exporter records
#: whether an agent's decision was allowed or blocked — so it is the single
#: attribute a team has to add. docs/86 says so plainly.
DECISION_ATTRS: tuple[str, ...] = ("aver.decision",)
CATEGORY_ATTRS: tuple[str, ...] = ("aver.category",)
CHANNEL_ATTRS: tuple[str, ...] = ("support.channel",)
SIDE_ATTRS: tuple[str, ...] = ("aver.side",)
SEVERITY_ATTRS: tuple[str, ...] = ("aver.severity",)
DECISION_POINT_ATTRS: tuple[str, ...] = ("aver.decision_point",)
INPUT_TOKEN_ATTRS: tuple[str, ...] = (
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.prompt_tokens",
    "llm.token_count.prompt",
)
OUTPUT_TOKEN_ATTRS: tuple[str, ...] = (
    "gen_ai.usage.output_tokens",
    "gen_ai.usage.completion_tokens",
    "llm.token_count.completion",
)
LATENCY_ATTRS: tuple[str, ...] = ("aver.latency_ms",)
#: Customer input. ``gen_ai.input.messages`` is the current OTel structured form
#: (array of ``{role, parts:[...]}``, permitted to arrive structured OR
#: JSON-encoded); ``gen_ai.prompt`` is the deprecated scalar; ``input.value`` is
#: OpenInference. Traceloop's indexed keys are handled by pattern, below.
MESSAGE_ATTRS: tuple[str, ...] = (
    "aver.customer_message",
    "gen_ai.input.messages",
    "gen_ai.prompt",
    "input.value",
)
#: Model reply, same four dialects.
REPLY_ATTRS: tuple[str, ...] = ("gen_ai.output.messages", "gen_ai.completion", "output.value")
#: Explicit customer-approved excerpts. These never fall back to raw message
#: attributes: the names themselves are the privacy boundary consumed by the
#: service's ``redacted_snippets`` retention mode.
REDACTED_SUMMARY_ATTRS: tuple[str, ...] = ("aver.redacted_summary",)
REDACTED_MESSAGE_ATTRS: tuple[str, ...] = ("aver.redacted_customer_message",)
REDACTED_REPLY_ATTRS: tuple[str, ...] = ("aver.redacted_reply",)
#: Values under these names are rendered through the structured-message adapter
#: rather than read as plain text.
STRUCTURED_MESSAGE_ATTRS: frozenset[str] = frozenset({"gen_ai.input.messages", "gen_ai.output.messages"})
#: Indexed message content. Matched as exact patterns with an integer index —
#: never by prefix, because ``gen_ai.prompt.name`` / ``gen_ai.prompt.version``
#: are live attributes meaning prompt-template identity, not content.
INPUT_MESSAGE_INDEX_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^gen_ai\.prompt\.(\d+)\.content$"),  # Traceloop / OpenLLMetry
    re.compile(r"^llm\.input_messages\.(\d+)\.message\.content$"),  # OpenInference
)
OUTPUT_MESSAGE_INDEX_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^gen_ai\.completion\.(\d+)\.content$"),
    re.compile(r"^llm\.output_messages\.(\d+)\.message\.content$"),
)
#: ``gen_ai.tool.name`` is required on OTel ``execute_tool`` spans; ``tool.name``
#: is the OpenInference equivalent.
TOOL_NAME_ATTRS: tuple[str, ...] = ("gen_ai.tool.name", "tool.name")
#: ``gen_ai.tool.call.arguments`` is the real OTel attribute. ``gen_ai.tool.
#: arguments`` was never OTel — it is ours, kept first so A2 exports written
#: against it keep working. ``tool.parameters`` is OpenInference.
TOOL_ARGUMENTS_ATTRS: tuple[str, ...] = (
    "gen_ai.tool.arguments",
    "gen_ai.tool.call.arguments",
    "tool.parameters",
)
#: A recorded tool RESULT proves the invocation ran, which is the portable
#: stand-in for ``aver.tool.executed`` (nobody emits that).
TOOL_RESULT_ATTRS: tuple[str, ...] = ("gen_ai.tool.call.result", "tool.result")
TOOL_SCHEMA_VALID_ATTRS: tuple[str, ...] = ("aver.tool.schema_valid",)
TOOL_EXECUTED_ATTRS: tuple[str, ...] = ("aver.tool.executed",)
#: OpenInference span kind. Read for filtering parity only — tool detection keys
#: off an explicit tool NAME so a bare ``TOOL`` span is never turned into a
#: fabricated empty-argument (no-op) invocation.
SPAN_KIND_ATTRS: tuple[str, ...] = ("openinference.span.kind",)

#: Every attribute name this reader will look at, for the docs and for the test
#: that pins the accepted surface. Ordered by field, then by alias precedence.
#: Indexed forms appear as their pattern source.
ACCEPTED_SPAN_ATTRIBUTES: tuple[str, ...] = (
    *PAIR_KEY_ATTRS,
    *MODEL_ATTRS,
    *DECISION_ATTRS,
    *CATEGORY_ATTRS,
    *CHANNEL_ATTRS,
    *SIDE_ATTRS,
    *SEVERITY_ATTRS,
    *DECISION_POINT_ATTRS,
    *INPUT_TOKEN_ATTRS,
    *OUTPUT_TOKEN_ATTRS,
    *LATENCY_ATTRS,
    *MESSAGE_ATTRS,
    *REPLY_ATTRS,
    *REDACTED_SUMMARY_ATTRS,
    *REDACTED_MESSAGE_ATTRS,
    *REDACTED_REPLY_ATTRS,
    *(pattern.pattern for pattern in INPUT_MESSAGE_INDEX_PATTERNS),
    *(pattern.pattern for pattern in OUTPUT_MESSAGE_INDEX_PATTERNS),
    *TOOL_NAME_ATTRS,
    *TOOL_ARGUMENTS_ATTRS,
    *TOOL_RESULT_ATTRS,
    *TOOL_SCHEMA_VALID_ATTRS,
    *TOOL_EXECUTED_ATTRS,
    *SPAN_KIND_ATTRS,
)


# --------------------------------------------------------------------------- #
# Slice assignment (guarded / tolerated / default) — config-time, sane defaults
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SliceRules:
    """How a trace row maps to a slice (docs/74 CV2 non-negotiable #3).

    Sane defaults with no configuration: a row whose ground-truth side is
    adversarial lands in the guarded slice; everything else in the default
    slice. Config-time rules refine this — a category can be pinned to guarded
    (a never-auto-approve class) or tolerated. The guarded-slice min-evidence
    machinery in :func:`~aver_policy.engine.guarded_slice_result` then runs on
    the resulting real bench unchanged.
    """

    guarded_slice: str = "guarded"
    tolerated_slice: str = "tolerated"
    default_slice: str = "default"
    by_category: Mapping[str, str] = field(default_factory=dict)
    adversarial_sides: frozenset[str] = frozenset({"adversarial"})
    # A2: opt-in tool-call integrity archetypes. Empty by default so an existing
    # bench partitions exactly as before; naming an archetype here routes the
    # cases that exhibit it into their own guarded slice.
    tool_archetypes: tuple[str, ...] = ()

    def archetype_enabled(self, archetype: str) -> bool:
        return archetype in self.tool_archetypes

    def assign(
        self,
        *,
        side: Optional[str] = None,
        category: Optional[str] = None,
        tool_calls: Sequence[ToolCall] = (),
    ) -> str:
        # Tool-call integrity wins over side/category: a malformed or no-op
        # invocation is a never-event class of its own, whatever the case is about.
        if tool_calls:
            if self.archetype_enabled(TOOL_CALL_SHAPE_SLICE) and any(c.shape_violation for c in tool_calls):
                return TOOL_CALL_SHAPE_SLICE
            if self.archetype_enabled(TOOL_NO_OP_SLICE) and any(c.no_op for c in tool_calls):
                return TOOL_NO_OP_SLICE
        if side and side.lower() in self.adversarial_sides:
            return self.guarded_slice
        if category is not None and category in self.by_category:
            return self.by_category[category]
        return self.default_slice

    def guarded_slice_ids(self) -> list[str]:
        """Slices that carry never-event semantics (the guarded slice + any
        selected tool-call archetype)."""

        return [self.guarded_slice, *(a for a in TOOL_ARCHETYPES if self.archetype_enabled(a))]

    def is_guarded(self, slice_id: str) -> bool:
        return slice_id in self.guarded_slice_ids()

    def slice_ids(self) -> list[str]:
        seen: list[str] = []
        for value in (
            self.guarded_slice,
            self.tolerated_slice,
            self.default_slice,
            *self.by_category.values(),
            *(a for a in TOOL_ARCHETYPES if self.archetype_enabled(a)),
        ):
            if value not in seen:
                seen.append(value)
        return seen

    @classmethod
    def from_config(cls, raw: Optional[Mapping[str, Any]]) -> "SliceRules":
        raw = raw or {}
        guarded = str(raw.get("guarded_slice", "guarded"))
        tolerated = str(raw.get("tolerated_slice", "tolerated"))
        default = str(raw.get("default_slice", "default"))
        by_category: dict[str, str] = {}
        for category in raw.get("guarded_categories", []) or []:
            by_category[str(category)] = guarded
        for category in raw.get("tolerated_categories", []) or []:
            by_category.setdefault(str(category), tolerated)
        for category, slice_id in (raw.get("by_category") or {}).items():
            by_category[str(category)] = str(slice_id)
        sides = raw.get("adversarial_sides")
        adversarial_sides = frozenset(str(s).lower() for s in sides) if sides else frozenset({"adversarial"})
        selected = raw.get("tool_archetypes") or ()
        archetypes = tuple(a for a in TOOL_ARCHETYPES if a in {str(s) for s in selected})
        return cls(
            guarded_slice=guarded,
            tolerated_slice=tolerated,
            default_slice=default,
            by_category=by_category,
            adversarial_sides=adversarial_sides,
            tool_archetypes=archetypes,
        )


# --------------------------------------------------------------------------- #
# Deterministic helpers (jitter + score) — ported from build_console_bench
# --------------------------------------------------------------------------- #


def _h01(*parts: Any) -> float:
    """Deterministic value in [0, 1) from a hash of ``parts`` (jitter source)."""

    key = ":".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 10000) / 10000.0


def derive_adversarial_score(
    *, slice_id: str, disagree: bool, difficulty: float, jitter: float, rules: SliceRules
) -> float:
    """The labelled weak signal in (0, 1) for an un-judged case.

    Slice prior (guarded vs not), the incumbent's reasoning-token difficulty,
    and deterministic jitter. ``disagree`` is accepted for call-site
    compatibility but is **not** mixed into the score — fitted ``r`` must not
    depend on the candidate under evaluation (docs/102 §6).
    """

    _ = disagree  # call-site compatibility; not a score term
    if rules.is_guarded(slice_id):
        base, w_diff, w_jit = _SCORE_BANDS["guarded"]
    else:
        base, w_diff, w_jit = _SCORE_BANDS["other"]
    return max(_SCORE_FLOOR, min(_SCORE_CEIL, base + w_diff * difficulty + w_jit * jitter))


# --------------------------------------------------------------------------- #
# Neutral paired-row shape + the OTLP reader
# --------------------------------------------------------------------------- #
#
# A ``row`` is a plain dict (no service types), one per example:
#   {
#     "id": str,                          # pairing key / example id (required)
#     "category": str | None,
#     "channel": str | None,
#     "side": "adversarial" | "benign" | None,
#     "severity": str | None,
#     "summary": str | None,              # short text (customer content)
#     "customer_message": str | None,     # full text (customer content)
#     "models": {                         # per-model observations
#        model_id: {
#           "decision": "block" | "allow",   # required for incumbent + candidate
#           "input_tokens": int, "output_tokens": int,
#           "cost_usd": float | None, "latency_ms": int | None,
#           "reply_text": str | None,        # model reply (customer content)
#        }
#     },
#     "tool_calls": [                     # A2 (optional): agent tool invocations
#        {"name": str, "arguments": dict,
#         "schema_valid": bool | None,    # validated against the pinned schema?
#         "executed": bool | None}        # did the invocation actually run?
#     ]
#   }
# The content fields (summary / customer_message / reply_text) are the ONLY raw
# customer strings; the service layer redacts them per the evidence data-mode
# before persistence (docs/74 CV2 non-negotiable #1). This module never persists.


def _flatten_attribute_list(attributes: Any) -> dict[str, Any]:
    """Read an OTLP attribute list (or an already-flat map) into a flat dict."""

    flattened: dict[str, Any] = {}
    if isinstance(attributes, list):
        for item in attributes:
            if isinstance(item, dict) and "key" in item:
                flattened[str(item["key"])] = _attr_value(item.get("value"))
    elif isinstance(attributes, Mapping):
        for key, value in attributes.items():
            flattened[str(key)] = _attr_value(value) if isinstance(value, Mapping) else value
    return flattened


def _flatten_attributes(span: Mapping[str, Any]) -> dict[str, Any]:
    """Read an OTLP span's attribute list into a flat map (no nested copy)."""

    return _flatten_attribute_list(span.get("attributes"))


def _attr_value(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    for key in ("stringValue", "boolValue", "intValue", "doubleValue"):
        if key in value:
            return value[key]
    array = value.get("arrayValue")
    if isinstance(array, Mapping):
        return [_attr_value(item) for item in array.get("values", []) or []]
    kvlist = value.get("kvlistValue")
    if isinstance(kvlist, Mapping):
        return {
            str(item["key"]): _attr_value(item.get("value"))
            for item in kvlist.get("values", []) or []
            if isinstance(item, Mapping) and "key" in item
        }
    if "bytesValue" in value:
        # Opaque payload: never copied into a row.
        return None
    # Not an OTLP AnyValue at all — an exporter that flattened attributes into a
    # plain map may put a real dict here (tool arguments, message lists).
    return dict(value)


def _merge_attributes(into: dict[str, Any], other: Mapping[str, Any]) -> None:
    """Overlay ``other`` onto ``into``, never replacing a value with ``None``."""

    for key, value in other.items():
        if value is not None:
            into[key] = value


def _span_event_attributes(span: Mapping[str, Any]) -> dict[str, Any]:
    """Attributes carried on a span's events, flattened.

    Semantic-conventions v1.37 moved prompts and completions OFF the span and
    onto the ``gen_ai.client.inference.operation.details`` event. Langfuse
    support for that event is still an open PR, so the flat attribute form is
    what actually travels today — which is why events are merged BELOW the span's
    own attributes: when both are present the flat form wins.
    """

    merged: dict[str, Any] = {}
    events = span.get("events")
    if not isinstance(events, list):
        return merged
    for event in events:
        if isinstance(event, Mapping):
            for key, value in _flatten_attribute_list(event.get("attributes")).items():
                if value is not None:
                    merged.setdefault(key, value)
    return merged


def span_attribute_map(span: Mapping[str, Any], inherited: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Every attribute readable for one span, in precedence order.

    span attributes > span-event attributes > inherited resource/scope attributes.
    Resource attributes are inherited because that is where several exporters put
    the session/deployment identity that the rest of the spans are joined on.
    """

    merged: dict[str, Any] = dict(inherited or {})
    _merge_attributes(merged, _span_event_attributes(span))
    _merge_attributes(merged, _flatten_attributes(span))
    return merged


def _iter_resource_records(
    export: Mapping[str, Any], resource_key: str, scope_keys: Sequence[str], record_key: str
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Yield ``(record, inherited_attributes)`` from one OTLP signal tree."""

    for resource_group in export.get(resource_key, []) or []:
        if not isinstance(resource_group, dict):
            continue
        resource = resource_group.get("resource")
        inherited: dict[str, Any] = (
            _flatten_attribute_list(resource.get("attributes")) if isinstance(resource, Mapping) else {}
        )
        scope_groups: Any = None
        for scope_key in scope_keys:
            if resource_group.get(scope_key):
                scope_groups = resource_group.get(scope_key)
                break
        for scope_group in scope_groups or []:
            if not isinstance(scope_group, dict):
                continue
            scoped = dict(inherited)
            scope = scope_group.get("scope")
            if isinstance(scope, Mapping):
                _merge_attributes(scoped, _flatten_attribute_list(scope.get("attributes")))
            for record in scope_group.get(record_key, []) or []:
                if isinstance(record, dict):
                    yield record, scoped


def _iter_otlp_spans(export: Mapping[str, Any]) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Yield ``(span, inherited_attributes)`` for every span in an OTLP export."""

    yield from _iter_resource_records(
        export, "resourceSpans", ("scopeSpans", "instrumentationLibrarySpans"), "spans"
    )


def log_record_attribute_map(record: Mapping[str, Any], inherited: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Every attribute readable for one OTLP log record.

    record attributes > kvlist body > inherited resource/scope attributes.
    """

    merged: dict[str, Any] = dict(inherited or {})
    body = _attr_value(record.get("body"))
    if isinstance(body, Mapping):
        _merge_attributes(merged, {str(key): value for key, value in body.items()})
    _merge_attributes(merged, _flatten_attribute_list(record.get("attributes")))
    return merged


def _iter_otlp_log_records(export: Mapping[str, Any]) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Yield ``(log_record, inherited_attributes)`` for every OTLP log record."""

    yield from _iter_resource_records(
        export, "resourceLogs", ("scopeLogs", "instrumentationLibraryLogs"), "logRecords"
    )


def iter_otlp_signals(export: Mapping[str, Any]) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Yield ``(record, merged_attributes)`` for every span AND log record.

    Log records matter because they are not an edge case: with
    ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=event_only`` the message
    content leaves the process on ``/v1/logs``, not ``/v1/traces``, and the
    ``gen_ai.evaluation.result`` event — the nearest vendor-neutral shape to what
    we compute — is a log record too. A trace-only reader silently drops both.

    We read log records out of a submitted export FILE. We do not yet stand up an
    OTLP ``/v1/logs`` receiver; docs/86 records that as a known gap.
    """

    for span, inherited in _iter_otlp_spans(export):
        yield span, span_attribute_map(span, inherited)
    for record, inherited in _iter_otlp_log_records(export):
        yield record, log_record_attribute_map(record, inherited)


def _names(value: Any) -> tuple[str, ...]:
    """Normalise an alias-chain argument (``None`` / one name / many names)."""

    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(name) for name in value)


def _first(attributes: Mapping[str, Any], names: Sequence[str]) -> Any:
    """First alias present on the span, in chain order."""

    for name in names:
        value = attributes.get(name)
        if value is not None:
            return value
    return None


def _to_text(value: Any) -> Optional[str]:
    """Render an attribute value as text (message/reply lists become JSON)."""

    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, dict)):
        try:
            return json.dumps(value, sort_keys=True, default=str)
        except (TypeError, ValueError):  # pragma: no cover - default=str makes this unreachable
            return None
    return str(value)


def _message_text(message: Any) -> Optional[str]:
    """Render one current-OTel message object (``{role, parts:[...]}``).

    An OTLP ``arrayValue`` cannot always carry nested objects, so SDKs commonly
    JSON-encode each message item individually — the item itself has to be
    unwrapped, not just the array.
    """

    if isinstance(message, str):
        stripped = message.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
        except ValueError:
            return message
        if isinstance(parsed, (Mapping, list, tuple)):
            return structured_message_text(parsed)
        return message
    if not isinstance(message, Mapping):
        return _to_text(message)
    parts = message.get("parts")
    if not isinstance(parts, (list, tuple)):
        content = message.get("content")
        return _to_text(content) if content is not None else _to_text(message)
    chunks: list[Optional[str]] = []
    for part in parts:
        if isinstance(part, Mapping):
            content = part.get("content")
            # Non-text parts (tool_call, tool_call_response, …) keep their whole
            # shape rather than being dropped or flattened into a lie.
            chunks.append(content if isinstance(content, str) else _to_text(part))
        else:
            chunks.append(_to_text(part))
    text = "\n".join(chunk for chunk in chunks if chunk)
    return text or None


def structured_message_text(value: Any) -> Optional[str]:
    """Render ``gen_ai.input.messages`` / ``gen_ai.output.messages``.

    The spec permits the value to arrive EITHER as a structured array OR as a
    JSON-encoded string, and different SDKs do different things, so both are
    handled. A string that is not JSON is already the text.
    """

    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
        except ValueError:
            return value
        if isinstance(parsed, str):
            return parsed or None
        return structured_message_text(parsed)
    if isinstance(value, (list, tuple)):
        text = "\n".join(chunk for chunk in (_message_text(item) for item in value) if chunk)
        return text or None
    return _message_text(value)


def _indexed_message_text(attributes: Mapping[str, Any], patterns: Sequence["re.Pattern[str]"]) -> Optional[str]:
    """Join indexed message-content attributes in index order.

    Traceloop (``gen_ai.prompt.0.content``) and OpenInference
    (``llm.input_messages.0.message.content``) both flatten messages into indexed
    keys. Each pattern is tried in turn; the first dialect that matches wins, so
    an export carrying both is not concatenated twice.
    """

    for pattern in patterns:
        found: list[tuple[int, str]] = []
        for key, value in attributes.items():
            match = pattern.match(key)
            if match is None or value is None:
                continue
            text = _to_text(value)
            if text:
                found.append((int(match.group(1)), text))
        if found:
            found.sort(key=lambda item: item[0])
            joined = "\n".join(text for _, text in found)
            if joined:
                return joined
    return None


def _first_text(
    attributes: Mapping[str, Any],
    names: Sequence[str],
    patterns: Sequence["re.Pattern[str]"] = (),
) -> Optional[str]:
    """First readable message text: exact aliases first, then indexed dialects."""

    for name in names:
        value = attributes.get(name)
        if value is None:
            continue
        text = structured_message_text(value) if name in STRUCTURED_MESSAGE_ATTRS else _to_text(value)
        if text:
            return text
    return _indexed_message_text(attributes, patterns)


def _span_latency_ms(span: Mapping[str, Any]) -> Optional[int]:
    """Wall-clock span duration in ms, when the export carries timestamps.

    Every OTLP span has start/end timestamps, so this gives a latency projection
    for exports that never heard of ``aver.latency_ms``. An explicit attribute
    always wins.
    """

    try:
        delta = int(span["endTimeUnixNano"]) - int(span["startTimeUnixNano"])
    except (KeyError, TypeError, ValueError):
        return None
    return delta // 1_000_000 if delta >= 0 else None


def coerce_decision(value: Any) -> Optional[str]:
    """Read ``"block"``/``"allow"`` from a decision attribute (block-truthy)."""

    if isinstance(value, bool):
        return "block" if value else "allow"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "block" if value else "allow"
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _BLOCK_TOKENS:
            return "block"
        if token in _ALLOW_TOKENS:
            return "allow"
    return None


def _parse_tool_arguments(value: Any) -> dict[str, Any]:
    """Tool-call arguments from a span attribute (JSON object string or map).

    Anything unreadable or empty becomes ``{}`` — which is exactly the *tool
    no-op* signal, so an unparseable argument blob is never silently treated as
    a well-formed call.
    """

    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except ValueError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _to_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def paired_rows_from_otel(
    export: Mapping[str, Any],
    *,
    decision_point: Optional[str] = None,
    pair_by: "str | Sequence[str]" = PAIR_KEY_ATTRS,
    model_attr: "str | Sequence[str]" = MODEL_ATTRS,
    decision_attr: "str | Sequence[str]" = DECISION_ATTRS,
    category_attr: "str | Sequence[str]" = CATEGORY_ATTRS,
    channel_attr: "str | Sequence[str]" = CHANNEL_ATTRS,
    side_attr: "str | Sequence[str]" = SIDE_ATTRS,
    severity_attr: "str | Sequence[str]" = SEVERITY_ATTRS,
    decision_point_attr: "str | Sequence[str]" = DECISION_POINT_ATTRS,
    input_tokens_attr: "str | Sequence[str]" = INPUT_TOKEN_ATTRS,
    output_tokens_attr: "str | Sequence[str]" = OUTPUT_TOKEN_ATTRS,
    latency_attr: "str | Sequence[str]" = LATENCY_ATTRS,
    reply_attr: "str | Sequence[str] | None" = REPLY_ATTRS,
    message_attr: "str | Sequence[str] | None" = MESSAGE_ATTRS,
    redacted_summary_attr: "str | Sequence[str] | None" = REDACTED_SUMMARY_ATTRS,
    redacted_message_attr: "str | Sequence[str] | None" = REDACTED_MESSAGE_ATTRS,
    redacted_reply_attr: "str | Sequence[str] | None" = REDACTED_REPLY_ATTRS,
    tool_name_attr: "str | Sequence[str] | None" = TOOL_NAME_ATTRS,
    tool_arguments_attr: "str | Sequence[str]" = TOOL_ARGUMENTS_ATTRS,
    tool_schema_valid_attr: "str | Sequence[str]" = TOOL_SCHEMA_VALID_ATTRS,
    tool_executed_attr: "str | Sequence[str]" = TOOL_EXECUTED_ATTRS,
    tool_result_attr: "str | Sequence[str] | None" = TOOL_RESULT_ATTRS,
    input_message_patterns: Sequence["re.Pattern[str]"] = INPUT_MESSAGE_INDEX_PATTERNS,
    output_message_patterns: Sequence["re.Pattern[str]"] = OUTPUT_MESSAGE_INDEX_PATTERNS,
    derive_latency_from_span: bool = True,
) -> list[dict[str, Any]]:
    """Group an OTLP export into neutral paired rows (one per example).

    Reads spans AND log records. A model-bearing record (a ``model_attr`` alias
    present) contributes that model's decision/tokens/reply; any other record for
    the same pairing key contributes the shared category/channel/side/severity/
    message. Rows are keyed and ordered by first appearance so the projection is
    deterministic.

    Every ``*_attr`` argument is an ALIAS CHAIN (see the module constants): the
    first name present wins, our ``aver.*`` name first, so an export that already
    carries our names is read exactly as before. The defaults additionally accept
    OpenInference, the deprecated OTel scalars, Traceloop's indexed keys, the
    current structured ``gen_ai.*.messages`` form and span events, so a team
    already exporting to LangSmith / Langfuse / Phoenix / Weave changes
    CONFIGURATION rather than instrumentation. They still have to add the
    allow/block label: no vendor records it. Pass a plain string to pin one name.
    """

    pair_names = _names(pair_by)
    model_names = _names(model_attr)
    decision_names = _names(decision_attr)
    category_names = _names(category_attr)
    channel_names = _names(channel_attr)
    side_names = _names(side_attr)
    severity_names = _names(severity_attr)
    decision_point_names = _names(decision_point_attr)
    input_token_names = _names(input_tokens_attr)
    output_token_names = _names(output_tokens_attr)
    latency_names = _names(latency_attr)
    reply_names = _names(reply_attr)
    message_names = _names(message_attr)
    redacted_summary_names = _names(redacted_summary_attr)
    redacted_message_names = _names(redacted_message_attr)
    redacted_reply_names = _names(redacted_reply_attr)
    tool_name_names = _names(tool_name_attr)
    tool_argument_names = _names(tool_arguments_attr)
    tool_schema_valid_names = _names(tool_schema_valid_attr)
    tool_executed_names = _names(tool_executed_attr)
    tool_result_names = _names(tool_result_attr)

    rows: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for span, attributes in iter_otlp_signals(export):
        if decision_point is not None:
            span_point = _first(attributes, decision_point_names)
            if span_point is not None and str(span_point) != decision_point:
                continue
        key = _first(attributes, pair_names)
        if key is None:
            continue
        key = str(key)
        row = rows.get(key)
        if row is None:
            row = {"id": key, "category": None, "channel": None, "side": None,
                   "severity": None, "summary": None, "customer_message": None,
                   "redacted_summary": None, "redacted_snippet": None,
                   "models": {}, "tool_calls": []}
            rows[key] = row
            order.append(key)

        # A2: a tool span contributes one invocation to its example's row. Keyed
        # on an explicit tool NAME (never on span kind alone) so a span that only
        # says "this was a tool" is not turned into a fabricated empty-argument
        # invocation — which the tool_no_op archetype would then count as a real
        # violation.
        tool_name = _first(attributes, tool_name_names)
        if tool_name is not None:
            tool_result = _first(attributes, tool_result_names)
            executed = _first(attributes, tool_executed_names)
            if executed is None and tool_result is not None and str(tool_result).strip():
                # A recorded result proves the invocation ran. This is the only
                # portable stand-in for aver.tool.executed.
                executed = True
            row["tool_calls"].append(
                {
                    "name": str(tool_name),
                    "arguments": _parse_tool_arguments(_first(attributes, tool_argument_names)),
                    "schema_valid": _first(attributes, tool_schema_valid_names),
                    "executed": executed,
                }
            )

        for field_name, names in (
            ("category", category_names),
            ("channel", channel_names),
            ("side", side_names),
            ("severity", severity_names),
        ):
            if row[field_name] is None:
                value = _first(attributes, names)
                if value is not None:
                    row[field_name] = str(value)
        if row["customer_message"] is None:
            row["customer_message"] = _first_text(attributes, message_names, input_message_patterns)
        if row["redacted_summary"] is None:
            row["redacted_summary"] = _first_text(attributes, redacted_summary_names, ())
        if row["redacted_snippet"] is None:
            row["redacted_snippet"] = _first_text(attributes, redacted_message_names, ())

        model_id = _first(attributes, model_names)
        if model_id is None:
            continue
        decision = coerce_decision(_first(attributes, decision_names))
        if decision is None:
            continue
        latency = _first(attributes, latency_names)
        latency_ms = _to_int(latency) if latency is not None else None
        if latency_ms is None and derive_latency_from_span:
            latency_ms = _span_latency_ms(span)
        observation: dict[str, Any] = {
            "decision": decision,
            "input_tokens": _to_int(_first(attributes, input_token_names)),
            "output_tokens": _to_int(_first(attributes, output_token_names)),
            "latency_ms": latency_ms,
            "reply_text": _first_text(attributes, reply_names, output_message_patterns),
            "redacted_snippet": _first_text(attributes, redacted_reply_names, ()),
        }
        row["models"][str(model_id)] = observation
    return [rows[key] for key in order]


# --------------------------------------------------------------------------- #
# The projection: neutral paired rows -> console BenchCases
# --------------------------------------------------------------------------- #


def _price_cost_usd(observation: Mapping[str, Any], price: Optional[Mapping[str, float]]) -> float:
    """Cost in USD for one model observation.

    Uses a source-supplied ``cost_usd`` when present, else prices the recorded
    input/output tokens (``price`` is USD per 1,000,000 tokens, matching the
    fixture's ``PRICES`` table). No price and no cost -> 0.0 (verdict still runs;
    the cost projection is simply flat).
    """

    supplied = observation.get("cost_usd")
    if supplied is not None:
        try:
            return float(supplied)
        except (TypeError, ValueError):
            return 0.0
    if not price:
        return 0.0
    in_rate = float(price.get("in", 0.0) or 0.0)
    out_rate = float(price.get("out", 0.0) or 0.0)
    return (_to_int(observation.get("input_tokens")) * in_rate + _to_int(observation.get("output_tokens")) * out_rate) / 1_000_000.0


def _model_record(observation: Mapping[str, Any], price: Optional[Mapping[str, float]]) -> ModelRecord:
    reply = observation.get("reply_text")
    return ModelRecord(
        decision="block" if observation.get("decision") == "block" else "allow",
        cost_usd=_price_cost_usd(observation, price),
        # Reasoning-token burn is the fixture's cost driver; carry output tokens.
        reasoning_tokens=_to_int(observation.get("output_tokens")),
        latency_ms=_to_int(observation.get("latency_ms")) if observation.get("latency_ms") is not None else 0,
        reply_text=str(reply) if reply else None,
    )


def _tool_calls(row: Mapping[str, Any]) -> tuple[ToolCall, ...]:
    """Read a row's ``tool_calls`` list into typed :class:`ToolCall`s (A2).

    Shape (all keys optional but ``name``)::

        "tool_calls": [
          {"name": "issue_refund", "arguments": {...},
           "schema_valid": true, "executed": true}
        ]
    """

    raw = row.get("tool_calls") or row.get("toolCalls") or ()
    if isinstance(raw, Mapping):
        raw = [raw]
    return tuple(ToolCall.from_dict(dict(c)) for c in raw if isinstance(c, Mapping))


def _safe_summary(row: Mapping[str, Any], incumbent: str, candidate: str) -> str:
    """A content-free row descriptor for the judge list.

    Real customer text (``row['summary']``) is used verbatim when present; the
    service layer is responsible for redacting it under a non-raw data mode. When
    absent, synthesise a descriptor from structural fields only (category /
    channel / the incumbent-vs-candidate split) so a redacted bench is still
    judgeable without leaking content.
    """

    text = str(row.get("summary") or "").strip()
    if text:
        return text
    models = row.get("models") or {}
    inc = (models.get(incumbent) or {}).get("decision", "?")
    cand = (models.get(candidate) or {}).get("decision", "?")
    bits = [str(row.get("category") or "decision")]
    if row.get("channel"):
        bits.append(str(row["channel"]))
    bits.append(f"current {inc} / candidate {cand}")
    return " · ".join(bits)


def bench_from_traces(
    rows: Sequence[Mapping[str, Any]],
    *,
    incumbent_id: str,
    candidate_id: str,
    slice_rules: Optional[SliceRules] = None,
    pricing: Optional[Mapping[str, Mapping[str, float]]] = None,
    committee: Optional[Scorer] = None,
) -> list[BenchCase]:
    """Project neutral paired trace rows into judgeable console ``BenchCase``s.

    Pairs each row's ``incumbent_id`` / ``candidate_id`` observations into a
    paired case, assigns a slice via ``slice_rules`` (sane defaults if omitted),
    and derives the ``adversarial_score`` weak signal — or, when a ``committee``
    scorer is supplied, uses its per-case reliability instead (evidence source
    (b), docs/73 §3.3). Rows missing either model or an unreadable decision are
    skipped, mirroring ``build_console_bench``. Deterministic (ordered by id).
    """

    rules = slice_rules or SliceRules()
    pricing = pricing or {}

    qualified: list[Mapping[str, Any]] = []
    for row in rows:
        models = row.get("models") or {}
        inc = models.get(incumbent_id)
        cand = models.get(candidate_id)
        if not isinstance(inc, Mapping) or not isinstance(cand, Mapping):
            continue
        if coerce_decision(inc.get("decision")) is None or coerce_decision(cand.get("decision")) is None:
            continue
        qualified.append(row)

    # Incumbent reasoning-token difficulty is normalised across the bench.
    inc_tokens = [_to_int((row["models"][incumbent_id]).get("output_tokens")) for row in qualified]
    rt_min = min(inc_tokens) if inc_tokens else 0
    rt_max = max(inc_tokens) if inc_tokens else 0
    span = (rt_max - rt_min) or 1

    cases: list[BenchCase] = []
    for row in sorted(qualified, key=lambda r: str(r["id"])):
        case_id = str(row["id"])
        inc_obs = row["models"][incumbent_id]
        cand_obs = row["models"][candidate_id]
        inc_record = _model_record(inc_obs, pricing.get(incumbent_id))
        cand_record = _model_record(cand_obs, pricing.get(candidate_id))
        tool_calls = _tool_calls(row)
        slice_id = rules.assign(side=row.get("side"), category=row.get("category"), tool_calls=tool_calls)
        disagree = inc_record.decision != cand_record.decision
        difficulty = (_to_int(inc_obs.get("output_tokens")) - rt_min) / span if rt_max > rt_min else 0.0
        score = derive_adversarial_score(
            slice_id=slice_id, disagree=disagree, difficulty=difficulty, jitter=_h01("s", case_id), rules=rules
        )
        cases.append(
            BenchCase(
                id=case_id,
                slice=slice_id,
                adversarial_score=round(score, 4),
                incumbent=inc_record,
                candidate=cand_record,
                category=str(row.get("category") or ""),
                channel=str(row.get("channel") or ""),
                summary=_safe_summary(row, incumbent_id, candidate_id),
                customer_message=(str(row["customer_message"]) if row.get("customer_message") else None),
                tool_calls=tool_calls,
            )
        )

    if committee is not None:
        cases = [replace(c, adversarial_score=round(_clamp01(committee.score(c)), 4)) for c in cases]
    return cases


def _clamp01(value: float) -> float:
    return max(_SCORE_FLOOR, min(_SCORE_CEIL, value))


def bench_from_otel(
    export: Mapping[str, Any],
    *,
    incumbent_id: str,
    candidate_id: str,
    decision_point: Optional[str] = None,
    slice_rules: Optional[SliceRules] = None,
    pricing: Optional[Mapping[str, Mapping[str, float]]] = None,
    committee: Optional[Scorer] = None,
    reader_options: Optional[Mapping[str, Any]] = None,
) -> list[BenchCase]:
    """Convenience: OTLP export -> paired rows -> ``BenchCase``s in one call."""

    rows = paired_rows_from_otel(export, decision_point=decision_point, **(reader_options or {}))
    return bench_from_traces(
        rows,
        incumbent_id=incumbent_id,
        candidate_id=candidate_id,
        slice_rules=slice_rules,
        pricing=pricing,
        committee=committee,
    )


def tool_archetype_for(case: BenchCase, archetypes: Sequence[str]) -> Optional[str]:
    """Which selected tool-call archetype (if any) this case exhibits (A2).

    Shape conformance is checked first: a call that fails the pinned schema is
    the more severe finding, so it never hides behind a no-op label.
    """

    selected = {str(a) for a in archetypes}
    if not case.tool_calls:
        return None
    if TOOL_CALL_SHAPE_SLICE in selected and case.has_tool_shape_violation():
        return TOOL_CALL_SHAPE_SLICE
    if TOOL_NO_OP_SLICE in selected and case.has_tool_no_op():
        return TOOL_NO_OP_SLICE
    return None


def apply_tool_archetypes(cases: Sequence[BenchCase], archetypes: Sequence[str]) -> list[BenchCase]:
    """Re-slice an existing bench onto the selected tool-call archetypes.

    Lets a policy adopt an A2 template without regenerating the bench: cases that
    carry ``tool_calls`` and exhibit the archetype move into its slice, and the
    existing constraint stage (:func:`~aver_policy.engine.guarded_slice_result`)
    scores them unchanged. Cases with no tool-call fields are untouched, so a
    bench that never recorded tool calls simply has an empty archetype slice —
    which the verdict then REFUSEs on for insufficient evidence, honestly.
    """

    selected = [a for a in TOOL_ARCHETYPES if a in {str(x) for x in archetypes}]
    if not selected:
        return list(cases)
    out: list[BenchCase] = []
    for case in cases:
        archetype = tool_archetype_for(case, selected)
        out.append(replace(case, slice=archetype) if archetype and case.slice != archetype else case)
    return out


def slice_distribution(cases: Sequence[BenchCase]) -> dict[str, int]:
    """Count cases per slice (deterministic ordering by first appearance)."""

    counts: dict[str, int] = {}
    for case in cases:
        counts[case.slice] = counts.get(case.slice, 0) + 1
    return counts
