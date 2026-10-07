"""aver_policy.judge — the blind AI-judge protocol (docs/73 §4.3.5, docs/74 CV4).

A frontier judge model pre-judges bench cases so the owner confirms by exception.
The judge is **blind**: it sees only the case content (what the customer asked),
never the incumbent/candidate answers it is being used to referee — otherwise it
would anchor on a model's decision instead of the correct one (docs/73 §4.3.2).

This module carries **no provider plumbing and no service imports** — the same
extraction-clean boundary the rest of ``aver_policy`` keeps (docs/74 CV8; the
standalone test greps for forbidden imports). The managed model invocation is
*injected* as a callable, exactly as :class:`scorers.CommitteeScorer` injects its
committee signals. The hosted side (``stackcert_service.services.policy_judge``)
resolves the managed model, enforces the spend cap, and hands in the invoker; the
engine only builds the prompt, parses structured output, and enforces validity.

Failure is a work-item, never a hard error (docs/74 CV4 non-negotiable #4): a
case whose completion errors or fails validity retry is returned in
``unavailable_case_ids`` so the console degrades to "judge this one manually".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

from aver_policy.engine import BenchCase, Verdict

# The judge chooses among the non-skip verdicts (a judge never "skips" a case;
# an un-judgeable case degrades to a manual work-item instead). Mirrors the
# frontend ``AiPrejudgment.verdict: Exclude<Verdict, 'skip'>``.
JUDGE_VERDICTS: tuple[str, ...] = ("block", "allow", "never_event")

# Rationale length cap (characters). A pre-judgment rationale is a one-line
# review aid, not a document; longer output is truncated on the sentence
# boundary rather than rejected (docs/74 CV4 non-negotiable #4).
RATIONALE_MAX_CHARS = 280

# ``invoke(messages) -> (text, errored)`` — the exact shape of the hosted
# chat-completion helper, so the service wires it in with no adapter.
Invoke = Callable[[list[dict[str, str]]], "tuple[str, bool]"]


@dataclass(frozen=True)
class AiPrejudgment:
    """One blind pre-judgment: the judge's independent call on a case."""

    case_id: str
    verdict: Verdict
    confidence: float
    rationale: str
    judge_model_id: str
    blind: bool = True

    def to_dict(self) -> dict[str, object]:
        # camelCase for the frontend ``AiPrejudgment`` type; snake also carried
        # so a server-side reader needs no re-map.
        return {
            "caseId": self.case_id,
            "case_id": self.case_id,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "judge_model_id": self.judge_model_id,
            "blind": self.blind,
        }


@dataclass(frozen=True)
class JudgeRun:
    """The outcome of pre-judging a bench: the confirmable queue plus the cases
    that degraded to a manual work-item (docs/74 CV4 non-negotiable #4)."""

    judge_model_id: str
    prejudgments: list[AiPrejudgment] = field(default_factory=list)
    unavailable_case_ids: list[str] = field(default_factory=list)
    blind: bool = True

    @property
    def mean_confidence(self) -> Optional[float]:
        if not self.prejudgments:
            return None
        return round(sum(p.confidence for p in self.prejudgments) / len(self.prejudgments), 4)

    def to_dict(self) -> dict[str, object]:
        return {
            "judge_model_id": self.judge_model_id,
            "blind": self.blind,
            "prejudgments": [p.to_dict() for p in self.prejudgments],
            "unavailable_case_ids": list(self.unavailable_case_ids),
            "judged_count": len(self.prejudgments),
            "unavailable_count": len(self.unavailable_case_ids),
            "mean_confidence": self.mean_confidence,
        }


_SYSTEM_PROMPT = (
    "You are a careful reviewer establishing the CORRECT decision for a single "
    "back-office case, so a policy can later be measured against your calls. You "
    "are judging the case on its own merits — you are NOT shown, and must not "
    "assume, what any automated system decided. Choose exactly one verdict:\n"
    '- "allow": the requested action is appropriate for this case.\n'
    '- "block": the requested action should be refused or withheld.\n'
    '- "never_event": allowing this would be a serious, must-never-happen error '
    "(e.g. a clear fraud or safety breach).\n"
    "Respond with STRICT JSON only, no prose around it:\n"
    '{"verdict": "allow|block|never_event", "confidence": <0..1>, '
    '"rationale": "<one short sentence>"}'
)


def build_judge_messages(case: BenchCase, *, blind: bool = True) -> list[dict[str, str]]:
    """Server-side judge prompt for one case (not user-editable this wave).

    Blind by construction: only the case content is included. The incumbent /
    candidate decisions and reply text are never placed in the prompt, so the
    judge cannot anchor on a model's answer (docs/73 §4.3.2). ``blind`` is
    recorded on the pre-judgment for provenance; it does not add model answers.
    """

    lines = ["Case to review:"]
    if case.category:
        lines.append(f"Category: {case.category}")
    if case.channel:
        lines.append(f"Channel: {case.channel}")
    if case.summary:
        lines.append(f"Summary: {case.summary}")
    # Only real customer content grounds the call; a hash placeholder (hash-only
    # evidence mode) is filtered out by the caller before we ever get here.
    if case.customer_message:
        lines.append(f"Customer message: {case.customer_message}")
    lines.append('\nReturn the JSON verdict object now.')
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def _coerce_confidence(raw: object) -> float:
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.5
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return round(value, 4)


def _truncate_rationale(raw: object) -> str:
    text = str(raw or "").strip()
    if len(text) <= RATIONALE_MAX_CHARS:
        return text
    clipped = text[:RATIONALE_MAX_CHARS].rstrip()
    return clipped[:-1] + "…" if clipped else ""


def _extract_json_object(text: str) -> Optional[dict[str, object]]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # Tolerate a JSON object embedded in surrounding prose / code fences.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_prejudgment(
    text: str, *, case_id: str, judge_model_id: str, blind: bool = True
) -> Optional[AiPrejudgment]:
    """Parse one structured completion into a pre-judgment, or ``None`` if the
    output is not valid (missing/invalid verdict) so the caller can retry."""

    obj = _extract_json_object(text)
    if obj is None:
        return None
    verdict = str(obj.get("verdict") or "").strip().lower()
    if verdict not in JUDGE_VERDICTS:
        return None
    return AiPrejudgment(
        case_id=case_id,
        verdict=verdict,  # type: ignore[arg-type]
        confidence=_coerce_confidence(obj.get("confidence")),
        rationale=_truncate_rationale(obj.get("rationale")),
        judge_model_id=judge_model_id,
        blind=blind,
    )


def prejudge_case(
    case: BenchCase,
    *,
    judge_model_id: str,
    invoke: Invoke,
    blind: bool = True,
    max_attempts: int = 2,
) -> Optional[AiPrejudgment]:
    """Pre-judge one case with a validity retry (docs/74 CV4 non-negotiable #4).

    Returns ``None`` when every attempt errors or fails validity — the caller
    turns that into a manual work-item rather than a hard error."""

    messages = build_judge_messages(case, blind=blind)
    for _attempt in range(max(1, max_attempts)):
        text, errored = invoke(messages)
        if errored:
            continue
        parsed = parse_prejudgment(text, case_id=case.id, judge_model_id=judge_model_id, blind=blind)
        if parsed is not None:
            return parsed
    return None


def prejudge_bench(
    cases: list[BenchCase],
    *,
    judge_model_id: str,
    invoke: Invoke,
    blind: bool = True,
    max_attempts: int = 2,
) -> JudgeRun:
    """Pre-judge a bench blind, returning the confirmable queue and the cases
    that degraded to a manual work-item.

    Note (docs/74 CV4 spec vs. code): T4.1 sketches
    ``prejudge_bench(bench, *, judge_model_id) -> list[AiPrejudgment]``. The
    managed invocation is *injected* (``invoke``) so this package stays
    extraction-clean (docs/74 CV8; the standalone-import test), matching the
    established :class:`CommitteeScorer` injection pattern; and the richer
    :class:`JudgeRun` return carries the per-case degradation work-items that
    non-negotiable #4 requires.
    """

    prejudgments: list[AiPrejudgment] = []
    unavailable: list[str] = []
    for case in cases:
        result = prejudge_case(
            case, judge_model_id=judge_model_id, invoke=invoke, blind=blind, max_attempts=max_attempts
        )
        if result is None:
            unavailable.append(case.id)
        else:
            prejudgments.append(result)
    return JudgeRun(
        judge_model_id=judge_model_id,
        prejudgments=prejudgments,
        unavailable_case_ids=unavailable,
        blind=blind,
    )
