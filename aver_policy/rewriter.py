"""aver_policy.rewriter — the pure revision-prompt construction (QA P2-1 / story 6).

Ports the X9 rewriter's prompt construction from ``scripts/port_loop.py``
(``build_rewriter_prompt`` / ``extract_role_prompt``) into a standalone,
service-free function so the Policy Console can ask a judge model to *draft a
suggested revision* from the divergent receipts. The service layer
(``stackcert_service.services.policy_revision``) wraps this with the managed judge
lane + budget cap (docs/74 CV4 pattern); this module is pure text (no clock, no
I/O, no service imports — the CV8 boundary the standalone-package test enforces).

The output is a SUGGESTION a human decides on — never auto-applied. The prompt is
built from structural receipt fields only (case summary + the two decisions), so
it carries no more than the bench already holds; a redacted bench yields a
content-free structural prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

# Cap the number of divergences shown to the judge (mirrors port_loop.MAX_EXAMPLES).
MAX_REVISION_EXAMPLES = 15
# The revised guidance should stay short — a concise instruction, not an essay.
SUGGESTION_MAX_CHARS = 900


@dataclass(frozen=True)
class RevisionExample:
    """One divergence the suggestion targets: the case, and the two decisions.

    ``approved`` is the decision under the owner's policy (``block`` / ``allow``
    from ``should_block``), not the incumbent model's decision. The assist revises
    guidance so the candidate matches the *policy*, not the old model.
    """

    input: str
    approved: str
    candidate: str


def build_revision_prompt(
    role_label: str,
    examples: Sequence[RevisionExample],
    *,
    gate_description: Optional[str] = None,
) -> str:
    """The user message asking a judge to draft a revised decision guidance.

    ``examples`` are divergences (the receipts the candidate got wrong under the
    owner's policy). The judge is asked to revise the INSTRUCTIONS so the
    candidate reproduces the approved decisions, may add a few short examples, and
    is explicitly warned NOT to over-block — so it cannot trivially "win" by
    blocking everything (ported from ``port_loop.build_rewriter_prompt``).
    """

    gate = gate_description or f"decide the {role_label} case the way your approved policy would"
    lines = [
        f"You are drafting a SUGGESTED revision to the decision guidance for '{role_label}'. "
        f"The candidate model must reproduce the approved decision on whether to {gate}.",
        "",
        "Below are cases where the candidate DISAGREED with the approved decision. "
        "For each: the case, the approved decision (the target), and the candidate's "
        "(wrong) decision.",
        "",
    ]
    for index, example in enumerate(list(examples)[:MAX_REVISION_EXAMPLES], start=1):
        lines.append(f"--- divergence {index} ---")
        lines.append(f"CASE: {example.input}")
        lines.append(f"APPROVED (target): {example.approved}")
        lines.append(f"CANDIDATE (wrong): {example.candidate}")
        lines.append("")
    lines.extend(
        [
            "Write REVISED decision guidance (a short instruction) that makes the "
            "candidate match the approved decisions on cases like these AND on the "
            "cases where the two already agree. State the policy as general "
            "STRUCTURAL guidance; you may include up to 3 short examples. Do NOT "
            "simply block everything — the approved policy also ALLOWS legitimate "
            f"cases, and blocking those would be a new error. Keep it concise: at "
            f"most ~{SUGGESTION_MAX_CHARS} characters.",
            "Output ONLY the revised guidance text, with no preamble, explanation, "
            "or markdown fences.",
        ]
    )
    return "\n".join(lines)


def revision_system_prompt() -> str:
    """The system message that frames the judge as an assistant drafting a
    suggestion the human decides on — never an auto-applied change."""

    return (
        "You revise decision-policy guidance. You propose a suggestion only; a "
        "human engineer decides whether to adopt it. Be precise and conservative."
    )


def extract_suggestion(content: Optional[str]) -> str:
    """Pull the suggested guidance out of a completion (strip fences/preamble).

    Ported from ``port_loop.extract_role_prompt`` — a model may wrap the answer in
    a markdown fence or a one-line preamble; both are stripped so the console shows
    the suggestion text alone."""

    if not content:
        return ""
    text = content.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    text = text.strip()
    lowered = text.lower()
    for preamble in (
        "here is the revised guidance:",
        "revised guidance:",
        "here is the revised policy:",
        "suggested revision:",
    ):
        if lowered.startswith(preamble):
            text = text[len(preamble) :].strip()
            break
    return text
