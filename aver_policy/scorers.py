"""aver_policy.scorers — per-case weak signals for un-judged cases (docs/73 §3.3).

A ``Scorer`` maps a bench case to a weak signal in (0, 1): higher = more
adversarial-looking. Under asymmetry ``r`` an un-judged case is treated as
should-block iff its signal clears ``tau(r)`` (see :func:`engine.should_block`).
A judged case always overrides the signal.

Two implementations:

* :class:`FixedSignalScorer` — reads the labelled ``adversarial_score`` already
  on the case (the current bench behaviour). This is the OSS default.
* :class:`CommitteeScorer` — evidence source (b) in docs/73 §3.3: a
  committee-calibrated reliability per case. To keep this package standalone and
  liftable (docs/74 CV8 — ZERO imports from the service *or* the CASS research
  package), the committee's calibrated per-case signals are **injected**. The
  hosted side (``stackcert``'s ``CommitteeEvidenceEngine``) builds the mapping
  and hands it in; the OSS engine only defines the interface.

The verdict path stays the two-stage loss; a scorer only supplies the signal for
un-judged cases. Console verdicts never route through ``build_certificate`` —
that is CASS-certified issuance (docs/47), a different evidence family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, runtime_checkable

from aver_policy.engine import BenchCase


@runtime_checkable
class Scorer(Protocol):
    """Produces a per-case weak signal in (0, 1)."""

    def score(self, case: BenchCase) -> float: ...


def _clamp01(value: float) -> float:
    return max(1e-9, min(1 - 1e-9, value))


@dataclass(frozen=True)
class FixedSignalScorer:
    """Reads the labelled ``adversarial_score`` on the case (bundle behaviour)."""

    def score(self, case: BenchCase) -> float:
        return _clamp01(case.adversarial_score)


@dataclass(frozen=True)
class CommitteeScorer:
    """Committee-calibrated per-case reliability (docs/73 §3.3 evidence source b).

    Dependency-injected so the OSS engine carries no committee code. Provide
    either a ``signals`` mapping (case_id -> signal in (0,1)) or a ``signal_fn``
    callable; ``default`` is used for cases the committee did not score.
    """

    signals: Mapping[str, float] | None = None
    signal_fn: Callable[[BenchCase], float] | None = None
    default: float = 0.5

    def score(self, case: BenchCase) -> float:
        if self.signal_fn is not None:
            return _clamp01(self.signal_fn(case))
        if self.signals is not None and case.id in self.signals:
            return _clamp01(self.signals[case.id])
        return _clamp01(self.default)


def apply_scorer(cases: list[BenchCase], scorer: Scorer) -> list[BenchCase]:
    """Return copies of ``cases`` whose ``adversarial_score`` is the scorer's
    signal — so a committee (or any) signal flows through the existing
    ``adversarial_score``-based math with no verdict-path change."""

    from dataclasses import replace

    return [replace(c, adversarial_score=scorer.score(c)) for c in cases]
