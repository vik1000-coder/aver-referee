"""aver_policy.variance — per-(role, model) flip-noise floor (docs/73 §3.5).

Productizes the X4 silent-update probe machinery (``scripts/model_probe_pack.py``
and ``scripts/real_model_divergence.py``'s ``noise`` command) into a reusable,
service-free profile: repeated identical calls on a probe set give a model's
intrinsic self-inconsistency floor ``p_noise`` with a Wilson CI. The verdict/
alarm path then counts a behavioural difference **only** where a paired flip
rate's lower bound exceeds the noise floor's upper bound (McNemar-style paired
treatment). An alarm therefore never fires inside a model's own measured churn —
X4's empirical lesson: reasoning models churn intrinsically (3/20 probes seconds
apart) while non-reasoning models were byte-stable over 10.6h, so per-model
calibration is mandatory, not optional.

CV8 boundary (``tests/test_aver_policy_is_standalone.py``): pure — no service
imports, only ``aver_policy.engine`` (which itself only reaches ``aver_cass.core``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Optional, Union

from aver_policy.engine import wilson_bounds

# A decision is 1 (block / never-event) or 0 (allow); ``None`` is an invalid or
# unparseable completion, dropped from the rate exactly like the X4 pack's
# per-probe ``valid`` flag.
Decision = Optional[int]

# Too few repeat-observations to assert a floor honestly: the caller renders this
# as a refusal-as-work-item ("not enough profile data yet"), never a fabricated 0.
MIN_PROBE_OBSERVATIONS = 8


def _coerce_decision(raw: object) -> Decision:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return 1 if raw else 0
    if isinstance(raw, str):
        token = raw.strip().lower()
        if token in ("block", "never_event", "1", "true"):
            return 1
        if token in ("allow", "0", "false"):
            return 0
        return None
    try:
        return 1 if int(raw) == 1 else 0
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class ProbeCaseResult:
    """Repeated identical-input decisions collected for one probe case."""

    case_id: str
    decisions: list[Decision]

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ProbeCaseResult":
        return cls(
            case_id=str(raw.get("case_id") or raw.get("id") or ""),
            decisions=[_coerce_decision(d) for d in (raw.get("decisions") or [])],  # type: ignore[union-attr]
        )

    def valid_decisions(self) -> list[int]:
        return [d for d in self.decisions if d is not None]

    def disagreements(self) -> int:
        """Valid decisions differing from this case's modal decision.

        This is the per-case numerator of ``real_model_divergence.py``'s
        ``self_inconsistency_rate = 1 - agreeing/n`` (``= disagreeing/n``). A case
        with fewer than two valid decisions has no pair that could disagree and
        contributes nothing.
        """

        valid = self.valid_decisions()
        if len(valid) < 2:
            return 0
        blocks = sum(valid)
        modal_count = max(blocks, len(valid) - blocks)
        return len(valid) - modal_count


ProbeResults = Iterable[Union[ProbeCaseResult, Mapping[str, object]]]


@dataclass(frozen=True)
class VarianceProfile:
    """The measured flip-noise floor for one (role, model)."""

    role_id: str
    model_id: str
    p_noise: float
    ci_low: float
    ci_high: float
    probe_n: int          # total counted repeat-observations (denominator)
    disagreements: int
    probe_cases: int      # cases with >= 2 valid decisions

    @property
    def band_ucb(self) -> float:
        """The established variance band ceiling — the Wilson upper on p_noise.

        The alarm/verdict gate compares an observed paired flip rate against this
        (never against the point estimate) so noisy small samples do not alarm.
        """

        return self.ci_high

    @property
    def sufficient(self) -> bool:
        """Enough repeat-observations to assert a floor (else a work-item)."""

        return self.probe_n >= MIN_PROBE_OBSERVATIONS

    def to_dict(self) -> dict:
        return {
            "role_id": self.role_id,
            "model_id": self.model_id,
            "p_noise": self.p_noise,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "probe_n": self.probe_n,
            "disagreements": self.disagreements,
            "probe_cases": self.probe_cases,
            "sufficient": self.sufficient,
        }


def variance_profile(
    role_id: str,
    model_id: str,
    probe_results: ProbeResults,
    *,
    z: float = 1.96,
) -> VarianceProfile:
    """Compute the flip-noise floor for a (role, model) from probe results.

    ``probe_results`` is an iterable of :class:`ProbeCaseResult` (or plain dicts
    with ``case_id`` + ``decisions``). ``p_noise`` is the total modal
    disagreements over the total counted repeat-observations, with a Wilson CI —
    generalizing ``real_model_divergence.py``'s single-case self-inconsistency
    rate across a stratified probe set (docs/73 §3.5).
    """

    disagreements = 0
    counted = 0
    probe_cases = 0
    for raw in probe_results:
        case = raw if isinstance(raw, ProbeCaseResult) else ProbeCaseResult.from_dict(raw)
        valid = case.valid_decisions()
        if len(valid) < 2:
            continue
        probe_cases += 1
        counted += len(valid)
        disagreements += case.disagreements()

    p_noise = (disagreements / counted) if counted else 0.0
    low, high = wilson_bounds(disagreements, counted, z) if counted else (0.0, 0.0)
    return VarianceProfile(
        role_id=role_id,
        model_id=model_id,
        p_noise=round(p_noise, 6),
        ci_low=round(low, 6),
        ci_high=round(high, 6),
        probe_n=counted,
        disagreements=disagreements,
        probe_cases=probe_cases,
    )


@dataclass(frozen=True)
class NoiseGate:
    """The McNemar-style paired verdict on an observed flip rate vs a noise band."""

    flips: int
    n: int
    observed_rate: float
    observed_lower: float
    observed_upper: float
    noise_ucb: float
    exceeds_band: bool    # the difference is real: alarm / count the regression
    applied: bool         # a measured band was available to gate against

    def to_dict(self) -> dict:
        return {
            "flips": self.flips,
            "n": self.n,
            "observed_rate": self.observed_rate,
            "observed_lower": self.observed_lower,
            "observed_upper": self.observed_upper,
            "noise_ucb": self.noise_ucb,
            "exceeds_band": self.exceeds_band,
            "applied": self.applied,
        }


def effective_noise_band(
    measured_ucb: Optional[float],
    variance_budget: Optional[float],
) -> tuple[float, str]:
    """The band the noise gate compares against, honoring an owner-set budget.

    ``variance_budget`` is the policy's deliberate flip-rate ceiling (docs/73 §2
    ``variance_budget``): the churn the owner tolerates on identical inputs. It
    can WIDEN the band — differences under the budget are within tolerance — but
    it can never NARROW it below the measured floor: a difference inside the
    model's own measured self-inconsistency is not attributable to the candidate,
    whatever the budget says (docs/73 §3.5 honesty rule; the McNemar-style gate
    never alarms inside measured churn). Returns ``(band, source)`` with source
    one of ``"none"`` (no floor at all), ``"measured"`` (the measured band
    governs), ``"budget"`` (the owner's budget governs — it is the wider, or the
    only, ceiling).
    """

    measured = None if measured_ucb is None else max(0.0, float(measured_ucb))
    budget = None if variance_budget is None else max(0.0, float(variance_budget))
    if measured is None and budget is None:
        return 0.0, "none"
    if budget is None:
        return float(measured or 0.0), "measured"
    if measured is None:
        return budget, "budget"
    if budget > measured:
        return budget, "budget"
    return measured, "measured"


def noise_floor_gate(
    *,
    flips: int,
    n: int,
    noise_ucb: float,
    z: float = 1.96,
) -> NoiseGate:
    """McNemar-style paired treatment (docs/73 §3.5).

    A behavioural difference counts only where the observed paired flip rate's
    lower Wilson bound exceeds the noise floor's upper bound. Below that, the
    difference sits inside the model's measured self-inconsistency band, so it is
    NOT an alarm and NOT a certified regression — the silent-update-alarm honesty
    story (docs/73 story 8).
    """

    band = max(0.0, float(noise_ucb))
    if n <= 0:
        return NoiseGate(
            flips=0,
            n=0,
            observed_rate=0.0,
            observed_lower=0.0,
            observed_upper=0.0,
            noise_ucb=round(band, 6),
            exceeds_band=False,
            applied=band > 0.0,
        )
    low, high = wilson_bounds(max(0, int(flips)), int(n), z)
    return NoiseGate(
        flips=int(flips),
        n=int(n),
        observed_rate=round(flips / n, 6),
        observed_lower=round(low, 6),
        observed_upper=round(high, 6),
        noise_ucb=round(band, 6),
        exceeds_band=low > band,
        applied=True,
    )


def safety_flip_count(
    incumbent: Mapping[str, Decision],
    candidate: Mapping[str, Decision],
    case_ids: Iterable[str],
) -> tuple[int, int]:
    """Directional block->allow flips over comparable cases.

    ``candidate`` allowing what ``incumbent`` blocked is the safety-relevant flip
    (allow->block is benign friction); a case where either side has no readable
    decision is excluded from the denominator. Returns ``(flips, comparable_n)`` —
    the pair the :func:`noise_floor_gate` consumes. Productizes
    ``real_model_divergence.flip_counts`` as a pure helper.
    """

    flips = 0
    comparable = 0
    for case_id in case_ids:
        base = incumbent.get(case_id)
        cand = candidate.get(case_id)
        if base is None or cand is None:
            continue
        comparable += 1
        if base == 1 and cand == 0:
            flips += 1
    return flips, comparable
