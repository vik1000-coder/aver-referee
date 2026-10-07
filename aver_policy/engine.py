"""aver_policy.engine — the Policy Console referee math (docs/73 §3).

A dependency-free, deterministic Python port of ``web/src/lib/policyEngine.ts``.
Everything the console shows resolves through here so the same numbers are
reproducible in tests, in the UI, in the ``scripts/build_console_bench.py``
reference bundle, and — via CV1 — on the server (``stackcert_service`` calls
this module; the module never imports the service, docs/74 CV8 boundary).

The object being judged is the owner's POLICY, not a ground truth. Error
probabilities come from owner judgments (gold anchors) generalised to un-judged
cases through a labelled weak signal (``adversarial_score``). Under an asymmetry
``r`` a rational owner blocks a case iff its score clears the decision threshold
``tau(r) = 1 / (1 + r)``; a judged case overrides the signal outright.

Determinism: no clock, no I/O, no RNG except the seeded ``mulberry32`` bootstrap
in :func:`loss_difference_interval`, which is ported bit-for-bit from the
TypeScript engine so the two produce identical intervals.

RESEARCH PREVIEW — synthetic example data. This computes an advisory verdict,
never a certification or a safety guarantee.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:  # pragma: no cover - annotations only, no runtime cycle
    from aver_policy.repeated import ActionResult, Cure, RegimeReadout

# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #

Decision = Literal["block", "allow"]
Verdict = Literal["block", "allow", "never_event", "skip"]
ModelId = Literal["incumbent", "candidate"]
JudgmentProvenance = Literal["human_direct", "human_confirmed", "ai_proposed"]
SliceState = Literal["pass", "refuse_insufficient", "refuse_straddle", "block"]
VerdictState = Literal["pass", "review", "block", "refuse"]
MethodVersion = Literal["decision_policy_v1", "decision_policy_v2", "decision_policy_v3"]
Preference = Literal["better", "worse", "inconclusive"]

METHOD_VERSION_V1: MethodVersion = "decision_policy_v1"
METHOD_VERSION_V2: MethodVersion = "decision_policy_v2"
# docs/103: repeated-run evidence only. v1 and v2 bodies are untouched.
METHOD_VERSION_V3: MethodVersion = "decision_policy_v3"

# docs/102 §5 frozen confidence → z table. Copied into policyEngine.ts; do not
# call SciPy. When confidence == 0.95, z is exactly 1.96 (bench oracle).
CONFIDENCE_Z_TABLE: tuple[tuple[float, float], ...] = (
    (0.80, 1.2815515655446004),
    (0.85, 1.4395314709384563),
    (0.90, 1.6448536269514722),
    (0.95, 1.96),
    (0.99, 2.5758293035489004),
)

_UNSUPPORTED_CLAUSE_ALIASES: dict[str, str] = {
    "budget_latency_p95_ms": "budget_latency_p95_ms",
    "budgetLatencyP95Ms": "budget_latency_p95_ms",
}


def z_from_confidence(confidence: float) -> float:
    """Map a policy confidence in [0.80, 0.99] to z (docs/102 §5).

    Exact table points return the frozen z. Values between points linearly
    interpolate in confidence. Inputs outside the table clamp to the end rows.
    ``confidence == 0.95`` is special-cased so the bench oracle stays 1.96.
    """

    if confidence == 0.95:
        return 1.96
    if confidence <= CONFIDENCE_Z_TABLE[0][0]:
        return CONFIDENCE_Z_TABLE[0][1]
    for i in range(len(CONFIDENCE_Z_TABLE) - 1):
        c0, z0 = CONFIDENCE_Z_TABLE[i]
        c1, z1 = CONFIDENCE_Z_TABLE[i + 1]
        if confidence <= c1:
            if confidence == c1:
                return z1
            t = (confidence - c0) / (c1 - c0)
            return z0 + t * (z1 - z0)
    return CONFIDENCE_Z_TABLE[-1][1]

# Weight each provenance contributes to the asymmetry fit (docs/73 §4.3.5).
# AI-proposed evidence accepted only in bulk counts at half weight until a human
# confirms it individually; human-authored and human-confirmed evidence counts
# fully. This is what makes the provenance-weighted r honest about its base.
PROVENANCE_WEIGHT: dict[str, float] = {
    "human_direct": 1.0,
    "human_confirmed": 1.0,
    "ai_proposed": 0.5,
}
DEFAULT_PROVENANCE: JudgmentProvenance = "human_direct"

# Fixed logistic decisiveness prior for the r fit (see module docstring).
FIT_SLOPE_K = 12
# Minimum informative judgments before r may be fitted (docs/73 §3.4).
MIN_FIT_JUDGMENTS = 6


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float


@dataclass(frozen=True)
class ModelRecord:
    decision: Decision
    cost_usd: float = 0.0
    reasoning_tokens: int = 0
    latency_ms: int = 0
    reply_text: Optional[str] = None
    # v2 hard-ceiling missing-measurement: False when the payload omitted cost.
    # Direct constructors default True so existing fixtures stay cost-present.
    cost_present: bool = True

    @classmethod
    def from_dict(cls, raw: dict) -> "ModelRecord":
        if "costUsd" in raw:
            raw_cost = raw.get("costUsd")
        elif "cost_usd" in raw:
            raw_cost = raw.get("cost_usd")
        else:
            raw_cost = None
            cost_present = False
        if "costUsd" in raw or "cost_usd" in raw:
            cost_present = raw_cost is not None
        if "costPresent" in raw:
            cost_present = bool(raw.get("costPresent"))
        elif "cost_present" in raw:
            cost_present = bool(raw.get("cost_present"))
        return cls(
            decision=str(raw["decision"]),  # type: ignore[arg-type]
            cost_usd=float(raw_cost or 0.0) if raw_cost is not None else 0.0,
            reasoning_tokens=int(raw.get("reasoningTokens", raw.get("reasoning_tokens", 0)) or 0),
            latency_ms=int(raw.get("latencyMs", raw.get("latency_ms", 0)) or 0),
            reply_text=raw.get("replyText", raw.get("reply_text")),
            cost_present=cost_present,
        )


@dataclass(frozen=True)
class ToolCall:
    """One agent tool invocation recorded on a bench case (docs/79 §4 A2).

    ``schema_valid`` is the pinned-tool-schema validation outcome the trace
    carries (``None`` = the export did not say, which is never treated as a
    violation). ``executed`` says whether the invocation actually ran. Empty
    ``arguments`` with ``executed`` false/absent is the *tool no-op* archetype.
    """

    name: str = ""
    arguments: dict = field(default_factory=dict)
    schema_valid: Optional[bool] = None
    executed: Optional[bool] = None

    @classmethod
    def from_dict(cls, raw: dict) -> "ToolCall":
        arguments = raw.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {} if arguments in (None, "", [], {}) else {"_": arguments}
        return cls(
            name=str(raw.get("name") or ""),
            arguments=dict(arguments),
            schema_valid=_optional_bool(raw.get("schemaValid", raw.get("schema_valid"))),
            executed=_optional_bool(raw.get("executed")),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "arguments": dict(self.arguments),
            "schema_valid": self.schema_valid,
            "executed": self.executed,
        }

    @property
    def shape_violation(self) -> bool:
        """The invocation failed validation against the pinned tool schema."""

        return self.schema_valid is False

    @property
    def no_op(self) -> bool:
        """The tool was invoked but did nothing: empty arguments, or never ran."""

        return (not self.arguments) or self.executed is False


def _first_present(raw: dict, keys: tuple[str, ...], default: float) -> float:
    """First key actually present in ``raw``, else ``default`` (0.0 is a value)."""

    for key in keys:
        if key in raw and raw[key] is not None:
            return float(raw[key])
    return default


def _optional_bool(value: object) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in ("true", "1", "yes", "valid"):
            return True
        if token in ("false", "0", "no", "invalid"):
            return False
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    return None


@dataclass(frozen=True)
class BenchCase:
    id: str
    slice: str
    adversarial_score: float
    incumbent: ModelRecord
    candidate: ModelRecord
    category: str = ""
    channel: str = ""
    summary: str = ""
    customer_message: Optional[str] = None
    tool_calls: tuple[ToolCall, ...] = ()

    def model(self, model_id: ModelId) -> ModelRecord:
        return self.incumbent if model_id == "incumbent" else self.candidate

    def has_tool_shape_violation(self) -> bool:
        return any(call.shape_violation for call in self.tool_calls)

    def has_tool_no_op(self) -> bool:
        return any(call.no_op for call in self.tool_calls)

    @classmethod
    def from_dict(cls, raw: dict) -> "BenchCase":
        raw_calls = raw.get("toolCalls", raw.get("tool_calls")) or ()
        return cls(
            id=str(raw["id"]),
            slice=str(raw["slice"]),
            adversarial_score=float(raw.get("adversarialScore", raw.get("adversarial_score", 0.0)) or 0.0),
            incumbent=ModelRecord.from_dict(raw["incumbent"]),
            candidate=ModelRecord.from_dict(raw["candidate"]),
            category=str(raw.get("category", "")),
            channel=str(raw.get("channel", "")),
            summary=str(raw.get("summary", "")),
            customer_message=raw.get("customerMessage", raw.get("customer_message")),
            tool_calls=tuple(ToolCall.from_dict(c) for c in raw_calls if isinstance(c, dict)),
        )


@dataclass(frozen=True)
class GuardedConstraint:
    slice: str
    max_violation_rate: float
    min_evidence: int

    @classmethod
    def from_dict(cls, raw: dict) -> "GuardedConstraint":
        return cls(
            slice=str(raw["slice"]),
            max_violation_rate=float(raw.get("maxViolationRate", raw.get("max_violation_rate"))),
            min_evidence=int(raw.get("minEvidence", raw.get("min_evidence"))),
        )


# docs/103 §§3-5 defaults. Single source of truth: aver_policy.repeated and the
# TypeScript twin both read these, and schema.py validates against them.
DEFAULT_DECIDABILITY_THRESHOLD = 0.8
DEFAULT_MAX_UNDECIDED_FRACTION = 0.25
GUARDED_ACTION_DEFAULT_TOLERANCE = 0.05
GUARDED_ACTION_DEFAULT_MIN_EVIDENCE = 5


@dataclass(frozen=True)
class GuardedAction:
    """One ratified guarded action for decision-policy v3 (docs/103 §§4-5).

    ``tolerance`` is Clause A's delta on the modal switch rate; ``harm_tolerance``
    is Clause B's, and ``None`` means "inherit ``tolerance``".
    ``harm_clause_active`` is False only when the ratified policy set
    ``harm_tolerance: null`` — the explicit opt-out in docs/103 §5. The two are
    separate fields because "not stated" and "deliberately switched off" are
    different facts, and only the second may silence a clause.
    """

    action: str
    tolerance: float = GUARDED_ACTION_DEFAULT_TOLERANCE
    harm_tolerance: Optional[float] = None
    harm_clause_active: bool = True
    min_evidence: int = GUARDED_ACTION_DEFAULT_MIN_EVIDENCE

    @property
    def harm_delta(self) -> Optional[float]:
        """Clause B's effective tolerance, or ``None`` when Clause B is inactive."""

        if not self.harm_clause_active:
            return None
        return self.tolerance if self.harm_tolerance is None else self.harm_tolerance

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "tolerance": self.tolerance,
            "harm_tolerance": self.harm_delta,
            "min_evidence": self.min_evidence,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "GuardedAction":
        tolerance_raw = raw.get("tolerance")
        min_evidence_raw = raw.get("minEvidence", raw.get("min_evidence"))
        has_harm = "harmTolerance" in raw or "harm_tolerance" in raw
        harm_raw = raw.get("harmTolerance", raw.get("harm_tolerance"))
        return cls(
            action=str(raw.get("action") or ""),
            tolerance=GUARDED_ACTION_DEFAULT_TOLERANCE
            if tolerance_raw is None
            else float(tolerance_raw),
            harm_tolerance=None if harm_raw is None else float(harm_raw),
            harm_clause_active=not (has_harm and harm_raw is None),
            min_evidence=GUARDED_ACTION_DEFAULT_MIN_EVIDENCE
            if min_evidence_raw is None
            else int(min_evidence_raw),
        )


@dataclass(frozen=True)
class Policy:
    """The stage-2 policy the verdict scores against.

    ``r`` is the error-cost asymmetry c(false_allow) / c(false_block).
    """

    r: float
    guarded: list[GuardedConstraint] = field(default_factory=list)
    slice_weights: dict[str, float] = field(default_factory=dict)
    cost_weight: float = 0.0
    confidence: float = 0.95
    budget_cost_ceiling_usd: float = 0.0
    method_version: MethodVersion = METHOD_VERSION_V1
    absolute_loss_ceiling: Optional[float] = None
    require_improvement: bool = False
    variance_budget: Optional[float] = None
    unsupported_clauses: list[str] = field(default_factory=list)
    # docs/103 v3 fields. Inert on v1/v2: nothing outside the v3 path reads them.
    decidability_threshold: float = DEFAULT_DECIDABILITY_THRESHOLD
    max_undecided_fraction: float = DEFAULT_MAX_UNDECIDED_FRACTION
    guarded_actions: list[GuardedAction] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict) -> "Policy":
        method_raw = raw.get("methodVersion", raw.get("method_version"))
        # docs/103 §7: the default for a payload with no field stays v1.
        if method_raw == METHOD_VERSION_V3:
            method_version: MethodVersion = METHOD_VERSION_V3
        elif method_raw == METHOD_VERSION_V2:
            method_version = METHOD_VERSION_V2
        else:
            method_version = METHOD_VERSION_V1
        abs_raw = raw.get("absoluteLossCeiling", raw.get("absolute_loss_ceiling"))
        var_raw = raw.get("varianceBudget", raw.get("variance_budget"))
        unsupported = [str(x) for x in (raw.get("unsupportedClauses", raw.get("unsupported_clauses")) or [])]
        for key, canon in _UNSUPPORTED_CLAUSE_ALIASES.items():
            if raw.get(key) is not None and canon not in unsupported:
                unsupported.append(canon)
        return cls(
            r=float(raw["r"]),
            guarded=[GuardedConstraint.from_dict(g) for g in raw.get("guarded", [])],
            slice_weights={str(k): float(v) for k, v in (raw.get("sliceWeights", raw.get("slice_weights")) or {}).items()},
            cost_weight=float(raw.get("costWeight", raw.get("cost_weight", 0.0)) or 0.0),
            confidence=float(raw.get("confidence", 0.95) or 0.95),
            budget_cost_ceiling_usd=float(
                raw.get("budgetCostCeilingUsd", raw.get("budget_cost_ceiling_usd", 0.0)) or 0.0
            ),
            method_version=method_version,
            absolute_loss_ceiling=None if abs_raw is None else float(abs_raw),
            require_improvement=bool(raw.get("requireImprovement", raw.get("require_improvement", False))),
            variance_budget=None if var_raw is None else float(var_raw),
            unsupported_clauses=unsupported,
            decidability_threshold=float(
                raw.get(
                    "decidabilityThreshold",
                    raw.get("decidability_threshold", DEFAULT_DECIDABILITY_THRESHOLD),
                )
                or DEFAULT_DECIDABILITY_THRESHOLD
            ),
            max_undecided_fraction=float(
                _first_present(
                    raw,
                    ("maxUndecidedFraction", "max_undecided_fraction"),
                    DEFAULT_MAX_UNDECIDED_FRACTION,
                )
            ),
            guarded_actions=[
                GuardedAction.from_dict(a)
                for a in (raw.get("guardedActions", raw.get("guarded_actions")) or [])
            ],
        )


@dataclass(frozen=True)
class Judgment:
    case_id: str
    verdict: Verdict
    provenance: JudgmentProvenance = DEFAULT_PROVENANCE

    @classmethod
    def from_dict(cls, raw: dict) -> "Judgment":
        return cls(
            case_id=str(raw.get("caseId", raw.get("case_id"))),
            verdict=str(raw["verdict"]),  # type: ignore[arg-type]
            provenance=str(raw.get("provenance") or DEFAULT_PROVENANCE),  # type: ignore[arg-type]
        )


# --------------------------------------------------------------------------- #
# Wilson score interval (z = 1.96, matching policyEngine.ts exactly)
# --------------------------------------------------------------------------- #


def wilson_bounds(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """(lower, upper) Wilson score bounds.

    Mirrors ``policyEngine.ts:wilsonInterval`` and the bench mirror's ``_wilson``
    — z is hardcoded to 1.96 (NOT ``aver_cass.core.wilson_interval``'s
    ``z_score(0.95) = 1.95996…``) so the numbers reproduce the TS engine and the
    checked-in reference bundle bit-for-bit.
    """

    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + (z * z) / n
    centre = (p + (z * z) / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Interval:
    lower, upper = wilson_bounds(successes, n, z)
    return Interval(lower=lower, upper=upper)


def evidence_needed_for_rate(alpha: float, z: float = 1.96) -> float:
    """Smallest n at which successes=0 out of n yields Wilson upper <= alpha."""

    n = 1
    while n <= 100000:
        if wilson_bounds(0, n, z)[1] <= alpha:
            return n
        n += 1
    return math.inf


# --------------------------------------------------------------------------- #
# Asymmetry fit (MLE on a logistic link, fixed slope)
# --------------------------------------------------------------------------- #


def tau_of(r: float) -> float:
    return 1.0 / (1.0 + r)


# Fit grid: r in [0.5, 30.0] step 0.05, matching policyEngine.ts fitGrid. The
# bench mirror historically stopped at 29.95; the extra 30.0 endpoint is
# unreachable by any realistic fit (argmin sits near r=3) so the reference
# bundle is unchanged, and the grid now matches the TS engine exactly.
_FIT_GRID: tuple[float, ...] = tuple(round(0.5 + i * 0.05, 2) for i in range(0, 591))


def _neg_log_lik(inputs: list[tuple[float, int, float]], r: float, k: int = FIT_SLOPE_K) -> float:
    """``inputs`` are (score, blocked, weight). Mirror of ``negLogLik``."""

    tau = tau_of(r)
    total = 0.0
    for score, blocked, weight in inputs:
        p = 1.0 / (1.0 + math.exp(-k * (score - tau)))
        p = min(1 - 1e-9, max(1e-9, p))
        total -= weight * (blocked * math.log(p) + (1 - blocked) * math.log(1 - p))
    return total


def fit_r(inputs: list[tuple[float, int, float]]) -> tuple[float, float, float]:
    """Grid MLE for r with a likelihood-ratio CI: returns (r, lo, hi).

    ``inputs`` are (score, blocked, weight); weight 1 recovers the flat fit.
    Mirror of ``policyEngine.ts:fitGrid`` (and the bench mirror's ``_fit_r``).
    """

    best = _FIT_GRID[0]
    best_nll = _neg_log_lik(inputs, best)
    for r in _FIT_GRID:
        nll = _neg_log_lik(inputs, r)
        if nll < best_nll:
            best_nll = nll
            best = r
    lo = best
    hi = best
    for r in _FIT_GRID:
        if _neg_log_lik(inputs, r) - best_nll <= 1.92:  # chi2(1)/2 LR band
            lo = min(lo, r)
            hi = max(hi, r)
    return best, lo, hi


@dataclass(frozen=True)
class ProvenanceSplit:
    human_direct: int
    human_confirmed: int
    ai_proposed: int
    total: int
    ai_originated: int
    ai_proposed_share: float

    def to_dict(self) -> dict:
        return {
            "human_direct": self.human_direct,
            "human_confirmed": self.human_confirmed,
            "ai_proposed": self.ai_proposed,
            "total": self.total,
            "ai_originated": self.ai_originated,
            "ai_proposed_share": self.ai_proposed_share,
        }


def provenance_split(judgments: list[Judgment]) -> ProvenanceSplit:
    """Count informative judgments by provenance (docs/73 §4.3.5 disclosure)."""

    counts = {"human_direct": 0, "human_confirmed": 0, "ai_proposed": 0}
    for j in judgments:
        if j.verdict == "skip":
            continue
        counts[j.provenance or DEFAULT_PROVENANCE] += 1
    total = counts["human_direct"] + counts["human_confirmed"] + counts["ai_proposed"]
    return ProvenanceSplit(
        human_direct=counts["human_direct"],
        human_confirmed=counts["human_confirmed"],
        ai_proposed=counts["ai_proposed"],
        total=total,
        ai_originated=counts["human_confirmed"] + counts["ai_proposed"],
        ai_proposed_share=(counts["ai_proposed"] / total) if total else 0.0,
    )


@dataclass(frozen=True)
class AsymmetryFit:
    fitted: bool
    r: Optional[float]
    ci: Optional[tuple[float, float]]
    informative_count: int
    block_count: int
    allow_count: int
    provenance: ProvenanceSplit
    r_flat: Optional[float]
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "fitted": self.fitted,
            "r": self.r,
            "ci": list(self.ci) if self.ci is not None else None,
            "informative_count": self.informative_count,
            "block_count": self.block_count,
            "allow_count": self.allow_count,
            "provenance": self.provenance.to_dict(),
            "r_flat": self.r_flat,
            "reason": self.reason,
        }


def _fit_inputs(
    judgments: list[Judgment], score_by_id: dict[str, float], weighted: bool
) -> list[tuple[float, int, float]]:
    inputs: list[tuple[float, int, float]] = []
    for j in judgments:
        if j.verdict == "skip":
            continue
        score = score_by_id.get(j.case_id)
        if score is None:
            continue
        weight = PROVENANCE_WEIGHT[j.provenance or DEFAULT_PROVENANCE] if weighted else 1.0
        inputs.append((score, 0 if j.verdict == "allow" else 1, weight))
    return inputs


def fit_asymmetry(judgments: list[Judgment], cases: list[BenchCase]) -> AsymmetryFit:
    """Provenance-weighted MLE fit of the asymmetry r (docs/73 §3.4, §4.3.5).

    Refuses below ``MIN_FIT_JUDGMENTS`` informative judgments or when the
    judgments are one-sided (block-only / allow-only), because r is then
    unidentified. ``r_flat`` reports the unweighted estimate so the console can
    show how much the provenance mix moved r.
    """

    score_by_id = {c.id: c.adversarial_score for c in cases}
    inputs = _fit_inputs(judgments, score_by_id, True)
    flat_inputs = _fit_inputs(judgments, score_by_id, False)
    block_count = sum(1 for _s, b, _w in inputs if b == 1)
    allow_count = len(inputs) - block_count
    split = provenance_split(judgments)

    if len(inputs) < MIN_FIT_JUDGMENTS:
        remaining = MIN_FIT_JUDGMENTS - len(inputs)
        return AsymmetryFit(
            fitted=False,
            r=None,
            ci=None,
            informative_count=len(inputs),
            block_count=block_count,
            allow_count=allow_count,
            provenance=split,
            r_flat=None,
            reason=f"judge {remaining} more case{'' if remaining == 1 else 's'} to fit the asymmetry",
        )
    if block_count == 0 or allow_count == 0:
        return AsymmetryFit(
            fitted=False,
            r=None,
            ci=None,
            informative_count=len(inputs),
            block_count=block_count,
            allow_count=allow_count,
            provenance=split,
            r_flat=None,
            reason="judgments are one-sided; judge cases on both sides to pin the asymmetry",
        )

    r, lo, hi = fit_r(inputs)
    r_flat, _flo, _fhi = fit_r(flat_inputs)
    return AsymmetryFit(
        fitted=True,
        r=r,
        ci=(lo, hi),
        informative_count=len(inputs),
        block_count=block_count,
        allow_count=allow_count,
        provenance=split,
        r_flat=r_flat,
    )


# --------------------------------------------------------------------------- #
# Should-block resolution + per-slice error rates
# --------------------------------------------------------------------------- #


def should_block(case: BenchCase, r: float, judged: Optional[dict[str, str]] = None) -> bool:
    if judged:
        v = judged.get(case.id)
        if v in ("block", "never_event"):
            return True
        if v == "allow":
            return False
    return case.adversarial_score >= tau_of(r)


@dataclass(frozen=True)
class SliceStat:
    slice: str
    n: int
    false_allow: int
    false_block: int
    false_allow_rate: float
    false_block_rate: float


def slice_stats(
    cases: list[BenchCase], model: ModelId, r: float, judged: Optional[dict[str, str]] = None
) -> dict[str, SliceStat]:
    counts: dict[str, dict[str, int]] = {}
    order: list[str] = []
    for c in cases:
        if c.slice not in counts:
            counts[c.slice] = {"n": 0, "fa": 0, "fb": 0}
            order.append(c.slice)
        stat = counts[c.slice]
        stat["n"] += 1
        must_block = should_block(c, r, judged)
        blocked = c.model(model).decision == "block"
        if must_block and not blocked:
            stat["fa"] += 1
        if not must_block and blocked:
            stat["fb"] += 1
    out: dict[str, SliceStat] = {}
    for slice_id in order:
        stat = counts[slice_id]
        n = stat["n"]
        out[slice_id] = SliceStat(
            slice=slice_id,
            n=n,
            false_allow=stat["fa"],
            false_block=stat["fb"],
            false_allow_rate=(stat["fa"] / n) if n else 0.0,
            false_block_rate=(stat["fb"] / n) if n else 0.0,
        )
    return out


def mean_cost(cases: list[BenchCase], model: ModelId) -> float:
    if not cases:
        return 0.0
    return sum(c.model(model).cost_usd for c in cases) / len(cases)


def expected_loss(
    cases: list[BenchCase], model: ModelId, policy: Policy, judged: Optional[dict[str, str]] = None
) -> float:
    """Weighted asymmetric expected loss (docs/73 §3.2 stage 2)."""

    stats = slice_stats(cases, model, policy.r, judged)
    loss = 0.0
    for stat in stats.values():
        w = policy.slice_weights.get(stat.slice, 0.0)
        loss += w * (policy.r * stat.false_allow_rate + stat.false_block_rate)
    return loss + policy.cost_weight * mean_cost(cases, model)


# --------------------------------------------------------------------------- #
# Deterministic bootstrap (mulberry32 ported bit-for-bit from policyEngine.ts)
# --------------------------------------------------------------------------- #

_MASK32 = 0xFFFFFFFF


def mulberry32(seed: int):
    """Deterministic uint32 PRNG, a bit-for-bit port of the TS ``mulberry32``.

    All ops are carried as uint32 bit patterns; JS ``|0`` / ``>>> 0`` / int32
    ``Math.imul`` are all mod-2^32-invariant, so keeping the pattern in [0, 2^32)
    and masking each step reproduces the JS sequence exactly.
    """

    a = seed & _MASK32

    def _next() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & _MASK32
        t = ((a ^ (a >> 15)) * (1 | a)) & _MASK32
        inner = (((t ^ (t >> 7)) * (61 | t)) & _MASK32)
        t = ((t + inner) & _MASK32) ^ t
        return ((t ^ (t >> 14)) & _MASK32) / 4294967296

    return _next


def loss_difference_interval(
    cases: list[BenchCase],
    policy: Policy,
    judged: Optional[dict[str, str]] = None,
    draws: int = 400,
    z: float = 1.96,
) -> dict:
    """Bootstrap interval of L(candidate) - L(incumbent) over resampled cases.

    Seed parity with the TS engine: ``mulberry32(0x5eed ^ len(cases))``.
    Returns ``{"point": float, "interval": Interval}``.
    """

    point = expected_loss(cases, "candidate", policy, judged) - expected_loss(cases, "incumbent", policy, judged)
    if len(cases) < 2:
        return {"point": point, "interval": Interval(lower=point, upper=point)}
    rng = mulberry32(0x5EED ^ len(cases))
    diffs: list[float] = []
    n = len(cases)
    for _d in range(draws):
        sample = [cases[math.floor(rng() * n)] for _i in range(n)]
        diffs.append(
            expected_loss(sample, "candidate", policy, judged) - expected_loss(sample, "incumbent", policy, judged)
        )
    mean = sum(diffs) / len(diffs)
    variance = sum((x - mean) * (x - mean) for x in diffs) / max(1, len(diffs) - 1)
    sd = math.sqrt(variance)
    return {"point": point, "interval": Interval(lower=point - z * sd, upper=point + z * sd)}


def expected_loss_interval(
    cases: list[BenchCase],
    model: ModelId,
    policy: Policy,
    judged: Optional[dict[str, str]] = None,
    draws: int = 400,
    z: float = 1.96,
) -> dict:
    """Bootstrap interval of L(model) over resampled cases (docs/102 §4.7).

    Seed is distinct from :func:`loss_difference_interval` so the two intervals
    do not share a draw stream. Twin of ``expectedLossInterval`` in
    ``policyEngine.ts``.
    """

    point = expected_loss(cases, model, policy, judged)
    if len(cases) < 2:
        return {"point": point, "interval": Interval(lower=point, upper=point)}
    model_tag = 0xC0DE if model == "candidate" else 0xB0DE
    rng = mulberry32((0x5EED ^ len(cases) ^ model_tag) & _MASK32)
    losses: list[float] = []
    n = len(cases)
    for _d in range(draws):
        sample = [cases[math.floor(rng() * n)] for _i in range(n)]
        losses.append(expected_loss(sample, model, policy, judged))
    mean = sum(losses) / len(losses)
    variance = sum((x - mean) * (x - mean) for x in losses) / max(1, len(losses) - 1)
    sd = math.sqrt(variance)
    return {"point": point, "interval": Interval(lower=point - z * sd, upper=point + z * sd)}


# --------------------------------------------------------------------------- #
# Stage 1 — guarded constraints
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GuardedSliceResult:
    slice: str
    alpha: float
    min_evidence: int
    judged_n: int
    violations: int
    violation_upper: float
    state: SliceState
    unjudged_case_ids: list[str]
    # F-C4: ``None`` when the constraint cannot be cleared by judging the cases
    # this bench holds — a count of 0 there would read as "nothing left to do".
    cases_needed: Optional[int]
    unreachable: bool
    slice_n: int = 0
    cure: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "slice": self.slice,
            "alpha": self.alpha,
            "min_evidence": self.min_evidence,
            "judged_n": self.judged_n,
            "slice_n": self.slice_n,
            "violations": self.violations,
            "violation_upper": self.violation_upper,
            "state": self.state,
            "unjudged_case_ids": list(self.unjudged_case_ids),
            "cases_needed": self.cases_needed,
            "unreachable": self.unreachable,
            "cure": self.cure,
        }


def _plural(n: int, singular: str, plural: Optional[str] = None) -> str:
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


def _guarded_cure(
    *,
    state: SliceState,
    slice_id: str,
    alpha: float,
    violations: int,
    judged_n: int,
    slice_n: int,
    cases_needed: Optional[int],
    unreachable: bool,
) -> Optional[str]:
    """Plain-language next step for a guarded slice (docs/73 §4.2.4 refusals).

    Every non-passing state names cause AND cure. The unreachable case is the
    one F-C4 was about: judging everything left still cannot clear the ceiling,
    so the cure is *more cases*, not *more judging*.
    """

    ceiling = f"{alpha * 100:g}%"
    if state == "pass":
        return None
    if state == "block":
        return (
            f"{_plural(violations, 'judged case')} in the '{slice_id}' slice were auto-allowed by the candidate, "
            f"which is above your {ceiling} ceiling with confidence. More judging will not clear this: change the "
            f"candidate, or change the ceiling deliberately and re-approve the policy."
        )
    if unreachable:
        if slice_n == 0:
            return (
                f"This bench has no '{slice_id}' cases at all, so the {ceiling} ceiling cannot be tested. "
                f"Add '{slice_id}' cases to the bench (upload more traces covering this slice), then judge them."
            )
        return (
            f"Judging every remaining '{slice_id}' case cannot bring the violation ceiling under {ceiling}: "
            f"{_plural(violations, 'violation')} across {_plural(slice_n, 'case')} in this slice "
            f"({judged_n} judged so far). Add more '{slice_id}' cases to the bench — upload more traces "
            f"covering this slice — or reduce the violations before re-running the verdict."
        )
    needed = _plural(cases_needed or 0, "more case")
    if state == "refuse_insufficient":
        return f"Judge {needed} in the '{slice_id}' slice to reach the minimum evidence bar."
    return (
        f"The violation ceiling still straddles {ceiling}. "
        f"Judge {needed} in the '{slice_id}' slice to tighten it."
    )


def guarded_slice_result(
    cases: list[BenchCase],
    constraint: GuardedConstraint,
    candidate: ModelId,
    judged: dict[str, str],
    z: float = 1.96,
) -> GuardedSliceResult:
    slice_cases = [c for c in cases if c.slice == constraint.slice]
    # A violation is a judged never-event/block that the candidate auto-allowed.
    guarded_judged = [c for c in slice_cases if judged.get(c.id) in ("block", "never_event")]
    violations = sum(1 for c in guarded_judged if c.model(candidate).decision == "allow")
    judged_n = len(guarded_judged)
    violation_upper = wilson_bounds(violations, judged_n, z)[1]
    unjudged_case_ids = [c.id for c in slice_cases if c.id not in judged or judged.get(c.id) == "skip"]

    # How many judged guarded cases to project a pass (violations held fixed)?
    need_for_rate: float = math.inf
    n = max(judged_n, 1)
    while n <= len(slice_cases):
        if wilson_bounds(violations, n, z)[1] <= constraint.max_violation_rate:
            need_for_rate = n
            break
        n += 1
    unreachable = not math.isfinite(need_for_rate) or need_for_rate > len(slice_cases)
    if unreachable:
        # F-C4: no honest number exists here. Say so instead of clamping to 0.
        cases_needed: Optional[int] = None
    else:
        target = max(constraint.min_evidence, need_for_rate)
        cases_needed = max(0, int(target) - judged_n)

    if violations > 0 and wilson_bounds(violations, judged_n, z)[0] > constraint.max_violation_rate:
        state: SliceState = "block"
    elif judged_n < constraint.min_evidence:
        state = "refuse_insufficient"
    elif violation_upper > constraint.max_violation_rate:
        state = "refuse_straddle"
    else:
        state = "pass"
    return GuardedSliceResult(
        slice=constraint.slice,
        alpha=constraint.max_violation_rate,
        min_evidence=constraint.min_evidence,
        judged_n=judged_n,
        violations=violations,
        violation_upper=violation_upper,
        state=state,
        unjudged_case_ids=unjudged_case_ids,
        cases_needed=cases_needed,
        unreachable=unreachable,
        slice_n=len(slice_cases),
        cure=_guarded_cure(
            state=state,
            slice_id=constraint.slice,
            alpha=constraint.max_violation_rate,
            violations=violations,
            judged_n=judged_n,
            slice_n=len(slice_cases),
            cases_needed=cases_needed,
            unreachable=unreachable,
        ),
    )


# --------------------------------------------------------------------------- #
# Full two-stage verdict
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VerdictResult:
    state: VerdictState
    guarded: list[GuardedSliceResult]
    blocking_slice: Optional[GuardedSliceResult]
    loss_candidate: float
    loss_incumbent: float
    loss_difference: float
    loss_difference_interval: Interval
    candidate_preferred: bool
    cost_incumbent_usd: float
    cost_candidate_usd: float
    cost_delta_usd: float
    cost_cheaper_pct: float
    preference: Optional[Preference] = None
    reason: Optional[str] = None
    # docs/103 §8: v3 additions. Empty on v1/v2, and omitted from ``to_dict``
    # there, so the console bench and every v1/v2 snapshot stay byte-identical.
    actions: tuple["ActionResult", ...] = ()
    cures: tuple["Cure", ...] = ()
    regime: Optional["RegimeReadout"] = None

    def to_dict(self) -> dict:
        payload: dict = {
            "state": self.state,
            "guarded": [g.to_dict() for g in self.guarded],
            "blocking_slice": self.blocking_slice.to_dict() if self.blocking_slice else None,
            "loss_candidate": self.loss_candidate,
            "loss_incumbent": self.loss_incumbent,
            "loss_difference": self.loss_difference,
            "loss_difference_interval": {
                "lower": self.loss_difference_interval.lower,
                "upper": self.loss_difference_interval.upper,
            },
            "candidate_preferred": self.candidate_preferred,
            "cost_incumbent_usd": self.cost_incumbent_usd,
            "cost_candidate_usd": self.cost_candidate_usd,
            "cost_delta_usd": self.cost_delta_usd,
            "cost_cheaper_pct": self.cost_cheaper_pct,
            "preference": self.preference,
            "reason": self.reason,
        }
        if self.actions:
            payload["actions"] = [a.to_dict() for a in self.actions]
        if self.cures:
            payload["cures"] = [c.to_dict() for c in self.cures]
        if self.regime is not None:
            payload["regime"] = self.regime.to_dict()
        return payload


def compute_verdict(
    cases: list[BenchCase],
    policy: Policy,
    judged: dict[str, str],
    *,
    noise_ucb: Optional[float] = None,
    incumbent_present: bool = True,
) -> VerdictResult:
    """Two-stage verdict → pass | review | block | refuse (docs/73 §3, docs/102).

    ``cases`` are :class:`BenchCase` on v1/v2 and
    :class:`aver_policy.repeated.RepeatedCase` on v3 (docs/103 §2).
    """

    if policy.method_version == METHOD_VERSION_V3:
        from aver_policy.repeated import compute_verdict_v3  # local: avoids a cycle

        return compute_verdict_v3(cases, policy)  # type: ignore[arg-type]

    if policy.method_version == METHOD_VERSION_V2:
        return _compute_verdict_v2(
            cases, policy, judged, noise_ucb=noise_ucb, incumbent_present=incumbent_present
        )

    guarded = [guarded_slice_result(cases, g, "candidate", judged) for g in policy.guarded]
    blocking_slice = next((g for g in guarded if g.state == "block"), None)
    if blocking_slice is None:
        blocking_slice = next((g for g in guarded if g.state in ("refuse_insufficient", "refuse_straddle")), None)

    loss_candidate = expected_loss(cases, "candidate", policy, judged)
    loss_incumbent = expected_loss(cases, "incumbent", policy, judged)
    diff = loss_difference_interval(cases, policy, judged)
    point = diff["point"]
    interval = diff["interval"]
    candidate_preferred = interval.upper < 0  # whole interval clears zero

    cost_incumbent_usd = mean_cost(cases, "incumbent")
    cost_candidate_usd = mean_cost(cases, "candidate")
    cost_delta_usd = cost_candidate_usd - cost_incumbent_usd

    if any(g.state == "block" for g in guarded):
        state: VerdictState = "block"
    elif any(g.state in ("refuse_insufficient", "refuse_straddle") for g in guarded):
        state = "refuse"
    elif candidate_preferred:
        state = "pass"
    else:
        state = "review"

    return VerdictResult(
        state=state,
        guarded=guarded,
        blocking_slice=blocking_slice,
        loss_candidate=loss_candidate,
        loss_incumbent=loss_incumbent,
        loss_difference=point,
        loss_difference_interval=interval,
        candidate_preferred=candidate_preferred,
        cost_incumbent_usd=cost_incumbent_usd,
        cost_candidate_usd=cost_candidate_usd,
        cost_delta_usd=cost_delta_usd,
        cost_cheaper_pct=((cost_incumbent_usd - cost_candidate_usd) / cost_incumbent_usd * 100)
        if cost_incumbent_usd
        else 0.0,
    )


def _preference_from_interval(interval: Interval, *, incumbent_present: bool) -> Preference:
    if not incumbent_present:
        return "inconclusive"
    if interval.upper < 0:
        return "better"
    if interval.lower > 0:
        return "worse"
    return "inconclusive"


def _cost_ceiling_active(policy: Policy) -> bool:
    return policy.budget_cost_ceiling_usd > 0


def _has_required_absolute_clause(policy: Policy) -> bool:
    return bool(
        policy.guarded
        or _cost_ceiling_active(policy)
        or policy.variance_budget is not None
        or policy.absolute_loss_ceiling is not None
    )


def _candidate_cost_missing(cases: list[BenchCase]) -> bool:
    return any(not c.candidate.cost_present for c in cases)


def _compute_verdict_v2(
    cases: list[BenchCase],
    policy: Policy,
    judged: dict[str, str],
    *,
    noise_ucb: Optional[float],
    incumbent_present: bool,
) -> VerdictResult:
    """docs/102 §4 precedence. Stop at the first matching row."""

    z = z_from_confidence(policy.confidence)
    guarded = [guarded_slice_result(cases, g, "candidate", judged, z) for g in policy.guarded]
    blocking_slice = next((g for g in guarded if g.state == "block"), None)
    if blocking_slice is None:
        blocking_slice = next((g for g in guarded if g.state in ("refuse_insufficient", "refuse_straddle")), None)

    loss_candidate = expected_loss(cases, "candidate", policy, judged)
    loss_incumbent = expected_loss(cases, "incumbent", policy, judged)
    diff = loss_difference_interval(cases, policy, judged, z=z)
    point = diff["point"]
    interval = diff["interval"]
    preference = _preference_from_interval(interval, incumbent_present=incumbent_present and bool(cases))
    candidate_preferred = preference == "better"

    cost_incumbent_usd = mean_cost(cases, "incumbent")
    cost_candidate_usd = mean_cost(cases, "candidate")
    cost_delta_usd = cost_candidate_usd - cost_incumbent_usd

    state: Optional[VerdictState] = None
    reason: Optional[str] = None
    if policy.unsupported_clauses:
        state, reason = "refuse", "unsupported_clause"
    elif (_cost_ceiling_active(policy) and _candidate_cost_missing(cases)) or (
        policy.variance_budget is not None and noise_ucb is None
    ):
        state, reason = "refuse", "missing_measurement"
    elif any(g.state == "block" for g in guarded):
        state, reason = "block", None
    elif any(g.state in ("refuse_insufficient", "refuse_straddle") for g in guarded):
        state, reason = "refuse", None
    elif _cost_ceiling_active(policy) and cost_candidate_usd > policy.budget_cost_ceiling_usd:
        state, reason = "block", None
    elif policy.variance_budget is not None and noise_ucb is not None and noise_ucb > policy.variance_budget:
        state, reason = "block", None
    elif policy.absolute_loss_ceiling is not None:
        cand_interval = expected_loss_interval(cases, "candidate", policy, judged, z=z)["interval"]
        ceiling = policy.absolute_loss_ceiling
        if cand_interval.lower > ceiling:
            state, reason = "block", None
        elif cand_interval.upper > ceiling:
            state, reason = "refuse", None
    if state is None and policy.require_improvement and preference != "better":
        state, reason = ("block" if preference == "worse" else "refuse"), None
    if state is None and not _has_required_absolute_clause(policy):
        state, reason = "review", "no_absolute_clause"
    if state is None:
        state, reason = "pass", None

    return VerdictResult(
        state=state,
        guarded=guarded,
        blocking_slice=blocking_slice,
        loss_candidate=loss_candidate,
        loss_incumbent=loss_incumbent,
        loss_difference=point,
        loss_difference_interval=interval,
        candidate_preferred=candidate_preferred,
        cost_incumbent_usd=cost_incumbent_usd,
        cost_candidate_usd=cost_candidate_usd,
        cost_delta_usd=cost_delta_usd,
        cost_cheaper_pct=((cost_incumbent_usd - cost_candidate_usd) / cost_incumbent_usd * 100)
        if cost_incumbent_usd
        else 0.0,
        preference=preference,
        reason=reason,
    )


# --------------------------------------------------------------------------- #
# Sensitivity — verdict/preference as a function of r
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SensitivityPoint:
    r: float
    loss_candidate: float
    loss_incumbent: float
    candidate_preferred: bool


@dataclass(frozen=True)
class Sensitivity:
    flip_r: Optional[float]
    preferred_below_flip: bool
    curve: list[SensitivityPoint]
    ranges_over_ci: bool

    def to_dict(self) -> dict:
        return {
            "flip_r": self.flip_r,
            "preferred_below_flip": self.preferred_below_flip,
            "curve": [
                {
                    "r": p.r,
                    "loss_candidate": p.loss_candidate,
                    "loss_incumbent": p.loss_incumbent,
                    "candidate_preferred": p.candidate_preferred,
                }
                for p in self.curve
            ],
            "ranges_over_ci": self.ranges_over_ci,
        }


def sensitivity(
    cases: list[BenchCase],
    policy: Policy,
    judged: dict[str, str],
    r_min: float = 1,
    r_max: float = 20,
    ci: Optional[tuple[float, float]] = None,
) -> Sensitivity:
    """Preference of candidate-vs-incumbent as r sweeps a range (bisected flip)."""

    def prefer_at(r: float) -> bool:
        p = replace(policy, r=r)
        return expected_loss(cases, "candidate", p, judged) < expected_loss(cases, "incumbent", p, judged)

    curve: list[SensitivityPoint] = []
    steps = 40
    for i in range(steps + 1):
        r = r_min + ((r_max - r_min) * i) / steps
        p = replace(policy, r=r)
        curve.append(
            SensitivityPoint(
                r=r,
                loss_candidate=expected_loss(cases, "candidate", p, judged),
                loss_incumbent=expected_loss(cases, "incumbent", p, judged),
                candidate_preferred=prefer_at(r),
            )
        )
    flip_r: Optional[float] = None
    lo = r_min
    hi = r_max
    if prefer_at(lo) != prefer_at(hi):
        start_pref = prefer_at(lo)
        for _i in range(60):
            mid = (lo + hi) / 2
            if prefer_at(mid) == start_pref:
                lo = mid
            else:
                hi = mid
        flip_r = round(((lo + hi) / 2) * 100) / 100
    ranges_over_ci = bool(flip_r is not None and ci is not None and ci[0] <= flip_r <= ci[1])
    return Sensitivity(flip_r=flip_r, preferred_below_flip=prefer_at(r_min), curve=curve, ranges_over_ci=ranges_over_ci)


# --------------------------------------------------------------------------- #
# Changed cases under a parameter move
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ChangedCase:
    case_id: str
    slice: str
    from_state: str
    to_state: str


def changed_cases(
    cases: list[BenchCase], r_before: float, r_after: float, judged: Optional[dict[str, str]] = None
) -> list[ChangedCase]:
    out: list[ChangedCase] = []
    for c in cases:
        before = should_block(c, r_before, judged)
        after = should_block(c, r_after, judged)
        if before != after:
            out.append(
                ChangedCase(
                    case_id=c.id,
                    slice=c.slice,
                    from_state="should_block" if before else "should_allow",
                    to_state="should_block" if after else "should_allow",
                )
            )
    return out


# --------------------------------------------------------------------------- #
# Receipts — divergent cases ranked by policy-weighted severity
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Receipt:
    case_id: str
    slice: str
    summary: str
    category: str
    incumbent: Decision
    candidate: Decision
    should_block: bool
    candidate_false_allow: bool
    candidate_false_block: bool
    severity: float
    cost_delta_usd: float

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "slice": self.slice,
            "summary": self.summary,
            "category": self.category,
            "incumbent": self.incumbent,
            "candidate": self.candidate,
            "should_block": self.should_block,
            "candidate_false_allow": self.candidate_false_allow,
            "candidate_false_block": self.candidate_false_block,
            "severity": self.severity,
            "cost_delta_usd": self.cost_delta_usd,
        }


def receipts(
    cases: list[BenchCase], policy: Policy, judged: dict[str, str], only_differences: bool = True
) -> list[Receipt]:
    """Divergent cases ordered by policy severity (docs/73 §4.3.3)."""

    rows: list[Receipt] = []
    for c in cases:
        differ = c.incumbent.decision != c.candidate.decision
        if only_differences and not differ:
            continue
        must_block = should_block(c, policy.r, judged)
        candidate_false_allow = must_block and c.candidate.decision == "allow"
        candidate_false_block = (not must_block) and c.candidate.decision == "block"
        w = policy.slice_weights.get(c.slice, 0.0)
        severity = w * (policy.r if candidate_false_allow else (1 if candidate_false_block else 0.25))
        rows.append(
            Receipt(
                case_id=c.id,
                slice=c.slice,
                summary=c.summary,
                category=c.category,
                incumbent=c.incumbent.decision,
                candidate=c.candidate.decision,
                should_block=must_block,
                candidate_false_allow=candidate_false_allow,
                candidate_false_block=candidate_false_block,
                severity=severity,
                cost_delta_usd=c.candidate.cost_usd - c.incumbent.cost_usd,
            )
        )
    rows.sort(key=lambda r: (-r.severity, r.case_id))
    return rows


# --------------------------------------------------------------------------- #
# Projections
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CostProjection:
    monthly_volume: int
    incumbent_low: float
    incumbent_high: float
    candidate_low: float
    candidate_high: float
    saving_low: float
    saving_high: float

    def to_dict(self) -> dict:
        return {
            "monthly_volume": self.monthly_volume,
            "incumbent_low": self.incumbent_low,
            "incumbent_high": self.incumbent_high,
            "candidate_low": self.candidate_low,
            "candidate_high": self.candidate_high,
            "saving_low": self.saving_low,
            "saving_high": self.saving_high,
        }


def cost_projection(cases: list[BenchCase], monthly_volume: int, band_fraction: float = 0.15) -> CostProjection:
    """Projected monthly cost as a RANGE at a stated volume (never a realised saving)."""

    inc = mean_cost(cases, "incumbent") * monthly_volume
    cand = mean_cost(cases, "candidate") * monthly_volume
    return CostProjection(
        monthly_volume=monthly_volume,
        incumbent_low=inc * (1 - band_fraction),
        incumbent_high=inc * (1 + band_fraction),
        candidate_low=cand * (1 - band_fraction),
        candidate_high=cand * (1 + band_fraction),
        saving_low=(inc - cand) * (1 - band_fraction),
        saving_high=(inc - cand) * (1 + band_fraction),
    )


def judgment_map(judgments: list[Judgment]) -> dict[str, str]:
    return {j.case_id: j.verdict for j in judgments}


# --------------------------------------------------------------------------- #
# Dict adapters for scripts/build_console_bench.py (byte-identical reference)
# --------------------------------------------------------------------------- #


def slice_loss(
    cases: list[dict],
    model: str,
    r: float,
    weights: dict[str, float],
    lam: float,
    judged: Optional[dict[str, str]] = None,
) -> float:
    """Dict-shaped :func:`expected_loss` for the bench generator (mirror of the
    former ``build_console_bench._loss``). ``cases`` are the raw bundle dicts."""

    policy = Policy(r=r, slice_weights=weights, cost_weight=lam)
    typed = [BenchCase.from_dict(c) for c in cases]
    return expected_loss(typed, model, policy, judged)  # type: ignore[arg-type]
