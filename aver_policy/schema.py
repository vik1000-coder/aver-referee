"""aver_policy.schema — the versioned ``.policy`` object (docs/73 §7).

The decision-policy DSL: asymmetry, guarded slices, slice weights/tolerances,
budget, variance budget, confidence. This is the CV8 OSS-standard seed — a
plain, JSON-schema'd dataclass with no service dependency. A ratified
``decision_policy`` version stores exactly ``DecisionPolicySpec.to_definition()``
inside its immutable ``definition`` (docs/74 CV1 T1.4), and the verdict path
rebuilds an :class:`aver_policy.engine.Policy` from it via :meth:`to_policy`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from aver_policy.engine import (
    DEFAULT_DECIDABILITY_THRESHOLD,
    DEFAULT_MAX_UNDECIDED_FRACTION,
    FIT_SLOPE_K,
    GUARDED_ACTION_DEFAULT_MIN_EVIDENCE,
    GUARDED_ACTION_DEFAULT_TOLERANCE,
    GuardedAction,
    GuardedConstraint,
    Policy,
)

# The kind discriminator this spec lands under in ``risk_policies.kind`` and in
# ``definition->>'kind'`` (docs/74 CV1 migration).
POLICY_KIND = "decision_policy"

# Asymmetry provenance. ``fitted`` = r comes from the judgment MLE (docs/73
# §3.4). ``manual`` = the owner set a deliberate r by hand; the fitted r and its
# interval are still carried for disclosure ("set by hand, fitted 4.2×
# (3.1–5.8×)"). An override never erases the fit — it sits beside it.
ASYMMETRY_MODE_FITTED = "fitted"
ASYMMETRY_MODE_MANUAL = "manual"
ASYMMETRY_MODES: tuple[str, ...] = (ASYMMETRY_MODE_FITTED, ASYMMETRY_MODE_MANUAL)

# Valid range for a deliberate manual r — the same range the fit grid searches
# (``aver_policy.engine._FIT_GRID``: 0.5 … 30.0), so a hand-set value can never
# leave the territory the sensitivity curve and the fit CI can describe.
MANUAL_R_MIN = 0.5
MANUAL_R_MAX = 30.0

# Guarded-slice provenance: ``default`` = derived from the bench (the fitted
# default), ``custom`` = the owner deliberately set the ceiling / evidence bar.
GUARDED_SOURCE_DEFAULT = "default"
GUARDED_SOURCE_CUSTOM = "custom"

# The never-event ceiling is a violation RATE and must stay below 1.0: a ceiling
# of 1.0 accepts every violation, so the Wilson upper bound (which is exactly
# 1.0 on zero evidence) could "clear" it with nothing judged — the one loosening
# that would let an override manufacture a pass out of no evidence.
CEILING_MIN = 0.0
CEILING_MAX_EXCLUSIVE = 1.0

# The variance budget is a flip-rate ceiling (share of identical-input repeats
# allowed to flip), so it lives in [0, 1].
VARIANCE_BUDGET_MIN = 0.0
VARIANCE_BUDGET_MAX = 1.0

# docs/102 method versions. ``from_parameters`` / ``Policy.from_dict`` default
# to v1 when the key is absent so already-stored payloads keep today's rules.
METHOD_VERSION_V1 = "decision_policy_v1"
METHOD_VERSION_V2 = "decision_policy_v2"
# docs/103: v3 applies only to evidence sets that carry repeated runs. A
# single-run evidence set still ratifies v2.
METHOD_VERSION_V3 = "decision_policy_v3"
METHOD_VERSIONS: tuple[str, ...] = (METHOD_VERSION_V1, METHOD_VERSION_V2, METHOD_VERSION_V3)

# docs/103 §3 bounds. theta below 0.5 is not a majority at all; 1.0 means
# unanimity, which is a legitimate (strict) choice.
DECIDABILITY_THRESHOLD_MIN = 0.5
DECIDABILITY_THRESHOLD_MAX = 1.0
MAX_UNDECIDED_FRACTION_MIN = 0.0
MAX_UNDECIDED_FRACTION_MAX = 1.0

# Future fields that must fail closed on v2 if they appear without an
# implementation (docs/102 §5). A latency ceiling is the named example.
UNSUPPORTED_CLAUSE_KEYS: tuple[str, ...] = ("budget_latency_p95_ms",)

# docs/104 §3: which side of a guarded action is the harmful one. The evidence
# importer maps ``fired`` to the engine's 1 = safe side with it, so a ratified
# action says which mapping its acceptance rules were written for.
GUARDED_ACTION_DIRECTION_FIRE_IS_HARMFUL = "fire_is_harmful"
GUARDED_ACTION_DIRECTION_WITHHOLD_IS_HARMFUL = "withhold_is_harmful"
GUARDED_ACTION_DIRECTIONS: tuple[str, ...] = (
    GUARDED_ACTION_DIRECTION_FIRE_IS_HARMFUL,
    GUARDED_ACTION_DIRECTION_WITHHOLD_IS_HARMFUL,
)

# JSON-schema for the ``definition.parameters`` sub-object a decision_policy
# version carries. Advisory (the DB enforces shape via CHECK constraints); kept
# here as the open-standard contract CV8 publishes.
PARAMETERS_JSON_SCHEMA: dict = {
    "type": "object",
    "required": ["asymmetry", "guarded_slices", "slice_weights", "confidence"],
    "properties": {
        "asymmetry": {
            "type": "object",
            "required": ["mode", "fit_slope_k"],
            "properties": {
                "mode": {"enum": list(ASYMMETRY_MODES)},
                "fit_slope_k": {"type": "number"},
                "r": {"type": ["number", "null"]},
                "ci": {
                    "type": ["array", "null"],
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "manual_r": {
                    "type": ["number", "null"],
                    "minimum": MANUAL_R_MIN,
                    "maximum": MANUAL_R_MAX,
                },
            },
        },
        "guarded_slices": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["slice", "max_violation_rate", "min_evidence"],
                "properties": {
                    "slice": {"type": "string"},
                    "max_violation_rate": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "min_evidence": {"type": "integer", "minimum": 0},
                    "source": {"enum": [GUARDED_SOURCE_DEFAULT, GUARDED_SOURCE_CUSTOM]},
                },
            },
        },
        "slice_weights": {"type": "object", "additionalProperties": {"type": "number"}},
        "slice_tolerances": {"type": "object", "additionalProperties": {"type": "number"}},
        "confidence": {"type": "number", "minimum": 0.8, "exclusiveMaximum": 1.0},
        "cost_weight": {"type": "number", "minimum": 0.0},
        "budget_cost_ceiling_usd": {"type": "number", "minimum": 0.0},
        "variance_budget": {
            "type": ["number", "null"],
            "minimum": VARIANCE_BUDGET_MIN,
            "maximum": VARIANCE_BUDGET_MAX,
        },
        "projection_monthly_volume": {"type": "integer", "minimum": 0},
        "method_version": {
            "enum": ["decision_policy_v1", "decision_policy_v2", "decision_policy_v3"]
        },
        "absolute_loss_ceiling": {"type": ["number", "null"]},
        "require_improvement": {"type": "boolean"},
        "decidability_threshold": {
            "type": "number",
            "minimum": DECIDABILITY_THRESHOLD_MIN,
            "maximum": DECIDABILITY_THRESHOLD_MAX,
        },
        "max_undecided_fraction": {
            "type": "number",
            "minimum": MAX_UNDECIDED_FRACTION_MIN,
            "maximum": MAX_UNDECIDED_FRACTION_MAX,
        },
        "guarded_actions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["action"],
                "properties": {
                    "action": {"type": "string"},
                    "tolerance": {"type": "number", "minimum": 0.0, "exclusiveMaximum": 1.0},
                    "harm_tolerance": {
                        "type": ["number", "null"],
                        "minimum": 0.0,
                        "exclusiveMaximum": 1.0,
                    },
                    "min_evidence": {"type": "integer", "minimum": 0},
                    "direction": {"enum": list(GUARDED_ACTION_DIRECTIONS)},
                },
            },
        },
    },
}


@dataclass
class AsymmetrySpec:
    r: Optional[float] = None
    ci: Optional[tuple[float, float]] = None
    mode: str = ASYMMETRY_MODE_FITTED
    fit_slope_k: int = FIT_SLOPE_K
    # A deliberate hand-set r (mode == "manual"). The fitted ``r``/``ci`` stay
    # populated beside it so every surface can disclose "set by hand" AND still
    # show the fitted interval the owner overrode.
    manual_r: Optional[float] = None

    @property
    def is_manual(self) -> bool:
        return self.mode == ASYMMETRY_MODE_MANUAL and self.manual_r is not None

    @property
    def effective_r(self) -> Optional[float]:
        """The r the verdict scores with: the manual override when set, else the fit."""

        if self.is_manual:
            return self.manual_r
        return self.r

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "fit_slope_k": self.fit_slope_k,
            "r": self.r,
            "ci": list(self.ci) if self.ci is not None else None,
            "manual_r": self.manual_r,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "AsymmetrySpec":
        ci = raw.get("ci")
        manual_r = raw.get("manual_r")
        # docs/102 §6: a new v2 policy that already has a hand-set r defaults
        # to manual (the fit still rides beside it). An explicit mode wins.
        if "mode" in raw:
            mode = str(raw["mode"])
        elif manual_r is not None:
            mode = ASYMMETRY_MODE_MANUAL
        else:
            mode = ASYMMETRY_MODE_FITTED
        return cls(
            r=raw.get("r"),
            ci=(float(ci[0]), float(ci[1])) if ci else None,
            mode=mode,
            fit_slope_k=int(raw.get("fit_slope_k", FIT_SLOPE_K)),
            manual_r=float(manual_r) if manual_r is not None else None,
        )


@dataclass
class GuardedSliceSpec:
    slice: str
    max_violation_rate: float
    min_evidence: int
    # Provenance disclosure: ``default`` = the bench-derived value; ``custom`` =
    # the owner deliberately set this ceiling / evidence bar. Legacy rows that
    # never carried the key read back as ``default``.
    source: str = GUARDED_SOURCE_DEFAULT

    @property
    def is_custom(self) -> bool:
        return self.source == GUARDED_SOURCE_CUSTOM

    def to_dict(self) -> dict:
        return {
            "slice": self.slice,
            "max_violation_rate": self.max_violation_rate,
            "min_evidence": self.min_evidence,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "GuardedSliceSpec":
        return cls(
            slice=str(raw["slice"]),
            max_violation_rate=float(raw["max_violation_rate"]),
            min_evidence=int(raw["min_evidence"]),
            source=str(raw.get("source") or GUARDED_SOURCE_DEFAULT),
        )

    def to_constraint(self) -> GuardedConstraint:
        return GuardedConstraint(
            slice=self.slice, max_violation_rate=self.max_violation_rate, min_evidence=self.min_evidence
        )


@dataclass
class GuardedActionSpec:
    """One ratified guarded action for decision-policy v3 (docs/103 §§4-5).

    ``action`` is the tool call whose firing is the consequential event.
    ``tolerance`` is Clause A's ceiling on the modal switch rate; ``harm_tolerance``
    is Clause B's. An explicit ``None`` for ``harm_tolerance`` switches Clause B
    OFF for this action (docs/103 §5 opt-out) and must be stated deliberately —
    ``harm_tolerance_set`` records whether the ratified document said anything at
    all, so an omission inherits ``tolerance`` instead of silencing the clause.
    """

    action: str
    tolerance: float = GUARDED_ACTION_DEFAULT_TOLERANCE
    harm_tolerance: Optional[float] = None
    harm_tolerance_set: bool = False
    min_evidence: int = GUARDED_ACTION_DEFAULT_MIN_EVIDENCE
    # docs/104 §3. The engine never reads it (evidence arrives already mapped to
    # 1 = safe side); the service uses it to map imported runs to the ratified
    # reading of the action.
    direction: str = GUARDED_ACTION_DIRECTION_FIRE_IS_HARMFUL

    @property
    def harm_clause_active(self) -> bool:
        return not (self.harm_tolerance_set and self.harm_tolerance is None)

    @property
    def effective_harm_tolerance(self) -> Optional[float]:
        if not self.harm_clause_active:
            return None
        return self.tolerance if self.harm_tolerance is None else self.harm_tolerance

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "tolerance": self.tolerance,
            "harm_tolerance": self.effective_harm_tolerance,
            "min_evidence": self.min_evidence,
            "direction": self.direction,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "GuardedActionSpec":
        tolerance = raw.get("tolerance")
        min_evidence = raw.get("min_evidence")
        harm_set = "harm_tolerance" in raw
        harm = raw.get("harm_tolerance")
        return cls(
            action=str(raw["action"]),
            tolerance=GUARDED_ACTION_DEFAULT_TOLERANCE if tolerance is None else float(tolerance),
            harm_tolerance=None if harm is None else float(harm),
            harm_tolerance_set=harm_set,
            min_evidence=GUARDED_ACTION_DEFAULT_MIN_EVIDENCE
            if min_evidence is None
            else int(min_evidence),
            direction=str(raw.get("direction") or GUARDED_ACTION_DIRECTION_FIRE_IS_HARMFUL),
        )

    def to_action(self) -> GuardedAction:
        return GuardedAction(
            action=self.action,
            tolerance=self.tolerance,
            harm_tolerance=self.harm_tolerance,
            harm_clause_active=self.harm_clause_active,
            min_evidence=self.min_evidence,
        )


@dataclass
class DecisionPolicySpec:
    """The elicited decision policy for one role (docs/73 §3)."""

    asymmetry: AsymmetrySpec = field(default_factory=AsymmetrySpec)
    guarded_slices: list[GuardedSliceSpec] = field(default_factory=list)
    slice_weights: dict[str, float] = field(default_factory=dict)
    slice_tolerances: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.95
    cost_weight: float = 0.0
    budget_cost_ceiling_usd: float = 0.0
    # McNemar-style flip-noise ceiling; CV5 populates from measured variance.
    variance_budget: Optional[float] = None
    projection_monthly_volume: int = 0
    method_version: str = METHOD_VERSION_V1
    # UCB ceiling on L(candidate); ``None`` is inactive (docs/102 §5).
    absolute_loss_ceiling: Optional[float] = None
    # When true, preference must be ``better`` (docs/102 §3). Default inactive.
    require_improvement: bool = False
    # Fail-closed channel for a stored clause the engine does not implement.
    unsupported_clauses: list[str] = field(default_factory=list)
    # docs/103 §§3-5 (v3 only; inert on v1/v2).
    decidability_threshold: float = DEFAULT_DECIDABILITY_THRESHOLD
    max_undecided_fraction: float = DEFAULT_MAX_UNDECIDED_FRACTION
    guarded_actions: list[GuardedActionSpec] = field(default_factory=list)

    def to_parameters(self) -> dict:
        """The ``definition.parameters`` sub-object stored on the version."""

        return {
            "asymmetry": self.asymmetry.to_dict(),
            "guarded_slices": [g.to_dict() for g in self.guarded_slices],
            "slice_weights": dict(self.slice_weights),
            "slice_tolerances": dict(self.slice_tolerances),
            "confidence": self.confidence,
            "cost_weight": self.cost_weight,
            "budget_cost_ceiling_usd": self.budget_cost_ceiling_usd,
            "variance_budget": self.variance_budget,
            "projection_monthly_volume": self.projection_monthly_volume,
            "method_version": self.method_version,
            "absolute_loss_ceiling": self.absolute_loss_ceiling,
            "require_improvement": self.require_improvement,
            "decidability_threshold": self.decidability_threshold,
            "max_undecided_fraction": self.max_undecided_fraction,
            "guarded_actions": [a.to_dict() for a in self.guarded_actions],
        }

    @classmethod
    def from_parameters(cls, raw: dict) -> "DecisionPolicySpec":
        method_raw = raw.get("method_version")
        # docs/103 §7: no field means v1, exactly as before.
        if method_raw == METHOD_VERSION_V3:
            method_version = METHOD_VERSION_V3
        elif method_raw == METHOD_VERSION_V2:
            method_version = METHOD_VERSION_V2
        else:
            method_version = METHOD_VERSION_V1
        abs_raw = raw.get("absolute_loss_ceiling")
        unsupported = [str(x) for x in (raw.get("unsupported_clauses") or [])]
        for key in UNSUPPORTED_CLAUSE_KEYS:
            if raw.get(key) is not None and key not in unsupported:
                unsupported.append(key)
        return cls(
            asymmetry=AsymmetrySpec.from_dict(raw.get("asymmetry", {})),
            guarded_slices=[GuardedSliceSpec.from_dict(g) for g in raw.get("guarded_slices", [])],
            slice_weights={str(k): float(v) for k, v in (raw.get("slice_weights") or {}).items()},
            slice_tolerances={str(k): float(v) for k, v in (raw.get("slice_tolerances") or {}).items()},
            confidence=float(raw.get("confidence", 0.95)),
            cost_weight=float(raw.get("cost_weight", 0.0) or 0.0),
            budget_cost_ceiling_usd=float(raw.get("budget_cost_ceiling_usd", 0.0) or 0.0),
            variance_budget=raw.get("variance_budget"),
            projection_monthly_volume=int(raw.get("projection_monthly_volume", 0) or 0),
            method_version=method_version,
            absolute_loss_ceiling=None if abs_raw is None else float(abs_raw),
            require_improvement=bool(raw.get("require_improvement", False)),
            unsupported_clauses=unsupported,
            decidability_threshold=float(
                DEFAULT_DECIDABILITY_THRESHOLD
                if raw.get("decidability_threshold") is None
                else raw["decidability_threshold"]
            ),
            max_undecided_fraction=float(
                DEFAULT_MAX_UNDECIDED_FRACTION
                if raw.get("max_undecided_fraction") is None
                else raw["max_undecided_fraction"]
            ),
            guarded_actions=[
                GuardedActionSpec.from_dict(a) for a in (raw.get("guarded_actions") or [])
            ],
        )

    def to_policy(self, *, r: Optional[float] = None) -> Policy:
        """Rebuild the engine :class:`Policy` for verdict scoring.

        ``r`` overrides the stored asymmetry (e.g. a freshly fitted r being
        previewed before it lands in the next proposed version). Otherwise the
        manual override wins when one is set, else the fitted r.
        """

        effective_r = r if r is not None else self.asymmetry.effective_r
        if effective_r is None:
            raise ValueError(
                "policy has no asymmetry r; fit judgments first, or set a deliberate manual r"
            )
        return Policy(
            r=float(effective_r),
            guarded=[g.to_constraint() for g in self.guarded_slices],
            slice_weights=dict(self.slice_weights),
            cost_weight=self.cost_weight,
            confidence=self.confidence,
            budget_cost_ceiling_usd=self.budget_cost_ceiling_usd,
            method_version=self.method_version,  # type: ignore[arg-type]
            absolute_loss_ceiling=self.absolute_loss_ceiling,
            require_improvement=self.require_improvement,
            variance_budget=None if self.variance_budget is None else float(self.variance_budget),
            unsupported_clauses=list(self.unsupported_clauses),
            decidability_threshold=self.decidability_threshold,
            max_undecided_fraction=self.max_undecided_fraction,
            guarded_actions=[a.to_action() for a in self.guarded_actions],
        )

    def validate(self) -> None:
        """Cheap structural validation (the DB CHECK constraints are the gate).

        Every rejection names the valid range — a caller's 422 repeats it
        verbatim, and nothing is ever silently clamped.
        """

        if self.method_version not in METHOD_VERSIONS:
            raise ValueError(f"method_version must be one of {list(METHOD_VERSIONS)}")
        if not 0.8 <= self.confidence < 1.0:
            raise ValueError("confidence must be in [0.8, 1.0)")
        if self.asymmetry.mode not in ASYMMETRY_MODES:
            raise ValueError(f"asymmetry mode must be one of {list(ASYMMETRY_MODES)}")
        if self.asymmetry.mode == ASYMMETRY_MODE_MANUAL:
            if self.asymmetry.manual_r is None:
                raise ValueError("manual asymmetry mode requires manual_r")
            if not MANUAL_R_MIN <= self.asymmetry.manual_r <= MANUAL_R_MAX:
                raise ValueError(
                    f"manual asymmetry r must be in [{MANUAL_R_MIN}, {MANUAL_R_MAX}]"
                )
        elif self.asymmetry.manual_r is not None:
            raise ValueError("manual_r requires asymmetry mode 'manual'")
        for g in self.guarded_slices:
            if not 0.0 <= g.max_violation_rate <= 1.0:
                raise ValueError("guarded max_violation_rate must be in [0, 1]")
            if g.is_custom and not CEILING_MIN <= g.max_violation_rate < CEILING_MAX_EXCLUSIVE:
                # A hand-set ceiling of exactly 1.0 would be clearable on zero
                # evidence (the Wilson upper on 0/0 is 1.0) — the one loosening
                # that could bypass the evidence gate. Refuse it by name.
                raise ValueError(
                    f"a custom never-event ceiling must be in [{CEILING_MIN}, {CEILING_MAX_EXCLUSIVE}) — "
                    "1.0 would accept every violation and can never refuse"
                )
            if g.min_evidence < 0:
                raise ValueError("guarded min_evidence must be non-negative")
            if g.source not in (GUARDED_SOURCE_DEFAULT, GUARDED_SOURCE_CUSTOM):
                raise ValueError(
                    f"guarded slice source must be '{GUARDED_SOURCE_DEFAULT}' or '{GUARDED_SOURCE_CUSTOM}'"
                )
        if self.variance_budget is not None and not (
            VARIANCE_BUDGET_MIN <= float(self.variance_budget) <= VARIANCE_BUDGET_MAX
        ):
            raise ValueError(
                f"variance_budget must be in [{VARIANCE_BUDGET_MIN}, {VARIANCE_BUDGET_MAX}] (a flip-rate ceiling)"
            )
        # docs/103 §3 / §5.
        if not DECIDABILITY_THRESHOLD_MIN <= self.decidability_threshold <= DECIDABILITY_THRESHOLD_MAX:
            raise ValueError(
                f"decidability_threshold must be in "
                f"[{DECIDABILITY_THRESHOLD_MIN}, {DECIDABILITY_THRESHOLD_MAX}]"
            )
        if not MAX_UNDECIDED_FRACTION_MIN <= self.max_undecided_fraction <= MAX_UNDECIDED_FRACTION_MAX:
            raise ValueError(
                f"max_undecided_fraction must be in "
                f"[{MAX_UNDECIDED_FRACTION_MIN}, {MAX_UNDECIDED_FRACTION_MAX}]"
            )
        seen_actions: set[str] = set()
        for a in self.guarded_actions:
            if not a.action:
                raise ValueError("a guarded action must name the action it guards")
            if a.action in seen_actions:
                raise ValueError(f"guarded action '{a.action}' is declared twice")
            seen_actions.add(a.action)
            if not 0.0 <= a.tolerance < 1.0:
                raise ValueError("guarded action tolerance must be in [0, 1)")
            if a.harm_tolerance is not None and not 0.0 <= a.harm_tolerance < 1.0:
                raise ValueError("guarded action harm_tolerance must be in [0, 1) or null")
            if a.min_evidence < 0:
                raise ValueError("guarded action min_evidence must be non-negative")
            if a.direction not in GUARDED_ACTION_DIRECTIONS:
                raise ValueError(
                    f"guarded action direction must be one of {list(GUARDED_ACTION_DIRECTIONS)}"
                )
        if self.method_version != METHOD_VERSION_V3 and self.guarded_actions:
            raise ValueError(
                "guarded_actions are a decision_policy_v3 clause; set method_version to "
                f"'{METHOD_VERSION_V3}' or remove them"
            )
