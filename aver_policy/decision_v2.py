"""Evidence-priced acceptance: staged looks, clause-aware allocation, planning in the user's units.

Method aver-decision-binary-v2.1 (docs/127, revised by docs/130 §8 — see docs/131). Extends aver-decision-binary-v1 (docs/126):
  * K planned looks; the 5% error budget is split across candidates x populations, then across looks
    (per-cell level alpha_U * s_j at look j, sum_j s_j = 1), so the chance of ever supporting a candidate
    that violates the bar is <= 5% over the whole procedure (union bound over looks, then the v1
    intersection-union argument).
  * Allocation of confirmation cases by population: every population but the heaviest gets at least half an
    equal share of the total; the heaviest carries the average and takes the remainder.
  * Design rates from the pilot (optimistic / central / pessimistic), operating characteristics by
    Monte Carlo with binomial increments per look, net value of checking and the best cap.
  * look(): evaluate cumulative evidence at a planned look; PASS / BLOCK stop, CONTINUE goes on,
    EXHAUSTED is v1's refuse at the last look. A stopped plan accepts no further looks.
Local assessments only; nothing here issues a record.

Revision 2026-10-01 (docs/130 §8, docs/131): the per-cell level is no longer split across populations (alpha_U =
0.05/q * s_j); weighted clauses (the average and each tolerance) use the Fay-Feuer gamma bound (`weighted_upper`,
exact Clopper-Pearson with one population); an audit of AI labels is two-phase (AI-"fine" rows and AI-flagged rows
counted separately) and replaces the limit by the audited limit, spending 0.01 of the 0.05 on the audit. A v3 spec
decided this way reports method aver-decision-binary-v3.1. The gamma functions use only +, -, *, / and comparisons
(exp and log are computed here, not by the platform), so the TypeScript twin reproduces them bit for bit.
"""
from __future__ import annotations

import math
from functools import lru_cache

from aver_policy.decision import (
    ALPHA, DecisionError, digest, explore, fields, label, need, number, sequential_sum, upper,
    validate_rows, validate_spec, zero_floor,
)

METHOD = 'aver-decision-binary-v2.1'   # docs/131: v2 specs are decided by the revised rule (no split, gamma bound)
METHOD_V3 = 'aver-decision-binary-v3.1'
SPEC_SCHEMA = 'aver-decision-spec/v2'
SPEC_SCHEMA_V3 = 'aver-decision-spec/v3'
GRID = list(range(600, 6001, 300))
ORIGINS_V2 = ('synthetic', 'external', 'customer', 'ai-labeled')
STANDARD_KINDS = ('wrong_output', 'incomplete', 'needless_escalation', 'unwanted_action', 'harmful_output', 'runaway_cost')


def is_v3(spec):
    return isinstance(spec, dict) and 'taxonomy' in spec


def method_of(spec):
    return METHOD_V3 if is_v3(spec) else METHOD


def schema_suffix(spec):
    return 'v3' if is_v3(spec) else 'v2'


def grader_error_of(spec):
    """The declared grader error. An audit no longer turns into point estimates of it (docs/130 §3.3); it enters the
    decision through the audited limit instead (`audited_limit`)."""
    return spec['grader_error']


# ----------------------------------------------------------------------------- portable exp, log, log-gamma
# Pure IEEE arithmetic (+, -, *, /, comparisons, floor), written identically in decisionWorkflowV2.ts, so the gamma bound
# is the same double in Python and in the browser. Accurate to about one unit in the last place.

_LN2_HI = 6.93147180369123816490e-01
_LN2_LO = 1.90821492927058770002e-10
_INV_LN2 = 1.44269504088896338700e+00
_SQRT2 = 1.4142135623730951
_SQRT_HALF = 0.7071067811865476
_TWO64 = 18446744073709551616.0
_TWOM64 = 1.0 / _TWO64


def _exp_coefficients():
    out = [1.0]; c = 1.0
    for n in range(1, 16):
        c = c / n; out.append(c)
    return out


_EXP_COEF = _exp_coefficients()
_LOG_COEF = [1.0 / d for d in range(1, 24, 2)]
_LANCZOS = [676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
GAMMA_ITERATIONS = 10000
QUANTILE_STEPS = 64


def _pow2(k):
    r = 1.0; b = 2.0 if k > 0 else 0.5; n = k if k > 0 else -k
    while n:
        if n & 1: r *= b
        b *= b; n >>= 1
    return r


def portable_exp(x):
    if x < -700: return 0.0
    if x > 700: return math.inf
    k = math.floor(x * _INV_LN2 + 0.5)
    r = (x - k * _LN2_HI) - k * _LN2_LO
    p = _EXP_COEF[15]
    for i in range(14, -1, -1): p = p * r + _EXP_COEF[i]
    return p * _pow2(k)


def portable_log(x):
    """Natural log of a positive finite x."""
    e = 0; m = x
    while m >= _TWO64: m = m * _TWOM64; e += 64
    while m < _TWOM64: m = m * _TWO64; e -= 64
    while m >= _SQRT2: m = m * 0.5; e += 1
    while m < _SQRT_HALF: m = m * 2.0; e -= 1
    f = m - 1.0; s = f / (2.0 + f); s2 = s * s
    q = _LOG_COEF[11]
    for i in range(10, -1, -1): q = q * s2 + _LOG_COEF[i]
    return e * _LN2_HI + (e * _LN2_LO + 2.0 * s * q)


def log_gamma(z):
    """Lanczos (g = 7), the same coefficients as logGamma in decisionWorkflow.ts; z > 0."""
    z -= 1; x = .99999999999980993
    for i, v in enumerate(_LANCZOS): x += v / (z + i + 1)
    t = z + 7.5
    return .5 * portable_log(2 * math.pi) + (z + .5) * portable_log(t) - t + portable_log(x)


def gamma_cdf(a, x, lg=None):
    """Normalised lower incomplete gamma P(a, x), a > 0: series below a + 1, continued fraction (modified Lentz) above."""
    if x <= 0: return 0.0
    if lg is None: lg = log_gamma(a)
    if x < a + 1:
        ap = a; term = 1.0 / a; total = term
        for _ in range(GAMMA_ITERATIONS):
            ap += 1; term *= x / ap; total += term
            if abs(term) < abs(total) * 1e-16: break
        return min(1.0, total * portable_exp(-x + a * portable_log(x) - lg))
    tiny = 1e-300
    b = x + 1 - a; c = 1 / tiny; d = 1 / b; h = d
    for i in range(1, GAMMA_ITERATIONS):
        an = -i * (i - a); b += 2
        d = an * d + b; d = tiny if abs(d) < tiny else d
        c = b + an / c; c = tiny if abs(c) < tiny else c
        d = 1 / d; delta = d * c; h *= delta
        if abs(delta - 1) < 1e-15: break
    return max(0.0, 1.0 - portable_exp(-x + a * portable_log(x) - lg) * h)


@lru_cache(maxsize=100000)
def gamma_quantile(q, a):
    """x with P(a, x) = q (unit scale): doubling bracket, then bisection deciding each step by gamma_cdf(mid) < q.
    Returns the upper end of the final bracket."""
    lg = log_gamma(a)
    lo, hi = 0.0, max(1.0, a)
    for _ in range(2000):
        if not gamma_cdf(a, hi, lg) < q: break
        lo = hi; hi = hi * 2
    for _ in range(QUANTILE_STEPS):
        mid = (lo + hi) / 2
        if mid <= lo or mid >= hi: break
        if gamma_cdf(a, mid, lg) < q: lo = mid
        else: hi = mid
    return hi


@lru_cache(maxsize=100000)
def _weighted_upper(ns, ks, ws, alpha):
    if len(ns) == 1: return upper(ns[0], ks[0], alpha)
    if any(n == 0 for n in ns): return 1.
    c = [w / n for w, n in zip(ws, ns)]
    y = sequential_sum(ci * k for ci, k in zip(c, ks)); v = sequential_sum(ci * ci * k for ci, k in zip(c, ks)); cm = max(c)
    shape = (y + cm) * (y + cm) / (v + cm * cm); scale = (v + cm * cm) / (y + cm)
    return min(1.0, scale * gamma_quantile(1 - alpha, shape))


def weighted_upper(ns, ks, ws, alpha):
    """Upper (1 - alpha) bound on sum_s w_s p_s from k_s failures in n_s cases (docs/130 §3.2, §5.5). One population: exact
    Clopper-Pearson, `upper(n, k, alpha)`. Several: the gamma bound of Fay and Feuer (1997), c_s = w_s / n_s; with no
    failures it is max_s c_s * ln(1/alpha)."""
    return _weighted_upper(tuple(ns), tuple(ks), tuple(ws), alpha)


@lru_cache(maxsize=100000)
def _weighted_lower(ns, ks, ws, alpha):
    if len(ns) == 1: return 0. if ns[0] == 0 or ks[0] == 0 else 1 - upper(ns[0], ns[0] - ks[0], alpha)
    if any(n == 0 for n in ns): return 0.
    c = [w / n for w, n in zip(ws, ns)]
    y = sequential_sum(ci * k for ci, k in zip(c, ks)); v = sequential_sum(ci * ci * k for ci, k in zip(c, ks))
    if not y > 0: return 0.
    return max(0., (v / y) * gamma_quantile(alpha, y * y / v))


def weighted_lower(ns, ks, ws, alpha):
    """Lower (1 - alpha) bound on sum_s w_s p_s (docs/140 §2.1). One population: exact Clopper-Pearson, 1 - U(n, n - k, alpha).
    Several: the Fay-Feuer (1997) gamma lower bound, (v / Y) * G^-1_{Y^2/v}(alpha); 0 with no counts."""
    return _weighted_lower(tuple(ns), tuple(ks), tuple(ws), alpha)


# ----------------------------------------------------------------------------- two-phase audit (docs/130 §8.1)

AUDIT_ALPHA = 0.005      # per audit half: missed failures among AI-"fine" rows; false flags among AI-flagged rows
AUDIT_SHARE = 0.8        # with an audit, 0.04 of the 0.05 goes to the looks and 0.01 to the two audit halves (once)
AUDIT_FIELDS = ('sampled_fine', 'missed_failures', 'sampled_flagged', 'false_flags')
OLD_AUDIT_MESSAGE = 'This audit was recorded in an older form that cannot be interpreted. Re-run the audit.'
AUDIT_BEFORE_FREEZE_MESSAGE = ('These labels came from a log reader, so a person must check a random sample of them before the plan is frozen. '
                               'Run the audit (aver decision audit), make the plan again from the audited spec, then freeze.')


def audit_of(spec):
    """The spec's two-phase audit, or None."""
    return spec.get('grader_audit') if is_v3(spec) and spec.get('grader_audit') else None


def audit_bounds(audit, alpha=AUDIT_ALPHA):
    """phi = P(truly failed | AI said fine) <= phi_U;  psi = P(truly fine | AI flagged) >= psi_L."""
    m, a, h, b = (audit[k] for k in AUDIT_FIELDS)
    phiU = upper(m, a, alpha) if m > 0 else 1.0
    psiL = (1 - upper(h, h - b, alpha)) if h > 0 else 0.0
    return phiU, psiL


def audited_limit(limit, phiU, psiL):
    """sum_s w_s p_s = phi + (1 - psi - phi) * sum_s w_s r_s (r = rate of AI failures). With phi <= phi_U, psi >= psi_L the
    true rate is within `limit` whenever the bound on r is <= (limit - phi_U) / (1 - psi_L - phi_U). None: no pass possible."""
    if limit is None: return None
    den = 1 - psiL - phiU
    if phiU >= limit or den <= 0: return None
    return (limit - phiU) / den


def audit_block_limit(limit, audit, alpha=AUDIT_ALPHA):
    """The AI-label rate above which the true rate exceeds `limit` whatever the audit allows: p >= (1 - psi) r >= (1 - psi_U) r,
    psi_U the upper bound on false flags. None (never block) when no AI-flagged row was checked."""
    if limit is None: return None
    h, b = audit['sampled_flagged'], audit['false_flags']
    psiU = upper(h, b, alpha) if h > 0 else 1.0
    if psiU >= 1: return None
    return limit / (1 - psiU)


def fine_rows_needed(limit, alpha=AUDIT_ALPHA):
    """Smallest number of AI-"fine" rows that, checked with no missed failures, keep the audited limit at or above half the
    limit (docs/130 §3.3: 324 at 0.032). None when no audit size can."""
    if limit is None or not limit > 0: return None
    def ok(m):
        v = audited_limit(limit, upper(m, 0, alpha), 0.0)
        return v is not None and v >= limit / 2
    lo, hi = 0, 1
    while not ok(hi):
        lo, hi = hi, hi * 2
        if hi > 2 ** 40: return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid): hi = mid
        else: lo = mid
    return hi


# ----------------------------------------------------------------------------- the paired clause (docs/140)
# The confirmation cases run on the finalist AND on the baseline (the option in use now). For each population, b = cases
# only the candidate failed, a = cases only current failed; the clause bounds delta = sum_s w_s (p_c,s - p_u,s) and
# passes when the bound is at most Delta = paired_limit_per_month / monthly_uses. It joins the intersection-union, so it
# takes no extra share of the error allowance (docs/140 §2.1).

PAIRED_CLAUSE = 'paired:failure'
PLANNING_PAIR_CORRELATION = 0.2   # phi between the two options' failures on one case, for planning only (docs/140 §5.3)


def paired_limit(spec):
    """Delta, the paired margin as a rate, or None when the spec has no paired test."""
    v = spec['requirements'].get('paired_limit_per_month')
    return None if v is None else v / spec['monthly_uses']


PAIRED_SCORE_CORRECTION = 1      # E3c: one discordant pair in the candidate's favour per population (docs/140 §2.5)
SCORE_STEPS = 50


def normal_upper_quantile(alpha):
    """z with P(Z > z) = alpha for 0 < alpha < 1/2, from the portable gamma quantile: Z^2 / 2 is gamma with shape 1/2, so
    P(|Z| <= z) = 1 - 2 alpha = P(1/2, z^2 / 2). Bit-identical in the TypeScript twin."""
    return math.sqrt(2 * gamma_quantile(1 - 2 * alpha, .5))


def _restricted_pi01(n, a, b, d0):
    """Tango's restricted maximum-likelihood estimate of the current-only rate when delta = d0 (docs/140 §2.5)."""
    B = b * (1 - d0) + a * (1 - 3 * d0) - 2 * (n - a - b) * d0
    return (B + math.sqrt(max(B * B + 8 * n * a * d0 * (1 - d0), 0.))) / (4 * n)


@lru_cache(maxsize=100000)
def _paired_upper(ns, bs, as_, ws, alpha):
    z = normal_upper_quantile(alpha)
    c = [w / n for w, n in zip(ws, ns)]
    num0 = sequential_sum(ci * (b - a + PAIRED_SCORE_CORRECTION) for ci, b, a in zip(c, bs, as_))
    def ok(d0):
        var = sequential_sum(ci * ci * n * max(2 * _restricted_pi01(n, a, b, d0) + d0 * (1 - d0), 0.) for ci, n, b, a in zip(c, ns, bs, as_))
        return (num0 - d0) / math.sqrt(var) >= -z if var > 0 else num0 - d0 >= 0
    if ok(1.): return 1.
    lo, hi = min(1., max(-1., num0)), 1.
    for _ in range(SCORE_STEPS):
        mid = (lo + hi) / 2
        if ok(mid): lo = mid
        else: hi = mid
    return hi


def paired_upper(ns, bs, as_, ws, alpha):
    """Upper (1 - alpha) bound on delta = sum_s w_s (p_c,s - p_u,s) from paired counts: the continuity-corrected score bound E3c
    (docs/140 §2.5, chosen by the pre-registered rule in §5.1). With c_s = w_s / n_s and a common d0 in every population,
    Z(d0) = (sum_s c_s (b_s - a_s + 1) - d0) / sqrt(sum_s c_s^2 n_s [2 pi01~_s(d0) + d0 (1 - d0)]); the bound is the largest d0 in
    [-1, 1] with Z(d0) >= -z_alpha, by bisection (50 steps, the upper end). One population: Tango (1998) with the correction."""
    if any(n == 0 for n in ns): return 1.
    return _paired_upper(tuple(ns), tuple(bs), tuple(as_), tuple(ws), alpha)


def paired_lower(ns, bs, as_, ws, alpha):
    """Lower bound on delta for blocking: lower bound on the candidate-only rate minus upper bound on the current-only rate,
    each at alpha / 2 (docs/140 §2.3)."""
    return weighted_lower(ns, bs, ws, alpha / 2) - weighted_upper(ns, as_, ws, alpha / 2)


def true_rate_upper(spec, r_up):
    """An upper bound on a true failure rate from an upper bound on the label rate: the two-phase audit's
    p <= phi_U + (1 - psi_L) r (docs/129 §3.5), else the declared grader error, p = (r - e_fp) / (1 - e_fn - e_fp)."""
    audit = audit_of(spec)
    if audit is not None:
        phiU, psiL = audit_bounds(audit)
        return min(1., phiU + (1 - psiL) * r_up)
    ge = grader_error_of(spec)
    return min(1., max(0., (r_up - ge['false_positive']) / (1 - ge['false_negative'] - ge['false_positive'])))


def paired_true_bounds(spec, up, lo):
    """(upper, lower or None) on the true difference from the bounds on the label difference (docs/140 §5.4).
    Declared grader error: r_c - r_u = (1 - e_fn - e_fp)(p_c - p_u), so both bounds are divided by that factor. Two-phase
    audit: delta = f * delta_r with f = 1 - psi - phi; assuming the reader's flag is informative (f >= 0) and with
    psi >= psi_L, delta <= (1 - psi_L) max(up, 0); blocking uses f >= 1 - psi_U - phi_U and blocks only when that is positive."""
    audit = audit_of(spec)
    if audit is None:
        ge = grader_error_of(spec); f = 1 - ge['false_negative'] - ge['false_positive']
        return up / f, lo / f
    phiU, psiL = audit_bounds(audit)
    up_t = (1 - psiL) * up if up >= 0 else 0.
    h, b = audit['sampled_flagged'], audit['false_flags']
    psiU = upper(h, b, AUDIT_ALPHA) if h > 0 else 1.
    f_lo = 1 - psiU - phiU
    lo_t = f_lo * lo if lo > 0 and f_lo > 0 else None
    return up_t, lo_t


def limits_of(spec):
    """L -> (pass limit, block limit) on the scale of the labels the evidence carries: the audited limits with a two-phase audit,
    else the declared grader-error transform (both sides equal)."""
    audit = audit_of(spec)
    if audit is not None:
        phiU, psiL = audit_bounds(audit)
        return lambda L: (audited_limit(L, phiU, psiL), audit_block_limit(L, audit))
    ge = grader_error_of(spec)
    return lambda L: (observed_limit(L, ge), observed_limit(L, ge))


def audited_limits(spec):
    """{clause key: pass limit actually used} for every failure and tolerance clause (None where no pass is possible)."""
    lim = limits_of(spec)
    out = {'average:failure': lim(spec['requirements']['failure_limit'])[0]}
    for s in sorted(spec['strata']):
        if spec['strata'][s]['failure_limit'] is not None: out[f'{s}:failed'] = lim(spec['strata'][s]['failure_limit'])[0]
    for m in tolerated_kinds(spec): out[f'tolerance:{m}'] = lim(tolerance_rate(spec, m))[0]
    return out


def raw_limits(spec):
    """{clause key: the limit as the user stated it} for the clauses in audited_limits."""
    out = {'average:failure': spec['requirements']['failure_limit']}
    for s in sorted(spec['strata']):
        if spec['strata'][s]['failure_limit'] is not None: out[f'{s}:failed'] = spec['strata'][s]['failure_limit']
    for m in tolerated_kinds(spec): out[f'tolerance:{m}'] = tolerance_rate(spec, m)
    return out


def tolerated_kinds(spec):
    if not is_v3(spec): return []
    return sorted(m for m, k in spec['taxonomy']['kinds'].items() if k['tolerance_per_month'] is not None)


def tolerance_rate(spec, kind):
    """N a month at the spec's volume, as a rate (docs/129 §3.1)."""
    return spec['taxonomy']['kinds'][kind]['tolerance_per_month'] / spec['monthly_uses']


# ----------------------------------------------------------------------------- spec

def v1_view(spec, kappa=None):
    """The v1-shaped projection of a v2/v3 spec, for the shared validators. For a v3 spec the v1 consequence is the
    pooled price per failure (docs/129 §3.3) when the caller supplies it."""
    keep = 'schema scope candidates baseline strata definitions requirements monthly_uses consequence_usd budget_runs population_statement'.split()
    out = {k: spec[k] for k in keep if k in spec}
    out['schema'] = 'aver-decision-spec/v1'
    if isinstance(out.get('requirements'), dict) and 'paired_limit_per_month' in out['requirements']:
        out['requirements'] = {k: v for k, v in out['requirements'].items() if k != 'paired_limit_per_month'}
    if kappa is not None: out['consequence_usd'] = kappa
    return out


def rows_v1(rows):
    """Project aver-outcomes/v2 rows (with `mistake`, origin ai-labeled) onto v1 for the shared validators."""
    out = []
    for r in rows:
        if not isinstance(r, dict): out.append(r); continue
        q = {k: v for k, v in r.items() if k != 'mistake'}
        if q.get('schema') == 'aver-outcomes/v2': q['schema'] = 'aver-outcomes/v1'
        if q.get('origin') == 'ai-labeled': q['origin'] = 'customer'
        out.append(q)
    return out


def validate_taxonomy(spec):
    tax = spec['taxonomy']
    fields(tax, 'version kinds', 'spec.taxonomy')
    need(type(tax['version']) is int and tax['version'] >= 1, 'invalid_taxonomy', 'spec.taxonomy.version', 'Taxonomy versions are positive integers.')
    kinds = tax['kinds']
    need(isinstance(kinds, dict) and kinds and all(label(k) and k.isascii() for k in kinds), 'invalid_taxonomy', 'spec.taxonomy.kinds', 'Name at least one mistake kind with an ASCII label.')
    for m, k in kinds.items():
        p = f'spec.taxonomy.kinds.{m}'
        fields(k, 'label severity price_usd tolerance_per_month standard', p)
        need(isinstance(k['label'], str) and 0 < len(k['label']) <= 120, 'invalid_taxonomy', p + '.label', 'Give each kind a short label.')
        need(k['severity'] in ('mistake', 'incident'), 'invalid_taxonomy', p + '.severity', 'Severity is mistake or incident.')
        need(k['price_usd'] is None or (number(k['price_usd']) and 0 <= k['price_usd'] <= 1000000), 'invalid_taxonomy', p + '.price_usd', 'Price is dollars per occurrence, or null.')
        need(k['tolerance_per_month'] is None or (number(k['tolerance_per_month']) and 0 < k['tolerance_per_month'] <= 1e9), 'invalid_taxonomy', p + '.tolerance_per_month', 'Tolerance is a positive number of occurrences a month, or null.')
        need(k['standard'] is None or k['standard'] in STANDARD_KINDS, 'invalid_taxonomy', p + '.standard', f'Standard kind must be one of {", ".join(STANDARD_KINDS)} or null.')
    dep = spec['deployment']
    fields(dep, 'mode description', 'spec.deployment')
    need(dep['mode'] in ('normal', 'agentic') and isinstance(dep['description'], str), 'invalid_deployment', 'spec.deployment', 'Deployment mode is normal or agentic, with a description.')
    audit = spec.get('grader_audit')
    if audit is not None:
        need(isinstance(audit, dict), 'invalid_audit', 'spec.grader_audit', 'An audit is an object of counts.')
        need(not ({'sampled', 'false_negatives', 'false_positives'} & set(audit)), 'invalid_audit', 'spec.grader_audit', OLD_AUDIT_MESSAGE)
        fields(audit, ' '.join(AUDIT_FIELDS) + (' kinds_checked' if 'kinds_checked' in audit else ''), 'spec.grader_audit')
        need(all(type(audit[k]) is int and audit[k] >= 0 for k in AUDIT_FIELDS) and audit['missed_failures'] <= audit['sampled_fine'] and audit['false_flags'] <= audit['sampled_flagged'],
             'invalid_audit', 'spec.grader_audit', 'An audit counts the cases the AI called fine that a person checked and how many of them had in fact failed, and the cases the AI flagged that a person checked and how many of them were in fact fine.')
        if 'kinds_checked' in audit:
            kc = audit['kinds_checked']
            need(isinstance(kc, list) and all(isinstance(m, str) and m in kinds for m in kc) and kc == sorted(set(kc)), 'invalid_audit', 'spec.grader_audit.kinds_checked', 'Kinds checked are distinct kinds from the taxonomy, in order.')


def validate_mistakes(spec, rows):
    """docs/129 §2.2: the `mistake` field against the taxonomy."""
    kinds = spec['taxonomy']['kinds']
    for i, r in enumerate(rows):
        p = f'evidence[{i}]'
        need(r.get('schema') in ('aver-outcomes/v1', 'aver-outcomes/v2'), 'unsupported_schema', p, 'Use aver-outcomes/v1 or v2 rows.')
        if r.get('schema') == 'aver-outcomes/v1':
            need(not tolerated_kinds(spec), 'mistake_required', p, 'Tolerances need rows that name the kind of mistake (aver-outcomes/v2).')
            continue
        need('mistake' in r, 'mistake_required', p, 'aver-outcomes/v2 rows carry a mistake kind or null.')
        m = r['mistake']
        failed = r['completed'] is False or r['incident'] is True
        if r['completed'] is None and r['incident'] is not True:
            need(m is None, 'invalid_mistake', p + '.mistake', 'An unknown outcome has no mistake kind.')
        elif failed:
            need(isinstance(m, str) and m in kinds, 'invalid_mistake', p + '.mistake', 'A failure or incident names a kind from the taxonomy.')
            if kinds[m]['severity'] == 'incident': need(r['incident'] is True, 'invalid_mistake', p + '.mistake', 'An incident-severity kind is an incident.')
        else:
            need(m is None, 'invalid_mistake', p + '.mistake', 'A completed case without incident has no mistake kind.')
        need(r['origin'] in ORIGINS_V2, 'invalid_origin', p, 'Declare synthetic, external, customer or ai-labeled provenance.')


def validate_spec_v2(spec):
    if is_v3(spec):
        fields(spec, 'schema scope candidates baseline strata definitions requirements monthly_uses consequence_usd budget_runs population_statement label_cost_usd horizon_months current_failure_rate grader_error taxonomy deployment' + (' grader_audit' if 'grader_audit' in spec else ''), 'spec')
        need(spec['schema'] == SPEC_SCHEMA_V3, 'unsupported_schema', 'spec.schema', 'A taxonomy needs aver-decision-spec/v3.')
    else:
        fields(spec, 'schema scope candidates baseline strata definitions requirements monthly_uses consequence_usd budget_runs population_statement label_cost_usd horizon_months current_failure_rate grader_error', 'spec')
        need(spec['schema'] == SPEC_SCHEMA, 'unsupported_schema', 'spec.schema', 'Expected aver-decision-spec/v2.')
    req = spec.get('requirements')
    pl = req.get('paired_limit_per_month') if isinstance(req, dict) else None
    need(pl is None or (number(pl) and 0 <= pl <= 1e9), 'invalid_paired_limit', 'spec.requirements.paired_limit_per_month',
         'The paired limit is how many more failures a month than current you accept: zero or more, or null for no paired test.')
    validate_spec(v1_view(spec))
    v = spec['label_cost_usd']
    need(v is None or (number(v) and 0 <= v <= 100000), 'invalid_label_cost', 'spec.label_cost_usd', 'Use a nonnegative dollar cost per labelled case, or null for unknown.')
    h = spec['horizon_months']
    need(number(h) and int(h) == h and 1 <= h <= 120, 'invalid_horizon', 'spec.horizon_months', 'Use an integer number of months between 1 and 120.')
    f = spec['current_failure_rate']
    need(f is None or (number(f) and 0 <= f <= 1), 'invalid_limit', 'spec.current_failure_rate', 'Use a measured rate between zero and one, or null to use the pilot.')
    fields(spec['grader_error'], 'false_negative false_positive', 'spec.grader_error')
    e = spec['grader_error']
    need(all(number(x) and 0 <= x < .5 for x in e.values()) and e['false_negative'] + e['false_positive'] < 1, 'invalid_grader_error', 'spec.grader_error', 'Grader error rates must be between 0 and 0.5 and sum below one.')
    if is_v3(spec): validate_taxonomy(spec)
    digest(spec)
    return spec


# ----------------------------------------------------------------------------- prices and shares (docs/129 §3.3)

def mistake_shares(spec, pilot_rows, candidate_id):
    """Share of a candidate's pilot failures by kind; pooled over all candidates when it has none; uniform when nobody has any."""
    kinds = sorted(spec['taxonomy']['kinds'])
    def count(rows):
        c = {m: 0 for m in kinds}
        for r in rows:
            m = r.get('mistake')
            if m in c: c[m] += 1
        return c
    own = count([r for r in pilot_rows if r['candidate_id'] == candidate_id])
    total = sequential_sum(own.values())
    if total == 0:
        own = count(pilot_rows); total = sequential_sum(own.values())
    if total == 0: return {m: 1 / len(kinds) for m in kinds}
    return {m: own[m] / total for m in kinds}


def effective_kappa(spec, pilot_rows, candidate_id):
    """kappa_c = sum_m price_m * share_{c,m}; unpriced kinds contribute nothing."""
    if not is_v3(spec): return spec['consequence_usd']
    kinds = spec['taxonomy']['kinds']
    sh = mistake_shares(spec, pilot_rows, candidate_id)
    return sequential_sum((kinds[m]['price_usd'] or 0) * sh[m] for m in sorted(kinds))


def pooled_kappa(spec, pilot_rows):
    if not is_v3(spec): return spec['consequence_usd']
    kinds = spec['taxonomy']['kinds']
    sh = mistake_shares(spec, pilot_rows, '__pooled__')
    return sequential_sum((kinds[m]['price_usd'] or 0) * sh[m] for m in sorted(kinds))


def kind_rates(spec, pilot_rows, candidate_id, rate):
    """Planning rates per tolerated kind at candidate failure rate `rate`: the candidate's pilot share of that kind,
    or, when the pilot never saw it, the rate at which seeing none in the whole pilot was an even chance, capped at half the
    tolerance (docs/129 §3.3)."""
    out = {}
    if not is_v3(spec): return out
    sh = mistake_shares(spec, pilot_rows, candidate_id)
    own = {m: 0 for m in spec['taxonomy']['kinds']}
    for r in pilot_rows:
        if r['candidate_id'] == candidate_id and r.get('mistake') in own: own[r['mistake']] += 1
    n_pooled = len(pilot_rows)
    for m in tolerated_kinds(spec):
        # never seen in the pilot: the rate at which seeing none in the whole pilot was an even chance, capped at half the
        # tolerance (the same stand-in the overall limit uses when the pilot is uninformative)
        out[m] = rate * sh[m] if own[m] > 0 else min(tolerance_rate(spec, m) / 2, upper(n_pooled, 0, .5) if n_pooled else tolerance_rate(spec, m) / 2)
    return out


def weighted_floor(spec, rate, looks, spending, q, mode='clause', ceiling=200000):
    """Smallest total of cases at which a weighted clause with limit `rate` can pass at the final look with no occurrences,
    under the plan's allocation shares and the gamma bound at alpha = 0.05/q * s_K (docs/130 §5.5). None when it cannot
    within the ceiling."""
    S = len(spec['strata']); a = ALPHA / q * spending[-1]
    if rate is None: return None
    strata = sorted(spec['strata']); ws = [spec['strata'][s]['weight'] for s in strata]
    def ok(total):
        al = allocation(spec, total, looks, mode)
        return weighted_upper([al['final'][s] for s in strata], [0] * S, ws, a) <= rate
    lo, hi = looks * S, ceiling
    if not ok(hi): return None
    while hi - lo > looks:
        mid = (lo + hi) // 2
        if ok(mid): hi = mid
        else: lo = mid
    return hi


def tolerance_floor(spec, kind, looks, spending, q, mode='clause', ceiling=200000):
    """Smallest total of cases at which a tolerated kind never seen can still pass at the final look (docs/129 §3.1)."""
    return weighted_floor(spec, observed_limit(tolerance_rate(spec, kind), grader_error_of(spec)), looks, spending, q, mode, ceiling)


def saving_floor(spec, looks, spending, q, mode='clause', ceiling=200000):
    """Smallest total of cases at which the paired clause could pass at the final look with no discordant pairs (the two options
    failed on exactly the same cases), docs/140 §5. None when there is no paired limit, when the limit is zero (then a pass
    needs cases that current failed and the candidate did not, so no such floor exists), or when it cannot within the ceiling."""
    D = paired_limit(spec)
    if D is None or not D > 0: return None
    ge = grader_error_of(spec); D_obs = D * (1 - ge['false_negative'] - ge['false_positive'])
    S = len(spec['strata']); a = ALPHA / q * spending[-1]
    strata = sorted(spec['strata']); ws = [spec['strata'][s]['weight'] for s in strata]
    def ok(total):
        al = allocation(spec, total, looks, mode)
        return paired_upper([al['final'][s] for s in strata], [0] * S, [0] * S, ws, a) <= D_obs
    lo, hi = looks * S, ceiling
    if not ok(hi): return None
    while hi - lo > looks:
        mid = (lo + hi) // 2
        if ok(mid): hi = mid
        else: lo = mid
    return hi


def _ai_fine(r):
    return r['completed'] is True and r['incident'] is not True


def _ai_flagged(r):
    return r['completed'] is False or r['incident'] is True


def audit_sample(rows, n_fine, n_flagged, seed=7):
    """A seeded two-phase sample of AI-labelled rows for a person to relabel (docs/130 §8.1): `n_fine` rows the reader called
    fine (completed, no incident) and `n_flagged` rows it called failures; rows with an unknown outcome are left out. All of a
    group is taken when it has fewer rows than asked. Each entry says which phase it belongs to."""
    need(type(n_fine) is int and n_fine >= 0 and type(n_flagged) is int and n_flagged >= 0, 'invalid_audit', 'audit', 'Ask for a whole number of cases of each kind.')
    rnd = mulberry32(seed)
    ai = [r for r in rows if r.get('origin') == 'ai-labeled']
    key = lambda r: (r['candidate_id'], r['case_id'])
    fine = sorted([r for r in ai if _ai_fine(r)], key=key)
    flagged = sorted([r for r in ai if _ai_flagged(r)], key=key)
    def pick(pool, k):
        pool = list(pool); out = []
        while pool and len(out) < k:
            i = int(rnd() * len(pool)); out.append(pool.pop(i))
        return out
    chosen = [(r, 'fine') for r in pick(fine, n_fine)] + [(r, 'flagged') for r in pick(flagged, n_flagged)]
    return [{'candidate_id': r['candidate_id'], 'case_id': r['case_id'], 'ai_completed': r['completed'], 'ai_incident': r['incident'], 'ai_mistake': r.get('mistake'), 'phase': ph} for r, ph in chosen]


def apply_audit(spec, sample, person_labels):
    """person_labels: {(candidate_id, case_id): {'completed': bool, 'incident': bool[, 'mistake': kind or None]}}. Returns a copy of
    the spec with the two-phase grader_audit. When the person's labels name kinds, `kinds_checked` lists the kinds the person
    confirmed at least once among AI-flagged rows."""
    m = a = h = b = 0
    with_kinds = False; confirmed = set()
    for s in sample:
        p = person_labels.get((s['candidate_id'], s['case_id']))
        need(p is not None, 'audit_incomplete', 'audit', 'Every sampled case needs the person\'s label.')
        ai_bad = s['ai_completed'] is False or s['ai_incident'] is True
        person_bad = p['completed'] is False or p['incident'] is True
        if 'mistake' in p: with_kinds = True
        if ai_bad:
            h += 1; b += (not person_bad)
            if person_bad and p.get('mistake') is not None and p.get('mistake') == s.get('ai_mistake'): confirmed.add(p['mistake'])
        else:
            m += 1; a += person_bad
    out = dict(spec); out['grader_audit'] = {'sampled_fine': m, 'missed_failures': int(a), 'sampled_flagged': h, 'false_flags': int(b)}
    if with_kinds: out['grader_audit']['kinds_checked'] = sorted(confirmed)
    return validate_spec_v2(out)


def observed_limit(limit, grader_error):
    """p <= L  iff  observed rate r <= L(1-e_fn-e_fp)+e_fp  when r = p(1-e_fn) + (1-p)e_fp."""
    if limit is None: return None
    fn, fp = grader_error['false_negative'], grader_error['false_positive']
    return limit * (1 - fn - fp) + fp


# ----------------------------------------------------------------------------- allocation and schedule

def allocation(spec, cap_total, looks, mode='clause'):
    """Final cases per population and cumulative cases per look. Rounded to multiples of `looks`."""
    strata = sorted(spec['strata'])
    S = len(strata)
    if mode == 'equal':
        share = {s: 1 / S for s in strata}
    else:
        # every population but the heaviest gets at least half an equal share of the total (so edge clauses can
        # resolve); the heaviest carries the average and takes the remainder. Falls back to traffic weights if
        # the remainder would not be positive.
        heaviest = max(strata, key=lambda s: (spec['strata'][s]['weight'], s))
        share = {s: max(spec['strata'][s]['weight'], 1 / (2 * S)) for s in strata if s != heaviest}
        rest = 1 - sequential_sum(share.values())
        if rest <= 0:
            share = {s: spec['strata'][s]['weight'] for s in strata}
        else:
            share[heaviest] = rest
    final = {s: max(looks, int(round(cap_total * share[s] / looks)) * looks) for s in strata}
    schedule = {s: [int(round(final[s] * (j + 1) / looks)) for j in range(looks)] for s in strata}
    return {'final': final, 'schedule': schedule, 'total': sequential_sum(final.values()), 'mode': mode}


def cumulative_executions(alloc, j):
    return sequential_sum(alloc['schedule'][s][j] for s in alloc['schedule'])


# ----------------------------------------------------------------------------- the staged rule

def decide(counts, spec, q, looks, j, spending):
    """counts[s] = {'n','failed','incident','unknown_failed','unknown_incident'[, 'by_mistake': {kind: k}]} at look j (1-based).
    Returns state in pass/block/continue/exhausted plus the clauses. v3 adds one clause per tolerated kind (docs/129 §3.1).
    docs/130 §8: no split across populations; the average and each tolerance are bounded by `weighted_upper`; per-population
    clauses by Clopper-Pearson at alpha_U (intersection-union); with a two-phase audit, 0.01 of the 0.05 is the audit's and
    limits are the audited limits. A clause whose audited limit does not exist cannot pass (state refuse)."""
    share = AUDIT_SHARE if audit_of(spec) is not None else 1.0
    aU = share * ALPHA / q * spending[j - 1]
    aL = aU / 2
    lim = limits_of(spec)
    lim_avg, blk_avg = lim(spec['requirements']['failure_limit'])
    lim_inc = spec['requirements']['incident_limit']
    strata = sorted(spec['strata'])
    for s in strata:
        if counts[s]['n'] == 0:
            return {'state': 'continue', 'clauses': [], 'avg_upper': 1., 'avg_lower': 0., 'estimate': None, 'binding': 'no evidence', 'by_mistake': {}}
    ws = [spec['strata'][s]['weight'] for s in strata]; ns = [counts[s]['n'] for s in strata]
    clauses, avg_l, est_avg, unknown_total, passed, blocked, binding = [], 0., 0., 0, True, False, None
    def judge(unk, hi, lo, pass_limit, block_limit):
        return 'refuse' if unk else 'pass' if pass_limit is not None and hi <= pass_limit else 'block' if block_limit is not None and lo > block_limit else 'refuse'
    for s, w in zip(strata, ws):
        c = counts[s]; n = c['n']
        for metric in ['failed', 'incident']:
            bad, unk = c[metric], c['unknown_' + metric]
            unknown_total += unk
            hi = upper(n, bad + unk, aU)
            lo = 1 - upper(n, n - bad, aL)
            est = bad / n if not unk else None
            if metric == 'failed':
                avg_l += w * lo; est_avg += w * (est or 0)
                present = spec['strata'][s]['failure_limit'] is not None
                limit, block_limit = lim(spec['strata'][s]['failure_limit']) if present else (None, None)
            else:
                present = lim_inc is not None
                limit, block_limit = lim_inc, lim_inc
            if present:
                state = judge(unk, hi, lo, limit, block_limit)
                if state != 'pass' and passed: binding = binding or f'{s} {"failure" if metric == "failed" else "incident"} limit'
                if state != 'pass': passed = False
                if state == 'block': blocked = True
                clauses.append({'key': f'{s}:{metric}', 'population': s, 'metric': 'failure' if metric == 'failed' else 'incident', 'estimate': est, 'lower': lo, 'upper': hi, 'limit': limit, 'cases': n, 'unknown': unk, 'state': state})
    avg_u = weighted_upper(ns, [counts[s]['failed'] + counts[s]['unknown_failed'] for s in strata], ws, aU)
    avg_state = judge(unknown_total, avg_u, avg_l, lim_avg, blk_avg)
    if avg_state != 'pass': passed = False; binding = binding or 'average failure limit'
    if avg_state == 'block': blocked = True
    clauses.insert(0, {'key': 'average:failure', 'population': 'all', 'metric': 'failure', 'estimate': None if unknown_total else est_avg, 'lower': avg_l, 'upper': avg_u, 'limit': lim_avg, 'cases': sequential_sum(ns), 'unknown': unknown_total, 'state': avg_state})
    by_mistake = {}
    if is_v3(spec):
        V = spec['monthly_uses']; kinds = spec['taxonomy']['kinds']; total_n = sequential_sum(ns)
        for m in sorted(kinds):
            km = [counts[s].get('by_mistake', {}).get(m, 0) for s in strata]; us = [counts[s]['unknown_failed'] for s in strata]
            hi = weighted_upper(ns, [k + u for k, u in zip(km, us)], ws, aU)
            lo = sequential_sum(w * (1 - upper(n, n - k, aL)) for w, n, k in zip(ws, ns, km))
            est = sequential_sum(w * k / n for w, n, k in zip(ws, ns, km))
            count = sequential_sum(km); unk = sequential_sum(us)
            row = {'kind': m, 'cases': total_n, 'count': count, 'unknown': unk, 'rate': est, 'upper': hi, 'lower': lo,
                   'per_month': est * V, 'per_month_upper': hi * V, 'per_month_lower': lo * V, 'tolerance_per_month': kinds[m]['tolerance_per_month'], 'price_usd': kinds[m]['price_usd']}
            if kinds[m]['tolerance_per_month'] is not None:
                lim_m, blk_m = lim(tolerance_rate(spec, m))
                st = judge(unk, hi, lo, lim_m, blk_m)
                if st != 'pass' and passed: binding = binding or f'{m} tolerance'
                if st != 'pass': passed = False
                if st == 'block': blocked = True
                clauses.append({'key': f'tolerance:{m}', 'population': 'all', 'metric': 'tolerance', 'kind': m, 'estimate': None if unk else est, 'lower': lo, 'upper': hi, 'limit': lim_m, 'cases': total_n, 'unknown': unk, 'state': st})
                row['state'] = st
            by_mistake[m] = row
    paired = None
    D = paired_limit(spec)
    if D is not None and all('paired' in counts[s] for s in strata):
        # docs/140 §5: counts[s]['paired'] = {n, b, a, b_lo, a_lo, unknown}; b/a count an unknown outcome against the
        # candidate (upper side), b_lo/a_lo in its favour (lower side)
        P = [counts[s]['paired'] for s in strata]; pn = [x['n'] for x in P]
        up_r = paired_upper(pn, [x['b'] for x in P], [x['a'] for x in P], ws, aU)
        lo_r = paired_lower(pn, [x['b_lo'] for x in P], [x['a_lo'] for x in P], ws, aL)
        up, lo = paired_true_bounds(spec, up_r, lo_r)
        unk = sequential_sum(x['unknown'] for x in P)
        est = None if unk else sequential_sum(w * (x['b'] - x['a']) / x['n'] for w, x in zip(ws, P))
        st = judge(unk, up, 0. if lo is None else lo, D, None if lo is None else D)
        if st != 'pass' and passed: binding = binding or 'paired limit'
        if st != 'pass': passed = False
        if st == 'block': blocked = True
        clauses.append({'key': PAIRED_CLAUSE, 'population': 'all', 'metric': 'paired', 'estimate': est, 'lower': lo, 'upper': up, 'limit': D,
                        'cases': sequential_sum(pn), 'unknown': unk, 'state': st})
        V = spec['monthly_uses']
        paired = {'cases': sequential_sum(pn), 'discordant': sequential_sum(x['b'] + x['a'] for x in P),
                  'candidate_only': sequential_sum(x['b'] for x in P), 'current_only': sequential_sum(x['a'] for x in P), 'unknown': unk,
                  'delta': est, 'delta_upper': up, 'delta_lower': lo, 'per_month': None if est is None else est * V,
                  'per_month_upper': up * V, 'per_month_lower': None if lo is None else lo * V,
                  'limit_per_month': spec['requirements']['paired_limit_per_month'], 'state': st}
    state = 'block' if blocked else 'pass' if passed else ('continue' if j < looks else 'exhausted')
    out = {'state': state, 'clauses': clauses, 'avg_upper': avg_u, 'avg_lower': avg_l, 'estimate': None if unknown_total else est_avg, 'binding': None if state in ('pass', 'block') else binding, 'by_mistake': by_mistake}
    if paired is not None: out['paired'] = paired
    return out


# ----------------------------------------------------------------------------- simulation (planning only)

def mulberry32(seed):
    """The same 32-bit generator as the TypeScript twin, so planning numbers agree across languages."""
    a = seed & 0xFFFFFFFF
    def rnd():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((a ^ (a >> 15)) * (a | 1)) & 0xFFFFFFFF
        t = (t ^ ((t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return rnd


def _binomial(rnd, n, p):
    if p <= 0 or n <= 0: return 0
    if p >= 1: return n
    lq = math.log1p(-p); t = 0; c = 0
    while True:
        t += int(math.floor(math.log(1 - rnd()) / lq)) + 1
        if t > n: return c
        c += 1


def pair_cells(p_c, p_u, phi):
    """(pi11, pi10, pi01) for two binary outcomes with rates p_c, p_u and phi correlation, the both-fail cell clamped to its
    feasible range. Planning only (docs/140 §5.3)."""
    sd = math.sqrt(p_c * (1 - p_c) * p_u * (1 - p_u))
    p11 = min(min(p_c, p_u), max(max(0., p_c + p_u - 1), p_c * p_u + phi * sd))
    return p11, p_c - p11, p_u - p11


def simulate(spec, alloc, looks, spending, rate, incident_rate, reps, seed, q=1, kinds=None, paired=None, saving=None):
    """Operating characteristics of the staged rule for a candidate whose true failure rate is `rate`
    (a number, or a map by population) and incident rate `incident_rate`, all labels known. `kinds` maps each tolerated
    kind to its true rate; occurrences are drawn independently of the failure count (planning approximation, docs/129 §3.3).
    `paired` = {'current_rate': number or map, 'correlation': phi}: current runs the same cases; each case's pair of outcomes
    is drawn from the four cells with that phi correlation, and the paired clause is decided (docs/140 §5.3)."""
    rng = mulberry32(seed)
    strata = sorted(spec['strata'])
    rates = rate if isinstance(rate, dict) else {s: rate for s in strata}
    kinds = kinds or {}; kind_names = sorted(kinds)
    cells = None
    if paired is not None:
        cr = paired['current_rate']; cur = cr if isinstance(cr, dict) else {s: cr for s in strata}
        cells = {s: pair_cells(rates[s], cur[s], paired['correlation']) for s in strata}
    out = {'pass': 0, 'block': 0, 'unresolved': 0, 'cases': 0., 'pass_by_look': [0] * looks, 'block_by_look': [0] * looks}
    shown = 0
    total = alloc['total']
    for _ in range(reps):
        t = {s: 0 for s in strata}; kf = {s: 0 for s in strata}; ki = {s: 0 for s in strata}; km = {s: {m: 0 for m in kind_names} for s in strata}
        pb = {s: 0 for s in strata}; pa = {s: 0 for s in strata}
        done = False
        for j in range(1, looks + 1):
            for s in strata:
                dn = alloc['schedule'][s][j - 1] - t[s]
                if cells is None:
                    kf[s] += _binomial(rng, dn, rates[s])
                else:
                    p11, p10, p01 = cells[s]
                    n11 = _binomial(rng, dn, p11)
                    n10 = _binomial(rng, dn - n11, p10 / (1 - p11)) if p11 < 1 else 0
                    n01 = _binomial(rng, dn - n11 - n10, p01 / (1 - p11 - p10)) if p11 + p10 < 1 else 0
                    kf[s] += n11 + n10; pb[s] += n10; pa[s] += n01
                ki[s] += _binomial(rng, dn, incident_rate)
                for m in kind_names: km[s][m] += _binomial(rng, dn, kinds[m])
                t[s] = alloc['schedule'][s][j - 1]
            cs = {s: {'n': t[s], 'failed': kf[s], 'incident': ki[s], 'unknown_failed': 0, 'unknown_incident': 0, 'by_mistake': km[s]} for s in strata}
            if cells is not None:
                for s in strata: cs[s]['paired'] = {'n': t[s], 'b': pb[s], 'a': pa[s], 'b_lo': pb[s], 'a_lo': pa[s], 'unknown': 0}
            d = decide(cs, spec, q, looks, j, spending)
            if d['state'] in ('pass', 'block'):
                if d['state'] == 'pass' and saving is not None:
                    lo = planning_saving_lower(spec, d, saving, paired is not None)
                    if lo is not None and lo > 0: shown += 1
                out[d['state']] += 1; out[d['state'] + '_by_look'][j - 1] += 1; out['cases'] += cumulative_executions(alloc, j - 1); done = True; break
        if not done:
            out['unresolved'] += 1; out['cases'] += total
    for k in ('pass', 'block', 'unresolved', 'cases'): out[k] /= reps
    out['pass_by_look'] = [x / reps for x in out['pass_by_look']]; out['block_by_look'] = [x / reps for x in out['block_by_look']]
    if saving is not None:
        out['p_saving_shown'] = None if any(v is None for v in saving.values()) else shown / reps
    return out


def planning_saving_lower(spec, decision, saving, paired=False):
    """Apply the shipped savings rule at the simulated stopping round. Costs and mistake mix stay pilot assumptions."""
    if any(v is None for v in saving.values()): return None
    kc, ku, xc, xu, current = (saving[k] for k in ('candidate_kappa', 'current_kappa', 'candidate_running', 'current_running', 'current_monthly'))
    hi = true_rate_upper(spec, decision['avg_upper'])
    if not paired: return current - spec['monthly_uses'] * (xc + kc * hi)
    delta = decision['paired']['delta_upper']
    return spec['monthly_uses'] * ((xu - xc) - min(kc, ku) * delta - max(0., kc - ku) * hi)


def design_rates(spec, exploration, candidate_id):
    """Three rates the plan is judged under. central: pooled pilot estimate when the pilot holds >= 3 failures,
    else half the limit; optimistic: half the central (floor L/8); pessimistic: the pilot's one-sided 75% upper,
    capped at 0.75 L (decided 2026-09-30)."""
    L = spec['requirements']['failure_limit']
    rows = [r for r in exploration['pilot'] if r['candidate_id'] == candidate_id]
    k = 0; n = 0; est = 0.
    for s in spec['strata']:
        rs = [r for r in rows if r['stratum'] == s]
        ks = sequential_sum(r['completed'] is not True for r in rs)
        k += ks; n += len(rs); est += spec['strata'][s]['weight'] * (ks / len(rs) if rs else 0)
    central = min(est, .75 * L) if k >= 3 else L / 2
    why = 'pooled pilot estimate' if k >= 3 else 'the pilot has fewer than 3 failures, so half your limit stands in'
    return {'optimistic': max(central / 2, L / 8), 'central': central, 'pessimistic': min(.75 * L, upper(n, k, .25)) if n else .75 * L, 'central_why': why}


def economics(spec, exploration, candidate_id, rate):
    """Monthly saving if the candidate is supported, at the pilot's costs and the design rate. Assumption, not measurement."""
    ex = {r['candidate_id']: r for r in exploration['ranking']}
    cur, cand = ex[spec['baseline']], ex[candidate_id]
    kappa = effective_kappa(spec, exploration['pilot'], candidate_id); kappa_cur = effective_kappa(spec, exploration['pilot'], spec['baseline'])
    if kappa is None or kappa_cur is None or cur['execution_cost_usd'] is None or cand['execution_cost_usd'] is None: return None
    f_cur = spec['current_failure_rate'] if spec['current_failure_rate'] is not None else cur['failure_rate']
    return spec['monthly_uses'] * ((cur['execution_cost_usd'] + kappa_cur * f_cur) - (cand['execution_cost_usd'] + kappa * rate))


def operating_characteristics(spec, exploration, candidate_id, alloc, looks, spending, incident_rate, reps=400, seed=20260930, q=1, paired=None, audit_cost=0.):
    """q: the number of finalists, which sets the decision level (docs/130 §8.3). paired: the planning assumption for a paired
    plan ({'current_rate', 'correlation'}, docs/140 §5.3); each case is then run and labelled on both options, so the price per
    case covers both. audit_cost: what the person's audit of AI labels costs, added once to the expected spend and taken off
    the net value (docs/131 §2.6)."""
    dr = design_rates(spec, exploration, candidate_id)
    ex = {r['candidate_id']: r for r in exploration['ranking']}
    per_case = None if spec['label_cost_usd'] is None or ex[candidate_id]['execution_cost_usd'] is None else ex[candidate_id]['execution_cost_usd'] + spec['label_cost_usd']
    if paired is not None and per_case is not None:
        x_u = ex[spec['baseline']]['execution_cost_usd']
        per_case = None if x_u is None else per_case + x_u + spec['label_cost_usd']
    out = {'design_rates': dr, 'incident_rate': incident_rate, 'reps': reps, 'seed': seed, 'per_case_usd': per_case, 'scenarios': {}}
    if paired is not None: out['paired'] = paired
    if audit_cost: out['audit_cost_usd'] = audit_cost
    saving = {'candidate_kappa': effective_kappa(spec, exploration['pilot'], candidate_id),
              'current_kappa': effective_kappa(spec, exploration['pilot'], spec['baseline']),
              'candidate_running': ex[candidate_id]['execution_cost_usd'], 'current_running': ex[spec['baseline']]['execution_cost_usd'],
              'current_monthly': current_monthly(spec, exploration)}
    out['savings_basis'] = 'same_cases' if paired is not None else 'current_logged_rate'
    for name, i in (('optimistic', 0), ('central', 1), ('pessimistic', 2)):
        sim = simulate(spec, alloc, looks, spending, dr[name], incident_rate, reps, seed + i, q=q, kinds=kind_rates(spec, exploration['pilot'], candidate_id, dr[name]), paired=paired, saving=saving)
        if candidate_id == spec['baseline']: sim['p_saving_shown'] = 0.
        sim['p_limits_met'] = sim['pass']
        sim['p_clear_answer'] = sim['pass'] + sim['block']
        b = economics(spec, exploration, candidate_id, dr[name])
        sim['monthly_saving_usd'] = b
        sim['expected_spend_usd'] = None if per_case is None else sim['cases'] * per_case + audit_cost
        sim['value_if_every_pass_used_usd'] = None if per_case is None or b is None else sim['pass'] * b * spec['horizon_months'] - sim['cases'] * per_case - audit_cost
        sim['net_value_usd'] = None if per_case is None or b is None or sim['p_saving_shown'] is None else sim['p_saving_shown'] * max(0., b) * spec['horizon_months'] - sim['cases'] * per_case - audit_cost
        out['scenarios'][name] = sim
    return out


def best_cap(spec, exploration, candidate_id, looks, spending, incident_rate, mode='clause', reps=300, seed=20260930, grid=GRID, q=1):
    """Net value of checking over a grid of caps at the central rate (3-point smoothed); the best cap and the break-even prices."""
    dr = design_rates(spec, exploration, candidate_id)
    ex = {r['candidate_id']: r for r in exploration['ranking']}
    b = economics(spec, exploration, candidate_id, dr['central'])
    per_case = None if spec['label_cost_usd'] is None or ex[candidate_id]['execution_cost_usd'] is None else ex[candidate_id]['execution_cost_usd'] + spec['label_cost_usd']
    saving = {'candidate_kappa': effective_kappa(spec, exploration['pilot'], candidate_id), 'current_kappa': effective_kappa(spec, exploration['pilot'], spec['baseline']),
              'candidate_running': ex[candidate_id]['execution_cost_usd'], 'current_running': ex[spec['baseline']]['execution_cost_usd'], 'current_monthly': current_monthly(spec, exploration)}
    pts = []
    for cap in grid:
        al = allocation(spec, cap, looks, mode)
        sim = simulate(spec, al, looks, spending, dr['central'], incident_rate, reps, seed, q=q, kinds=kind_rates(spec, exploration['pilot'], candidate_id, dr['central']), saving=saving)
        shown = 0. if candidate_id == spec['baseline'] else sim['p_saving_shown']
        nv = None if per_case is None or b is None or shown is None else shown * max(0., b) * spec['horizon_months'] - sim['cases'] * per_case
        pts.append({'cap': al['total'], 'pass': sim['pass'], 'p_saving_shown': shown, 'cases': sim['cases'], 'net_value_usd': nv})
    if per_case is None or b is None:
        return {'points': pts, 'best': None, 'monthly_saving_usd': b, 'per_case_usd': per_case, 'reason': 'economics_unknown'}
    for i, p in enumerate(pts):
        window = [x['net_value_usd'] for x in pts[max(0, i - 1):i + 2]]
        p['net_value_smoothed_usd'] = sequential_sum(window) / len(window)
    best = max(pts, key=lambda p: p['net_value_smoothed_usd'])
    H = spec['horizon_months']
    uses_star = None if b <= 0 or best['p_saving_shown'] <= 0 else math.ceil(best['cases'] * per_case / (best['p_saving_shown'] * (b / spec['monthly_uses']) * H) / 1000) * 1000
    label_star = None if best['p_saving_shown'] <= 0 else best['p_saving_shown'] * max(0., b) * H / best['cases'] - ex[candidate_id]['execution_cost_usd']
    return {'points': pts, 'best': best, 'monthly_saving_usd': b, 'per_case_usd': per_case,
            'break_even_uses_per_month': uses_star, 'break_even_label_cost_usd': None if label_star is None else max(0., label_star),
            'worth_it': best['net_value_smoothed_usd'] >= 0}


def cases_for_target(spec, alloc_mode, looks, spending, rate, incident_rate, limit, target=.8, reps=160, seed=5, ceiling=12000, kinds=None, q=1):
    """Smallest cap (on a 60-case grid) whose pass probability at `rate` reaches `target` under failure limit `limit`; None if not within `ceiling`."""
    s2 = dict(spec); s2['requirements'] = dict(spec['requirements'], failure_limit=limit)
    def ok(cap):
        return simulate(s2, allocation(s2, cap, looks, alloc_mode), looks, spending, rate, incident_rate, reps, seed, q=q, kinds=kinds)['pass'] >= target
    lo, hi = 300, ceiling
    if not ok(hi): return None
    for _ in range(6):
        mid = int(round((lo + hi) / 2 / 60)) * 60
        if ok(mid): hi = mid
        else: lo = mid
    return allocation(s2, hi, looks, alloc_mode)['total']


def move_the_bar(spec, exploration, candidate_id, looks, spending, incident_rate, mode='clause', q=1):
    dr = design_rates(spec, exploration, candidate_id)
    L = spec['requirements']['failure_limit']
    rows = []
    for factor, word in ((.75, 'stricter'), (1., 'yours'), (1.25, 'looser')):
        lim = L * factor
        kappa = effective_kappa(spec, exploration['pilot'], candidate_id)
        rows.append({'limit': lim, 'word': word,
                     'cases_central': cases_for_target(spec, mode, looks, spending, dr['central'], incident_rate, lim, kinds=kind_rates(spec, exploration['pilot'], candidate_id, dr['central']), q=q),
                     'cases_pessimistic': cases_for_target(spec, mode, looks, spending, dr['pessimistic'], incident_rate, lim, kinds=kind_rates(spec, exploration['pilot'], candidate_id, dr['pessimistic']), q=q),
                     'exposure_tasks_per_month': spec['monthly_uses'] * (lim - L),
                     'exposure_usd_per_month': None if kappa is None else spec['monthly_uses'] * (lim - L) * kappa})
    mine, looser = rows[1], rows[2]
    rows[2]['incident_limit_binds'] = bool(mine['cases_central'] and looser['cases_central'] and looser['cases_central'] >= mine['cases_central'])
    return rows


# ----------------------------------------------------------------------------- plan, freeze, look

def make_plan(spec, pilot_rows, finalists=1, looks=3, cap_total=None, spending=None, mode='clause', incident_rate=.01, reps=400, seed=20260930, paired=False):
    """paired=True (docs/140): every confirmation case also runs on spec['baseline'], and the plan carries the paired clause
    "no worse than current by more than N a month" (spec.requirements.paired_limit_per_month)."""
    validate_spec_v2(spec)
    need(type(paired) is bool, 'invalid_paired', 'paired', 'Paired is true or false.')
    need(not paired or spec['requirements'].get('paired_limit_per_month') is not None, 'paired_limit_required', 'spec.requirements.paired_limit_per_month',
         'A paired test needs a paired limit: how many more failures a month than current you accept (zero or more).')
    if is_v3(spec):
        validate_rows(v1_view(spec), rows_v1(pilot_rows), 'pilot'); validate_mistakes(spec, pilot_rows)
        ex = explore(v1_view(spec, pooled_kappa(spec, pilot_rows)), rows_v1(pilot_rows))
        ex['spec'] = v1_view(spec); ex['pilot'] = pilot_rows
    else:
        ex = explore(v1_view(spec), pilot_rows)
    need(type(finalists) is int and 1 <= finalists <= len(spec['candidates']), 'invalid_finalists', 'finalists', 'Select one to the full candidate count.')
    need(type(looks) is int and 1 <= looks <= 5, 'invalid_looks', 'looks', 'Plan one to five looks.')
    spending = spending or [1 / looks] * looks
    need(len(spending) == looks and all(number(x) and x > 0 for x in spending) and abs(sequential_sum(spending) - 1) < 1e-9, 'invalid_spending', 'spending', 'Spending shares must be positive and sum to one, one per look.')
    need(mode in ('clause', 'equal'), 'invalid_allocation', 'allocation', 'Choose clause or equal allocation.')
    need(number(incident_rate) and 0 <= incident_rate < 1, 'invalid_limit', 'incident_rate', 'Planning incident rate must be a rate below one.')
    need(all(r['score_usd'] is not None for r in ex['ranking']) or finalists == len(spec['candidates']), 'missing_economics', 'spec.consequence_usd', 'Screening needs complete pilot costs, outcomes and a declared failure consequence. Compare all candidates instead, or repair these inputs.')
    spent = len(pilot_rows)
    if cap_total is None: cap_total = max(0, (int(spec['budget_runs']) - spent) // finalists)
    need(type(cap_total) is int and cap_total >= 0, 'invalid_budget', 'cap', 'Cap must be a nonnegative integer number of cases per finalist.')
    alloc = allocation(spec, cap_total, looks, mode)
    selected = [r['candidate_id'] for r in ex['ranking'][:finalists]]
    options = len(set(selected) | {spec['baseline']}) if paired else finalists     # option runs per confirmation case
    a_final = ALPHA / finalists * spending[-1]          # docs/130 §5.1: no split across populations
    ge = grader_error_of(spec)
    # per-population clauses only; the average is a weighted clause with its own floor (docs/130 §8.2)
    limits = [spec['requirements']['incident_limit'], *[observed_limit(v['failure_limit'], ge) for v in spec['strata'].values() if v['failure_limit'] is not None]]
    floors = [zero_floor(x, a_final) for x in limits]
    minimum = None if None in floors else max(floors)
    average_floor = weighted_floor(spec, observed_limit(spec['requirements']['failure_limit'], ge), looks, spending, finalists, mode)
    tolerance_floors = {m: tolerance_floor(spec, m, looks, spending, finalists, mode) for m in tolerated_kinds(spec)}
    tolerances_ok = all(f is not None and f <= alloc['total'] for f in tolerance_floors.values())
    average_ok = average_floor is not None and average_floor <= alloc['total']
    s_floor = saving_floor(spec, looks, spending, finalists, mode) if paired else None
    # with a zero paired limit there is no zero-discordance floor (docs/140 §5.2), so it does not enter feasibility
    saving_ok = not paired or not paired_limit(spec) > 0 or (s_floor is not None and s_floor <= alloc['total'])
    manifest = []
    for s in sorted(spec['strata']):
        prev = 0
        for j, cum in enumerate(alloc['schedule'][s]):
            for i in range(prev, cum): manifest.append({'case_id': f'confirmation-{s}-{i}', 'stratum': s, 'look': j + 1})
            prev = cum
    audit = None
    if is_v3(spec):
        # what an audit of AI labels would need (docs/130 §3.3, §8.1); shown when the evidence is AI-labelled
        # sized for the TIGHTEST limit the audit must support (overall, per-population, tolerances), not only the overall one
        raw = {k: v for k, v in raw_limits(spec).items() if v is not None}
        tightest = min(sorted(raw), key=lambda k: raw[k])
        frn = fine_rows_needed(raw[tightest]); flagged = min(sequential_sum(1 for r in pilot_rows if _ai_flagged(r)), 100)
        # docs/131 §2.6: the person's audit is part of the price of the test when the labels came from a reader
        ai = any(r.get('origin') == 'ai-labeled' for r in pilot_rows)
        cost = 0. if not ai else None if spec['label_cost_usd'] is None or frn is None else (frn + flagged) * spec['label_cost_usd']
        audit = {'fine_rows_needed': frn, 'for_limit': tightest, 'flagged_rows_suggested': flagged, 'cost_usd': cost}
    pair_plan = None
    if paired and spec['baseline'] != selected[0]:
        base = next(r for r in ex['ranking'] if r['candidate_id'] == spec['baseline'])
        pair_plan = {'current_rate': spec['current_failure_rate'] if spec['current_failure_rate'] is not None else base['failure_rate'], 'correlation': PLANNING_PAIR_CORRELATION}
    oc = operating_characteristics(spec, ex, selected[0], alloc, looks, spending, incident_rate, reps, seed, q=finalists, paired=pair_plan,
                                   audit_cost=(audit['cost_usd'] or 0.) if audit else 0.)
    econ_keys = ('label_cost_usd', 'horizon_months', 'current_failure_rate', 'grader_error') + (('taxonomy', 'deployment') + (('grader_audit',) if 'grader_audit' in spec else ()) if is_v3(spec) else ())
    econ = {k: spec[k] for k in econ_keys}
    if paired: econ['paired_limit_per_month'] = spec['requirements']['paired_limit_per_month']
    plan = {'schema': f'aver-test-plan/{schema_suffix(spec)}', 'method_version': method_of(spec), 'exploration': ex, 'selected': selected,
            'economics_inputs': econ,
            'cap_total': cap_total, 'looks': looks, 'spending': spending, 'allocation': alloc, 'planning_incident_rate': incident_rate,
            'pilot_executions': spent, 'confirmation_executions': alloc['total'] * options, 'paired': paired, 'saving_floor': s_floor,
            'minimum_cases_per_population': minimum, 'average_floor': average_floor,
            'feasible': minimum is not None and min(alloc['final'].values()) >= minimum and tolerances_ok and average_ok and saving_ok,
            'confirmation_cases': manifest, 'operating_characteristics': oc,
            'scope': 'staged_looks_independent_binary_cases'}
    if is_v3(spec):
        plan['tolerance_floors'] = tolerance_floors; plan['kappa'] = {c['id']: effective_kappa(spec, pilot_rows, c['id']) for c in spec['candidates']}
        plan['audit'] = audit
    return plan


def freeze(plan):
    v3 = isinstance(plan, dict) and plan.get('schema') == 'aver-test-plan/v3'
    fields(plan, 'schema method_version exploration selected economics_inputs cap_total looks spending allocation planning_incident_rate pilot_executions confirmation_executions paired saving_floor minimum_cases_per_population average_floor feasible confirmation_cases operating_characteristics scope' + (' tolerance_floors kappa audit' if v3 else ''), 'plan')
    need(type(plan['paired']) is bool, 'invalid_plan', 'plan.paired', 'Supply a plan created by Aver.')
    need((plan['schema'] == 'aver-test-plan/v2' and plan['method_version'] == METHOD) or (v3 and plan['method_version'] == METHOD_V3), 'unsupported_schema', 'plan', 'Expected an aver-test-plan/v2 or v3 plan.')
    ex = plan['exploration']; spec = ex.get('spec') if isinstance(ex, dict) else None
    need(isinstance(spec, dict), 'invalid_plan', 'plan', 'Supply a plan created by Aver.')
    spec2 = spec_to_v2(spec, plan)
    again = make_plan(spec2, ex.get('pilot'), len(plan['selected']), plan['looks'], plan['cap_total'], plan['spending'], plan['allocation']['mode'], plan['planning_incident_rate'], plan['operating_characteristics']['reps'], plan['operating_characteristics']['seed'], plan['paired'])
    need(digest(plan_core(again)) == digest(plan_core(plan)), 'plan_changed', 'plan', 'Plan does not reproduce from its specification and pilot.')
    # docs/131 §2.7 (owner decision 2026-10-01): labels a log reader wrote are audited before the plan is frozen
    need(not (v3 and any(isinstance(r, dict) and r.get('origin') == 'ai-labeled' for r in ex['pilot']) and 'grader_audit' not in plan['economics_inputs']),
         'audit_required_before_freeze', 'plan', AUDIT_BEFORE_FREEZE_MESSAGE)
    need(plan['feasible'], 'infeasible_plan', 'plan', 'This budget cannot support these requirements even with zero failures at the final look. Increase the cap, use fewer looks, or test fewer candidates.')
    return {'schema': f'aver-frozen-plan/{"v3" if v3 else "v2"}', 'plan': plan, 'plan_digest': digest(plan_core(plan))}


def plan_core(plan):
    """Everything that binds the decision; the Monte Carlo operating characteristics are informational and excluded."""
    return {k: v for k, v in plan.items() if k != 'operating_characteristics'}


def spec_to_v2(spec_v1, plan):
    """A v2 plan carries the v1 projection in its exploration and the v2-only inputs in economics_inputs."""
    econ = plan.get('economics_inputs')
    need(isinstance(econ, dict), 'invalid_plan', 'plan.economics_inputs', 'Plans carry the label cost, horizon, current rate and grader error assumptions.')
    out = dict(spec_v1); out['schema'] = SPEC_SCHEMA_V3 if 'taxonomy' in econ else SPEC_SCHEMA
    out.update({k: v for k, v in econ.items() if k != 'paired_limit_per_month'})
    if 'paired_limit_per_month' in econ: out['requirements'] = dict(out['requirements'], paired_limit_per_month=econ['paired_limit_per_month'])
    return validate_spec_v2(out)


def look(frozen, rows, previous=None):
    fields(frozen, 'schema plan plan_digest', 'frozen_plan')
    need(frozen['schema'] in ('aver-frozen-plan/v2', 'aver-frozen-plan/v3') and freeze(frozen['plan']) == frozen, 'plan_changed', 'frozen_plan', 'The frozen plan changed.')
    plan = frozen['plan']; ex = plan['exploration']; spec = spec_to_v2(ex['spec'], plan); selected = plan['selected']; K = plan['looks']; alloc = plan['allocation']
    paired = plan['paired']
    runs = selected + ([spec['baseline']] if paired and spec['baseline'] not in selected else [])     # options that run every case
    if is_v3(spec):
        validate_rows(v1_view(spec), rows_v1(rows), 'confirmation', runs); validate_mistakes(spec, rows)
    else:
        rows = validate_rows(v1_view(spec), rows, 'confirmation', runs)
    if paired:
        # docs/140 §5: every case carries the candidate's outcome AND current's (same case id, same origin), round by round
        by_case = {}
        for r in rows: by_case.setdefault(r['case_id'], set()).add(r['candidate_id'])
        need(all(v == set(runs) for v in by_case.values()), 'paired_incomplete', 'evidence',
             "A paired plan needs current's outcome on every case the candidate ran, and the candidate's on every case current ran: "
             + ', '.join(runs) + ' on the same case ids.')
    pilot = ex['pilot']
    need(not ({r['case_id'] for r in rows} & {r['case_id'] for r in pilot}) and not ({r['case_sha256'] for r in rows} & {r['case_sha256'] for r in pilot}), 'pilot_leakage', 'evidence', 'Pilot cases or content cannot be confirmation evidence.')
    need(not rows or rows[0]['origin'] == pilot[0]['origin'], 'mixed_origins', 'evidence', 'Pilot and confirmation origins must agree.')
    manifest = {c['case_id']: c for c in plan['confirmation_cases']}
    need(all(r['case_id'] in manifest and manifest[r['case_id']]['stratum'] == r['stratum'] for r in rows), 'unplanned_case', 'evidence', 'Collect only cases assigned in the frozen manifest; additional looks require a new plan.')
    if previous is not None:
        verified = verify_result(previous)
        need(verified['frozen_plan'] == frozen, 'previous_plan_changed', 'previous', 'The previous receipt belongs to a different plan.')
        need(verified['report']['stopped'] == 'continue', 'plan_stopped', 'previous', 'The previous look stopped this plan; a new plan is needed for more evidence.')
        current = {(r['candidate_id'], r['case_id']): r for r in rows}
        need(all(current.get((r['candidate_id'], r['case_id'])) == r for r in verified['evidence']), 'earlier_evidence_changed', 'evidence', 'Earlier confirmation rows must remain unchanged during resume.')
    q = len(selected)
    per_look = [cumulative_executions(alloc, j) * len(runs) for j in range(K)]
    all_rows = rows
    j_all = next((i + 1 for i, n in enumerate(per_look) if n == len(all_rows)), None)
    complete_all = j_all is not None and all(manifest[r['case_id']]['look'] <= j_all for r in all_rows)
    # A staged rule stops at the FIRST round that decides, whether the rounds arrive one at a time or together
    # (docs/131 §2.5): evaluate round 1 on its rows, then round 2 on the rows of rounds 1-2, and so on.
    stages = [(1 + i, [r for r in all_rows if manifest[r['case_id']]['look'] <= 1 + i]) for i in range(j_all)] if complete_all else [(j_all, all_rows)]
    for j, rows in stages:
        received = len(rows)
        complete_look = complete_all
        report = _evaluate_look(frozen, plan, spec, selected, K, pilot, rows, j, q, per_look, received, complete_look)
        if report['stopped'] != 'continue' or j == stages[-1][0]: break
    unused = len(all_rows) - len(rows)
    if unused:
        report['unused_executions'] = unused
        report['limitations'].append(f'{unused} rows collected after round {j} were not needed for the decision and were not used.')
    return {'schema': f'aver-decision-result/{schema_suffix(spec)}', 'frozen_plan': frozen, 'evidence': rows, 'report': report}


def _paired_counts(rows, c, base, s):
    """Paired counts for candidate c against the baseline in population s (docs/140 §2.1, §5): b and a count an unknown outcome
    against the candidate (b, a for the upper bound), b_lo and a_lo in its favour (for the lower bound)."""
    cur = {r['case_id']: r for r in rows if r['candidate_id'] == base and r['stratum'] == s}
    pairs = [(r['completed'], cur[r['case_id']]['completed']) for r in rows if r['candidate_id'] == c and r['stratum'] == s]
    return {'n': len(pairs),
            'b': sequential_sum(x is not True and y is not False for x, y in pairs), 'a': sequential_sum(x is True and y is False for x, y in pairs),
            'b_lo': sequential_sum(x is False and y is True for x, y in pairs), 'a_lo': sequential_sum(x is not False and y is not True for x, y in pairs),
            'unknown': sequential_sum(x is None or y is None for x, y in pairs)}


def _weighted_cost(spec, rows, c):
    """Traffic-weighted observed running cost per use of option c on these rows, or None when any cost is missing."""
    cost = 0.
    for s in sorted(spec['strata']):
        rs = [r for r in rows if r['candidate_id'] == c and r['stratum'] == s]
        if not rs or any(r['cost_usd'] is None for r in rs): return None
        cost += spec['strata'][s]['weight'] * sequential_sum(r['cost_usd'] for r in rs) / len(rs)
    return cost


def paired_saving(spec, pilot, c, x_c, x_u, p_c, p_u, delta_up, p_c_up):
    """(lower, point) monthly saving of candidate c against current, from the paired rows (docs/140 §6). x: observed running
    cost per use (averages, not bounded); p: observed failure rates; delta_up: the paired clause's upper bound; p_c_up: an
    upper bound on the candidate's true failure rate (used only when the candidate's price per failure is the higher one)."""
    k_c = effective_kappa(spec, pilot, c); k_u = effective_kappa(spec, pilot, spec['baseline'])
    if k_c is None or k_u is None or x_c is None or x_u is None: return None, None
    V = spec['monthly_uses']
    if k_c <= k_u: lower = V * ((x_u - x_c) - k_c * delta_up)
    else: lower = V * ((x_u - x_c) - k_u * delta_up - (k_c - k_u) * p_c_up)
    point = None if p_c is None or p_u is None else V * ((x_u + k_u * p_u) - (x_c + k_c * p_c))
    return lower, point


def _evaluate_look(frozen, plan, spec, selected, K, pilot, rows, j, q, per_look, received, complete_look):
    summaries = []
    base = spec['baseline']; paired = plan['paired']
    for c in selected:
        if not complete_look:
            summaries.append({'candidate_id': c, 'state': 'refuse', 'clauses': [], 'execution_usd_per_use': None, 'modeled_usd_per_month': None,
                              'modeled_usd_per_month_upper': None, 'binding': None})
            continue
        counts = {}
        cost, cost_known = 0., True
        for s in sorted(spec['strata']):
            rs = [r for r in rows if r['candidate_id'] == c and r['stratum'] == s]; n = len(rs)
            counts[s] = {'n': n, 'failed': sequential_sum(r['completed'] is False for r in rs), 'incident': sequential_sum(r['incident'] is True for r in rs),
                         'unknown_failed': sequential_sum(r['completed'] is None for r in rs), 'unknown_incident': sequential_sum(r['incident'] is None for r in rs)}
            if is_v3(spec): counts[s]['by_mistake'] = {m: sequential_sum(r.get('mistake') == m for r in rs) for m in sorted(spec['taxonomy']['kinds'])}
            if paired and c != base: counts[s]['paired'] = _paired_counts(rows, c, base, s)
            cost_known = cost_known and n > 0 and all(r['cost_usd'] is not None for r in rs)
            cost += spec['strata'][s]['weight'] * sequential_sum(r['cost_usd'] or 0 for r in rs) / max(1, n)
        d = decide(counts, spec, q, K, j, plan['spending'])
        kappa = effective_kappa(spec, pilot, c)
        score = None if not cost_known or d['estimate'] is None or kappa is None else cost + kappa * d['estimate']
        # docs/148: the same cost at the clause's upper bound on the true failure rate; the hedged saving reads it
        score_up = None if score is None else cost + kappa * true_rate_upper(spec, d['avg_upper'])
        summary = {'candidate_id': c, 'state': d['state'], 'clauses': d['clauses'], 'execution_usd_per_use': cost if cost_known else None,
                   'modeled_usd_per_month': None if score is None else score * spec['monthly_uses'],
                   'modeled_usd_per_month_upper': None if score_up is None else score_up * spec['monthly_uses'], 'binding': d['binding'],
                   'avg_upper': d['avg_upper'], 'avg_lower': d['avg_lower']}
        if is_v3(spec): summary['by_mistake'] = d['by_mistake']; summary['kappa_usd'] = kappa
        if 'paired' in d:
            pr = dict(d['paired'])
            x_u = _weighted_cost(spec, rows, base)
            p_u = None if pr['unknown'] else sequential_sum(spec['strata'][s]['weight'] * sequential_sum(r['completed'] is False for r in rows if r['candidate_id'] == base and r['stratum'] == s)
                                                            / counts[s]['n'] for s in sorted(spec['strata']))
            lower, point = paired_saving(spec, pilot, c, cost if cost_known else None, x_u, d['estimate'], p_u, pr['delta_upper'], true_rate_upper(spec, d['avg_upper']))
            pr.update({'candidate_usd_per_use': cost if cost_known else None, 'current_usd_per_use': x_u, 'current_failure_rate': p_u,
                       'saving_usd_per_month_lower': lower, 'saving_usd_per_month': point})
            summary['paired'] = pr
        summaries.append(summary)
    states = [c['state'] for c in summaries]
    supported = [c for c in summaries if c['state'] == 'pass']
    ranked = sorted([c for c in supported if c['modeled_usd_per_month'] is not None], key=lambda c: (c['modeled_usd_per_month'], c['candidate_id']))
    choice = ranked[0]['candidate_id'] if ranked else None
    ai_labeled = any(r['origin'] == 'ai-labeled' for r in rows) or any(r['origin'] == 'ai-labeled' for r in pilot)
    audit = audit_of(spec)
    audit_required = ai_labeled and audit is None
    used = audited_limits(spec) if audit is not None else {}
    impossible = sorted(k for k, v in used.items() if v is None)     # clauses no evidence can pass with this audit
    audit_too_small = bool(impossible)
    if not complete_look: stopped = 'incomplete'
    elif any(s == 'block' for s in states) and not supported: stopped = 'block'
    elif supported and audit_required: stopped = 'continue' if j < K else 'exhausted'
    elif supported: stopped = 'pass'
    elif j < K: stopped = 'continue'
    else: stopped = 'exhausted'
    if audit_required and supported: choice = None; supported = []
    state = 'pass' if choice else 'review' if supported else 'block' if stopped == 'block' else 'refuse'
    reason = ('lowest_observed_cost_supported' if choice else 'missing_economics' if supported else 'requirements_violated' if stopped == 'block'
              else 'incomplete_look' if stopped == 'incomplete' else 'audit_required' if audit_required and stopped in ('continue', 'exhausted') and any(s == 'pass' for s in states)
              else 'audit_too_small' if audit_too_small and stopped in ('continue', 'exhausted')
              else 'unresolved_continue' if stopped == 'continue' else 'requirements_unresolved')
    per_case = None if spec['label_cost_usd'] is None else (summaries[0]['execution_usd_per_use'] or 0) + spec['label_cost_usd']
    next_code = {'pass': 'review_local_result', 'block': 'change_candidate_or_requirements', 'continue': 'continue_to_next_look', 'exhausted': 'keep_current_or_new_plan_or_raise_bar', 'incomplete': 'complete_this_look'}[stopped]
    report = {'method_version': method_of(spec), 'verdict': {'state': state, 'reason_code': reason}, 'choice': choice, 'candidates': summaries,
              'look': j, 'looks': K, 'stopped': stopped, 'next_look_executions': per_look[j] if (j is not None and j < K) else None,
              'evidence_spent': {'cases': received, 'labels': received, 'usd': None if per_case is None else received * per_case},
              'savings': _savings(spec, plan, summaries, choice, None if j is None else 1 - (AUDIT_SHARE if audit_of(spec) is not None else 1.) * ALPHA / q * plan['spending'][j - 1]),
              'release_eligibility': 'not_evaluated',
              'progress': {'received_executions': received, 'planned_executions': per_look, 'complete_look': complete_look},
              'lineage': {'plan_digest': frozen['plan_digest'], 'evidence_digest': digest(rows)},
              'origin': pilot[0]['origin'], 'scope': 'local_assessment_of_declared_population',
              'next_actions': [{'code': next_code, 'actor': 'engineer', 'automatic_retry': False}],
              'limitations': ['Local assessment; not a signed record or release authorization.', 'Sampling and grader validity are declared, not independently verified.',
                              'The 5% selection allowance is split across candidates and planned looks (an audit of AI labels takes 1% of it); lower violation tests have a separate allowance.',
                              'Savings have not been confirmed against current.']}
    if report['savings']['status'] == 'bounded':
        report['limitations'][-1] = ('The saving is a bound from the same cases run on current: the difference in failures is bounded; running costs are '
                                     'the observed averages on those cases, not bounded.')
    elif report['savings']['status'] == 'estimated':
        report['limitations'][-1] = ('The saving is an estimate against current\'s pilot rate and running cost; its lower bound uses the candidate\'s '
                                     'upper bound on failures only. Current was not run on these cases.')
    if is_v3(spec):
        if audit_required: report['audit'] = {'status': 'required', 'applied': False}
        elif audit is not None:
            report['audit'] = {'status': 'applied', 'applied': True, 'counts': dict(audit), 'audited_limit': used}
            if audit_too_small:
                raw = raw_limits(spec)
                needed = fine_rows_needed(min(raw[k] for k in impossible))
                report['audit']['fine_rows_needed'] = needed
                report['audit']['message'] = (f"The audit is too small for a pass: a person checked {audit['sampled_fine']} cases the AI called fine. "
                                              + (f'Check at least {needed} of them, with no missed failures, and re-run the audit.' if needed is not None else 'No audit size can support a pass at a limit of zero.'))
        else: report['audit'] = {'status': 'not_needed', 'applied': False}
        report['taxonomy_version'] = spec['taxonomy']['version']
        report['deployment'] = spec['deployment']
    return report


def current_monthly(spec, exploration):
    """Current's monthly cost, mistakes included, from the pilot: running cost per use plus its price per failure times its pilot
    rate (or the declared current_failure_rate), at the monthly volume. None when a price or cost is unknown."""
    cur = next((r for r in exploration['ranking'] if r['candidate_id'] == spec['baseline']), None)
    kappa_cur = effective_kappa(spec, exploration['pilot'], spec['baseline'])
    if cur is None or kappa_cur is None or cur['execution_cost_usd'] is None: return None
    f_cur = spec['current_failure_rate'] if spec['current_failure_rate'] is not None else cur['failure_rate']
    return spec['monthly_uses'] * (cur['execution_cost_usd'] + kappa_cur * f_cur)


def _savings(spec, plan, summaries, choice, level):
    """report.savings. docs/140 §5: a bound for the chosen candidate (else the first one with a paired comparison) when the plan is
    paired and the costs and prices are known. docs/148: otherwise, for an unpaired pass with known costs, an estimate against
    current's pilot cost with a lower bound from the candidate's upper bound on failures (`level` is that bound's one-sided
    level at this look). Else not evaluated."""
    paired = [c for c in summaries if c.get('paired') and c['paired']['saving_usd_per_month_lower'] is not None]
    pick = next((c for c in paired if c['candidate_id'] == choice), paired[0] if paired else None)
    if pick is not None:
        return {'status': 'bounded', 'candidate_id': pick['candidate_id'], 'lower_usd_per_month': pick['paired']['saving_usd_per_month_lower'],
                'point_usd_per_month': pick['paired']['saving_usd_per_month']}
    cand = next((c for c in summaries if c['candidate_id'] == choice), None) if choice is not None else None
    if cand is None or level is None or cand['candidate_id'] == spec['baseline']: return {'status': 'not_evaluated'}
    if cand.get('modeled_usd_per_month') is None or cand.get('modeled_usd_per_month_upper') is None: return {'status': 'not_evaluated'}
    cur_m = current_monthly(spec, plan['exploration'])
    if cur_m is None: return {'status': 'not_evaluated'}
    return {'status': 'estimated', 'candidate_id': cand['candidate_id'], 'point_usd_per_month': cur_m - cand['modeled_usd_per_month'],
            'lower_usd_per_month': cur_m - cand['modeled_usd_per_month_upper'], 'level': level}


def verify_result(result):
    fields(result, 'schema frozen_plan evidence report', 'result')
    need(result['schema'] in ('aver-decision-result/v2', 'aver-decision-result/v3'), 'unsupported_schema', 'result', 'Expected a v2 or v3 decision result.')
    expected = look(result['frozen_plan'], result['evidence'])
    def same(a, b):
        if number(a) and number(b): return abs(a - b) <= 1e-9 * max(1, abs(a), abs(b))
        if isinstance(a, dict) and isinstance(b, dict): return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
        if isinstance(a, list) and isinstance(b, list): return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        return type(a) is type(b) and a == b
    need(same(expected, result), 'result_changed', 'result', 'Result does not reproduce from its evidence. Recompute it.')
    return expected


# ----------------------------------------------------------------------------- demo

# The example pilot: 60 cases per option, 20 per population, with a few failures in the edge populations so the
# options differ. Lean and current each hold three failures (8.5 per 1,000 weighted); balanced three (6.5); guarded one,
# so its central design rate falls back to half the limit. Identical in the TypeScript twin (decisionWorkflowV2.ts).
# docs/129 §8: the example pilot's failures by option, each with its kind. 60 cases per option, 20 per population.
DEMO_PILOT_FAILURES = {
    'current': {('ambiguous', 3): 'wrong_field', ('ambiguous', 11): 'missed_page', ('tool_failure', 7): 'wrong_field'},
    'lean': {('ambiguous', 2): 'wrong_field', ('ambiguous', 14): 'invented_value', ('tool_failure', 5): 'missed_page'},
    'balanced': {('ambiguous', 8): 'wrong_field', ('tool_failure', 12): 'wrong_field', ('tool_failure', 3): 'missed_page'},
    'guarded': {('ambiguous', 9): 'missed_page'},
}
DEMO_TAXONOMY = {'version': 1, 'kinds': {
    'wrong_field': {'label': 'Wrong field extracted', 'severity': 'mistake', 'price_usd': 2., 'tolerance_per_month': None, 'standard': 'wrong_output'},
    'missed_page': {'label': 'Missed a page', 'severity': 'mistake', 'price_usd': 2., 'tolerance_per_month': None, 'standard': 'incomplete'},
    'invented_value': {'label': 'Invented a value', 'severity': 'mistake', 'price_usd': 3., 'tolerance_per_month': None, 'standard': 'wrong_output'},
    'unwanted_write': {'label': 'Wrote to the ledger when it should not have', 'severity': 'incident', 'price_usd': None, 'tolerance_per_month': 450., 'standard': 'unwanted_action'},
}}
DEMO_CONFIRMATION_KINDS = ('wrong_field', 'missed_page', 'invented_value')


DEMO_PAIRED_LIMIT = 300.     # the paired demo: Lean may fail at most 300 more a month than Current (1% at 30,000 uses)
DEMO_PAIRED_CURRENT_EXTRA = {('ambiguous', 5): 'wrong_field', ('ambiguous', 17): 'wrong_field'}   # Current fails a bit more in the pilot too
DEMO_PAIRED_CURRENT_EVERY = 50   # in the paired demo's confirmation rows Current also fails one case in 50 that the candidate gets right


def demo(scenario='document-processing', outcome='pass', looks=3, paired=False):
    """paired=True (docs/140 §5.6): the plan also runs every case on Current, the spec carries a paired limit of 300 a month,
    and the rows include Current's outcomes on the same cases; Current fails the cases Lean fails plus a few more."""
    from aver_policy.decision import demo as demo_v1
    need(outcome in ('pass', 'continue', 'block', 'exhausted'), 'unknown_outcome', 'outcome', 'Choose pass, continue, block or exhausted.')
    base = demo_v1(scenario, 'pass')
    spec = dict(base['spec']); spec['schema'] = SPEC_SCHEMA_V3
    spec.update({'label_cost_usd': 2., 'horizon_months': 12, 'current_failure_rate': None, 'grader_error': {'false_negative': 0, 'false_positive': 0},
                 'monthly_uses': 30000, 'consequence_usd': 2., 'taxonomy': DEMO_TAXONOMY,
                 'deployment': {'mode': 'agentic', 'description': 'Extracts fields from inbound documents and writes them to a ledger without a person in the loop.'}})
    if paired: spec['requirements'] = dict(spec['requirements'], paired_limit_per_month=DEMO_PAIRED_LIMIT)
    failures = DEMO_PILOT_FAILURES if not paired else dict(DEMO_PILOT_FAILURES, current={**DEMO_PILOT_FAILURES['current'], **DEMO_PAIRED_CURRENT_EXTRA})
    def pilot_row(r):
        kind = failures[r['candidate_id']].get((r['stratum'], int(r['case_id'].rsplit('-', 1)[1])))
        return dict(r, schema='aver-outcomes/v2', completed=kind is None, mistake=kind)
    pilot = [pilot_row(r) for r in base['pilot']]
    plan = make_plan(spec, pilot, 1, looks, 1500, None, 'clause', .01, 200, paired=paired)
    frozen = freeze(plan)
    chosen = next(c for c in spec['candidates'] if c['id'] == plan['selected'][0])
    current = next(c for c in spec['candidates'] if c['id'] == spec['baseline'])
    costs = {'current': .03, 'lean': .008, 'balanced': .015, 'guarded': .025}
    def row(case, bad, i, incident=False, who=chosen):
        return {'schema': 'aver-outcomes/v2', 'case_id': case['case_id'], 'case_sha256': digest({'scenario': scenario, 'case': case['case_id']}), 'candidate_id': who['id'], 'configuration_sha256': who['digest'], 'stratum': case['stratum'], 'phase': 'confirmation', 'origin': 'synthetic', 'completed': not bad, 'incident': incident, 'cost_usd': costs[who['id']], 'mistake': 'unwanted_write' if incident else DEMO_CONFIRMATION_KINDS[i % 3] if bad else None}
    cases = plan['confirmation_cases']; first = [c for c in cases if c['look'] <= 1]
    if outcome == 'block':
        spots = [(c, i % 2 == 0, i, False) for i, c in enumerate(first)]
    elif outcome == 'continue':
        spots = [(c, i % 25 == 0, i, False) for i, c in enumerate(first)]
    elif outcome == 'exhausted':
        spots = [(c, i % 28 == 0, i, False) for i, c in enumerate(cases)]
    else:
        # round 1 carries a few failures and one unwanted write (keep testing); round 2 is clean and the rule passes at round 2
        two = [c for c in cases if c['look'] <= 2]
        spots = [(c, c['look'] == 1 and i % 60 == 0, i, c['look'] == 1 and i == 7) for i, c in enumerate(two)]
    rows = [row(c, bad, i, inc) for c, bad, i, inc in spots]
    if paired and chosen['id'] != current['id']:
        # Current fails every case the candidate fails, and one case in 50 more (docs/140 §5.6)
        rows += [row(c, bad or i % DEMO_PAIRED_CURRENT_EVERY == 13, i, False, current) for c, bad, i, inc in spots]
    return {'spec': spec, 'pilot': pilot, 'plan': plan, 'frozen_plan': frozen, 'confirmation': rows, 'result': look(frozen, rows)}
