"""Dependency-free CASS committee primitives."""

from __future__ import annotations

import math


_Z_SCORES = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}


def z_score(confidence: float) -> float:
    """Return the two-sided normal z-score for a confidence level."""

    if confidence in _Z_SCORES:
        return _Z_SCORES[confidence]
    p = 1.0 - (1.0 - confidence) / 2.0
    return _inverse_standard_normal_cdf(p)


def _inverse_standard_normal_cdf(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


def wilson_interval(passes: int, n: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """Return (center, low, high) for a binomial pass rate."""

    if n <= 0:
        return 0.0, 0.0, 1.0
    if passes < 0 or passes > n:
        raise ValueError("passes must be between 0 and n")
    p_hat = passes / n
    z = z_score(confidence)
    denom = 1.0 + z * z / n
    midpoint = (p_hat + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n))
    return p_hat, max(0.0, midpoint - half), min(1.0, midpoint + half)


def aggregate_blocks(blocks: list[bool], rule: str) -> bool:
    """Return True when a committee blocks under the requested rule."""

    if not blocks:
        return True
    if rule == "serial":
        rule = "serial_veto"
    block_count = sum(1 for flag in blocks if flag)
    total = len(blocks)
    if rule == "serial_veto":
        return block_count >= 1
    if rule == "majority":
        return block_count >= math.ceil(total / 2)
    if rule == "supermajority":
        return block_count >= math.ceil(2 * total / 3)
    if rule == "unanimous_block":
        return block_count == total
    if rule.startswith("quota_"):
        return block_count >= int(rule.removeprefix("quota_"))
    raise ValueError(f"Unknown aggregation rule: {rule}")


def effective_block_quota(rule: str, size: int) -> int:
    """Return the block-count threshold a rule reduces to for a committee size.

    Every supported aggregation rule is a quota rule ("block when at least Q
    members block"), so two rules are behaviorally identical on a committee of
    ``size`` members exactly when they share this quota. A quota above ``size``
    is normalized to ``size + 1`` (the committee can never block), so all
    unreachable quotas dedupe together.
    """

    if size < 1:
        raise ValueError("committee size must be at least 1")
    if rule == "serial":
        rule = "serial_veto"
    if rule == "serial_veto":
        return 1
    if rule == "majority":
        return math.ceil(size / 2)
    if rule == "supermajority":
        return math.ceil(2 * size / 3)
    if rule == "unanimous_block":
        return size
    if rule.startswith("quota_"):
        quota = int(rule.removeprefix("quota_"))
        if quota < 1:
            raise ValueError(f"quota rule must be at least 1: {rule}")
        return min(quota, size + 1)
    raise ValueError(f"Unknown aggregation rule: {rule}")


def _standard_normal_sf(z: float) -> float:
    """Survival function of the standard normal, P(Z >= z)."""

    return 0.5 * math.erfc(z / math.sqrt(2.0))


def sign_test_p_one_sided(better: int, worse: int) -> float:
    """Exact one-sided sign-test p-value on discordant pairs.

    Under the null of no difference, each discordant pair favors either side
    with probability 1/2. Returns P(X >= better) for X ~ Binomial(better +
    worse, 1/2) — the probability of seeing evidence at least this lopsided by
    chance. Uses the exact tail for n <= 500 and a continuity-corrected normal
    tail above that (the approximation error there is far below any decision
    threshold in use).
    """

    if better < 0 or worse < 0:
        raise ValueError("discordant counts must be non-negative")
    n = better + worse
    if n == 0:
        return 1.0
    if n <= 500:
        tail = sum(math.comb(n, i) for i in range(better, n + 1))
        return tail / (2**n)
    z = (better - 0.5 - n / 2.0) / math.sqrt(n / 4.0)
    return _standard_normal_sf(z)


def required_sample_size(
    threshold: float,
    *,
    confidence: float = 0.95,
    tolerated_mismatches: int = 0,
    max_n: int = 1_000_000,
) -> int | None:
    """Smallest n whose verdict can clear ``threshold`` on both the point rate
    and the Wilson lower bound, given at most ``tolerated_mismatches`` misses.

    Returns None when no n up to ``max_n`` suffices (threshold too close to 1
    for the tolerated miss count). With zero misses this reduces to the closed
    form n >= z^2 * t / (1 - t), used to seed the search.
    """

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1")
    if tolerated_mismatches < 0:
        raise ValueError("tolerated_mismatches must be non-negative")
    z = z_score(confidence)
    floor_n = max(tolerated_mismatches + 1, int(z * z * threshold / (1.0 - threshold)))
    for n in range(floor_n, max_n + 1):
        matched = n - tolerated_mismatches
        if matched <= 0 or matched / n < threshold:
            continue
        _, low, _ = wilson_interval(matched, n, confidence)
        if low >= threshold:
            return n
    return None
