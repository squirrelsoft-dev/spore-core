"""Native statistics for the EvalHarness — NO specialized stats lib (Rules
26-28). Mirrors ``rust/crates/spore-eval/src/stats.rs`` byte-for-byte.

Ships:
  * :class:`MetricStats` aggregation (mean, stddev, p50, p95, n).
  * :func:`welch_t_test` — unequal-variance two-sample t with
    Welch-Satterthwaite df and a two-sided p-value via the regularized
    incomplete beta function (Lentz continued fraction). Rule 26.
  * :func:`bootstrap_ci` — percentile confidence interval, default 1000
    iters, using an inline :class:`SplitMix64` PRNG so cross-language replay
    is byte-identical. Rule 27.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# SplitMix64 / wrapping arithmetic operate on unsigned 64-bit values.
_U64_MASK = 0xFFFF_FFFF_FFFF_FFFF


# ============================================================================
# MetricStats
# ============================================================================


@dataclass
class MetricStats:
    """Aggregated sample statistics for one metric over a set of runs
    (Rule 19)."""

    mean: float
    stddev: float
    p50: float
    p95: float
    n: int

    @classmethod
    def from_samples(cls, samples: list[float]) -> MetricStats:
        """Aggregate samples. ``stddev`` is the sample standard deviation
        (Bessel's correction, n-1); with fewer than two samples it is ``0.0``.
        Percentiles use the nearest-rank method on the sorted samples."""
        n = len(samples)
        if n == 0:
            return cls(mean=0.0, stddev=0.0, p50=0.0, p95=0.0, n=0)
        mean = sum(samples) / n
        if n < 2:
            stddev = 0.0
        else:
            var = sum((x - mean) ** 2 for x in samples) / (n - 1)
            stddev = math.sqrt(var)
        ordered = sorted(samples)
        return cls(
            mean=mean,
            stddev=stddev,
            p50=percentile(ordered, 50.0),
            p95=percentile(ordered, 95.0),
            n=n,
        )


def percentile(ordered: list[float], p: float) -> float:
    """Nearest-rank percentile on already-sorted data. ``p`` is in ``[0, 100]``."""
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = math.ceil((p / 100.0) * len(ordered))
    idx = min(max(rank, 1), len(ordered)) - 1
    return ordered[idx]


# ============================================================================
# Welch's t-test (Rule 26)
# ============================================================================


@dataclass
class WelchResult:
    """Result of a two-sample Welch t-test."""

    t: float
    df: float
    p_value: float


def welch_t_test(a: list[float], b: list[float]) -> WelchResult:
    """Welch's unequal-variance two-sample t-test with Welch-Satterthwaite df
    and a two-sided p-value (Rule 26).

    Degenerate inputs (either sample with n < 2, or both variances zero) return
    ``t = 0``, ``df = 0``, ``p_value = 1.0`` — "no detectable difference" rather
    than a divide-by-zero."""
    na = len(a)
    nb = len(b)
    if na < 2 or nb < 2:
        return WelchResult(t=0.0, df=0.0, p_value=1.0)
    mean_a = sum(a) / na
    mean_b = sum(b) / nb
    var_a = sum((x - mean_a) ** 2 for x in a) / (na - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (nb - 1)

    sa = var_a / na
    sb = var_b / nb
    denom = sa + sb
    if denom <= 0.0:
        # Both samples are constant.
        p = 1.0 if abs(mean_a - mean_b) <= 2.220446049250313e-16 else 0.0
        return WelchResult(t=0.0, df=0.0, p_value=p)

    t = (mean_a - mean_b) / math.sqrt(denom)
    df = denom**2 / (sa**2 / (na - 1) + sb**2 / (nb - 1))
    p_value = _two_sided_p(t, df)
    return WelchResult(t=t, df=df, p_value=p_value)


def _two_sided_p(t: float, df: float) -> float:
    """Two-sided p-value for a t-statistic with ``df`` degrees of freedom:
    ``p = I_{df/(df+t^2)}(df/2, 1/2)``."""
    if df <= 0.0:
        return 1.0
    x = df / (df + t * t)
    return min(max(_betai(df / 2.0, 0.5, x), 0.0), 1.0)


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function ``I_x(a, b)`` via the Lentz
    continued fraction (Numerical Recipes' ``betai``)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = _ln_gamma(a + b) - _ln_gamma(a) - _ln_gamma(b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) + ln_beta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _betacf(a: float, b: float, x: float) -> float:
    """Lentz continued fraction for the incomplete beta."""
    max_iter = 200
    eps = 3.0e-12
    fpmin = 1.0e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d

    for m_int in range(1, max_iter + 1):
        m = float(m_int)
        m2 = 2.0 * m
        # Even step.
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        # Odd step.
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


_LANCZOS_G = 7.0
_LANCZOS_COEF = (
    0.9999999999998099,
    676.5203681218851,
    -1259.1392167224028,
    771.3232867877531,
    -176.6150291621406,
    12.507343278686905,
    -0.13857109526572012,
    9.984369578019572e-6,
    1.5056327351493116e-7,
)


def _ln_gamma(x: float) -> float:
    """Lanczos approximation of ``ln(gamma(x))`` for x > 0."""
    if x < 0.5:
        # Reflection formula.
        return math.log(math.pi) - math.log(math.sin(math.pi * x)) - _ln_gamma(1.0 - x)
    x -= 1.0
    a = _LANCZOS_COEF[0]
    t = x + _LANCZOS_G + 0.5
    for i in range(1, len(_LANCZOS_COEF)):
        a += _LANCZOS_COEF[i] / (x + i)
    return 0.5 * math.log(2.0 * math.pi) + (x + 0.5) * math.log(t) - t + math.log(a)


# ============================================================================
# Bootstrap CI (Rule 27)
# ============================================================================


@dataclass
class ConfidenceInterval:
    """A bootstrap confidence interval."""

    lower: float
    upper: float
    level: float


class SplitMix64:
    """Inline SplitMix64 PRNG (Rule 27): a fixed, seedable, byte-identical
    generator so a bootstrap CI replays identically across all four
    languages."""

    __slots__ = ("state",)

    def __init__(self, seed: int) -> None:
        self.state = seed & _U64_MASK

    def next_u64(self) -> int:
        """Next 64-bit value (the canonical SplitMix64 step)."""
        self.state = (self.state + 0x9E37_79B9_7F4A_7C15) & _U64_MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58_476D_1CE4_E5B9) & _U64_MASK
        z = ((z ^ (z >> 27)) * 0x94D0_49BB_1331_11EB) & _U64_MASK
        return z ^ (z >> 31)

    def next_index(self, n: int) -> int:
        """Uniform index in ``[0, n)``. ``n`` must be non-zero."""
        return self.next_u64() % n


def bootstrap_ci(
    samples: list[float],
    iterations: int,
    level: float,
    seed: int,
) -> ConfidenceInterval | None:
    """Percentile bootstrap CI for the mean (Rule 27). Resamples ``samples``
    with replacement ``iterations`` times using a :class:`SplitMix64` seeded
    with ``seed``, then takes the ``[(1-level)/2, 1-(1-level)/2]`` percentiles
    of the resample means. Returns ``None`` for an empty sample."""
    if not samples:
        return None
    rng = SplitMix64(seed)
    n = len(samples)
    means: list[float] = []
    for _ in range(iterations):
        total = 0.0
        for _ in range(n):
            total += samples[rng.next_index(n)]
        means.append(total / n)
    means.sort()
    alpha = 1.0 - level
    lower = percentile(means, alpha / 2.0 * 100.0)
    upper = percentile(means, (1.0 - alpha / 2.0) * 100.0)
    return ConfidenceInterval(lower=lower, upper=upper, level=level)


DEFAULT_BOOTSTRAP_ITERATIONS = 1000
"""Default bootstrap iteration count (Rule 27)."""

DEFAULT_BOOTSTRAP_SEED = 0x5EED_5EED_5EED_5EED
"""Fixed bootstrap seed so replays are byte-identical cross-language."""
