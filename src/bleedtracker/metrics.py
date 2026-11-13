"""The metrics every result is measured with."""

import math

import numpy as np

TOP_K = 50
WILSON_Z = 1.645  # 90% two-sided


def precision_at_k(scores, labels, k: int = TOP_K) -> float:
    """Share of declining pages among the k highest scores.

    Pages tied at the cut share the remaining slots. They count with the share of
    declines among them. A random pick among the tied pages gives the same on
    average. The order of the rows never decides the result.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    if len(s) < k:
        raise ValueError(f"Precision@{k} needs at least {k} pages, got {len(s)}.")
    cut = np.sort(s)[::-1][k - 1]
    above = s > cut
    tied = s == cut
    slots = k - int(above.sum())
    return float((y[above].sum() + slots * y[tied].mean()) / k)


def margin_points(precision: float, base_rate: float) -> float:
    """How far a queue's precision sits above the base rate of its pool, in points."""
    return 100 * (precision - base_rate)


def wilson_interval(hits: float, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    """Wilson score interval for a share of hits out of n (90% by default)."""
    p = hits / n
    denominator = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / denominator, (centre + half) / denominator
