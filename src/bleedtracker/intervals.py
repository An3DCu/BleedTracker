"""Intervals on the results.

Both bootstraps draw from one generator seeded with 42. The second one carries on
where the first stopped. My original analysis ran them that way.
The order matters for the numbers to match.
"""

import numpy as np

from bleedtracker.metrics import TOP_K, wilson_interval
from bleedtracker.model import SEED

BOOTSTRAP_DRAWS = 10_000


def bootstrap_means(values, rng, draws: int = BOOTSTRAP_DRAWS) -> np.ndarray:
    """Means of `draws` resamples of `values`, with replacement."""
    values = np.asarray(values, dtype=float)
    return np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(draws)])


def sealed_wilson(precision: float) -> list[float]:
    """90% Wilson interval for a queue of 50, from its precision."""
    return [round(v, 4) for v in wilson_interval(int(round(precision * TOP_K)), TOP_K)]


def uncertainty(rows: list[dict], shadow_rows: list[dict]) -> dict:
    """Intervals on the forest minus the rule and on the shadow queue's lift.

    `rows` and `shadow_rows` are the ladder's rows. The anchor is left out. Only the
    four forward deployments count.
    """
    forward = rows[1:]
    diff = np.array([r["forest_precision_at_50"] - r["rule_precision_at_50"] for r in forward])
    rng = np.random.default_rng(SEED)
    boot = bootstrap_means(diff, rng)
    lift = np.array([r["shadow_lift_points"] for r in shadow_rows[1:]])
    boot_lift = bootstrap_means(lift, rng)
    sealed = rows[-1]
    return {
        "forest_minus_rule_points": [round(float(d) * 100, 1) for d in diff],
        "forest_minus_rule_mean_points": round(float(diff.mean()) * 100, 2),
        "forest_minus_rule_interval_90": [
            round(float(np.percentile(boot, 5)) * 100, 2),
            round(float(np.percentile(boot, 95)) * 100, 2),
        ],
        "sealed_rule_wilson_90": sealed_wilson(sealed["rule_precision_at_50"]),
        "sealed_forest_wilson_90": sealed_wilson(sealed["forest_precision_at_50"]),
        "shadow_lift_points": [round(float(v), 1) for v in lift],
        "shadow_lift_mean_points": round(float(lift.mean()), 1),
        "shadow_lift_interval_90": [
            round(float(np.percentile(boot_lift, 5)), 1),
            round(float(np.percentile(boot_lift, 95)), 1),
        ],
    }
