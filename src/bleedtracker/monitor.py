"""The checks around each release.

The pre-release gate runs before anyone sees a queue. It has no labels. It can only
look at the pages and the feature distributions. The alarms run once the
labels arrive. They can only change the next queue.

Every threshold here was written down before the sealed month was opened.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bleedtracker.config import FEATURES

PRE_REGISTERED = {
    "population": 0.20,
    "psi": 0.25,
    "base_rate": 0.10,
    "queue_margin": 0.10,
    "shadow_close_months": 2,
    "promotion_months": 2,
}
PSI_WATCH = 0.10


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index of `actual` against the deciles of `expected`.

    A value of 0.10 is worth watching. A value of 0.25 is worth a closer look. The cut
    points are a common convention. They aren't derived from this data.
    """
    e_arr = np.asarray(expected, dtype=float)
    a_arr = np.asarray(actual, dtype=float)
    cuts = np.unique(np.quantile(e_arr, np.linspace(0, 1, bins + 1)))
    if len(cuts) < 3:
        return float("nan")
    cuts[0], cuts[-1] = -np.inf, np.inf
    e = np.histogram(e_arr, cuts)[0] / len(e_arr)
    a = np.histogram(a_arr, cuts)[0] / len(a_arr)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift(train: pd.DataFrame, score: pd.DataFrame, train_days: int, score_days: int):
    """Each feature's PSI against the training month. The worst one comes back too.

    `active_days` counts the month's own days. Its raw PSI jumps between months of
    different length. The worst value uses it per day instead.
    """
    per_feature = {f: psi(train[f], score[f]) for f in FEATURES}
    active_per_day = psi(train.active_days / train_days, score.active_days / score_days)
    worst = max([v for f, v in per_feature.items() if f != "active_days"] + [active_per_day])
    return per_feature, active_per_day, worst


@dataclass
class GateResult:
    """What the pre-release gate saw and what it decided."""

    pages_scored: int
    pages_previous_month: int
    pages_vs_previous_month: float
    nulls_in_features: int
    feature_types: dict
    psi_against_training: dict
    psi_against_previous_month: dict
    active_days_per_day_psi: float
    worst_psi: float
    A1_population_moved: bool
    A2_features_drifted: bool

    @property
    def decision(self) -> str:
        return "HOLD" if self.A1_population_moved or self.A2_features_drifted else "RELEASE"

    def summary(self) -> dict:
        return {
            "pages_scored": self.pages_scored,
            "pages_previous_month": self.pages_previous_month,
            "pages_vs_previous_month": round(self.pages_vs_previous_month, 4),
            "nulls_in_features": self.nulls_in_features,
            "psi_against_training": {k: round(v, 4) for k, v in self.psi_against_training.items()},
            "psi_against_previous_month": {
                k: round(v, 4) for k, v in self.psi_against_previous_month.items()
            },
            "active_days_per_day_psi": round(self.active_days_per_day_psi, 4),
            "worst_psi": round(self.worst_psi, 4),
            "A1_population_moved": self.A1_population_moved,
            "A2_features_drifted": self.A2_features_drifted,
            "decision": self.decision,
        }


def pre_release_gate(
    train: pd.DataFrame,
    score: pd.DataFrame,
    previous: pd.DataFrame,
    train_days: int,
    score_days: int,
) -> GateResult:
    """Check a release before it ships.

    `train` is the training month. `score` is the month about to be scored.
    `previous` is the same clients one month earlier. None of them may carry a label
    for the month being scored.
    """
    if "label_declining" in score.columns:
        raise ValueError("The pre-release gate must not see the outcome it releases.")
    ratio = len(score) / len(previous)
    per_feature, active_per_day, worst = drift(train, score, train_days, score_days)
    return GateResult(
        pages_scored=len(score),
        pages_previous_month=len(previous),
        pages_vs_previous_month=ratio,
        nulls_in_features=int(score[FEATURES].isna().sum().sum()),
        feature_types={f: str(score[f].dtype) for f in FEATURES},
        psi_against_training=per_feature,
        psi_against_previous_month={f: psi(previous[f], score[f]) for f in FEATURES},
        active_days_per_day_psi=active_per_day,
        worst_psi=worst,
        A1_population_moved=bool(abs(ratio - 1) >= PRE_REGISTERED["population"]),
        A2_features_drifted=bool(worst >= PRE_REGISTERED["psi"]),
    )


def alarm_row(
    release: str,
    gate: GateResult,
    base_rate_before: float,
    base_rate_now: float,
    deployment: dict,
    shadow: dict,
) -> dict:
    """All six alarms for one release, once its labels are known.

    The queue margin and the trial use the ladder's stored values, at 4 decimals.
    """
    move = base_rate_now - base_rate_before
    margin = deployment["rule_precision_at_50"] - deployment["base_rate"]
    return {
        "release": release,
        "A1_population_moved": gate.A1_population_moved,
        "A2_features_drifted": gate.A2_features_drifted,
        "A3_base_rate_moved": bool(abs(move) >= PRE_REGISTERED["base_rate"]),
        "A4_queue_below_margin": bool(margin < PRE_REGISTERED["queue_margin"]),
        "A5_shadow_lift_not_positive": bool(shadow["shadow_lift_points"] / 100 <= 0),
        "A6_trial_costs_precision": bool(
            deployment["trial_40_10_precision"] < deployment["rule_precision_at_50"]
        ),
        "base_rate_move_points": round(move * 100, 1),
        "queue_margin_points": round(margin * 100, 1),
    }


def in_a_row(flags, months: int) -> bool:
    """True when the flag is set in `months` releases in a row."""
    run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        if run >= months:
            return True
    return False


def decisions(alarms: list[dict]) -> dict:
    """The two pre-registered decision rules, read over the releases in order."""
    return {
        "close_shadow_queue": in_a_row(
            [a["A5_shadow_lift_not_positive"] for a in alarms],
            PRE_REGISTERED["shadow_close_months"],
        ),
        "promote_trial": in_a_row(
            [not a["A6_trial_costs_precision"] for a in alarms],
            PRE_REGISTERED["promotion_months"],
        ),
    }
