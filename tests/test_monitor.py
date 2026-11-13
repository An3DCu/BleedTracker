import math

import numpy as np
import pandas as pd
import pytest

from bleedtracker.leakage import leak_probe, no_outcome_or_snapshot_features
from bleedtracker.monitor import (
    GateResult,
    alarm_row,
    decisions,
    in_a_row,
    pre_release_gate,
    psi,
)
from bleedtracker.release import ReleaseHeld, open_sealed_month


def test_psi_of_the_same_values_is_zero():
    values = np.arange(1, 101)
    assert psi(values, values) == pytest.approx(0.0)


def test_psi_worked_by_hand():
    # Ten values give ten bins with a share of 0.1 each. Every actual value lands in
    # the first bin. The other nine bins are empty and count as 1e-6.
    expected = 0.9 * math.log(1 / 0.1) + 9 * (1e-6 - 0.1) * math.log(1e-6 / 0.1)
    assert psi(np.arange(1, 11), np.ones(10)) == pytest.approx(expected)


def test_psi_needs_at_least_two_bins():
    assert math.isnan(psi(np.ones(20), np.ones(20)))


def frame(n: int, seed: int, position_shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "log_impressions": rng.normal(7, 1, n),
            "active_days": rng.integers(20, 31, n),
            "position": rng.normal(15 + position_shift, 5, n),
            "ctr": rng.uniform(0, 0.05, n),
            "momentum": rng.uniform(0, 2, n),
        }
    )


def test_gate_refuses_a_score_frame_with_a_label():
    score = frame(100, 1).assign(label_declining=0)
    with pytest.raises(ValueError):
        pre_release_gate(frame(100, 0), score, frame(100, 2), 30, 31)


def test_gate_releases_a_steady_month():
    gate = pre_release_gate(frame(1000, 0), frame(1000, 1), frame(1000, 2), 30, 31)
    assert gate.pages_vs_previous_month == 1.0
    assert gate.decision == "RELEASE"


def test_a_population_move_of_25_percent_holds_the_release():
    gate = pre_release_gate(frame(1000, 0), frame(1250, 1), frame(1000, 2), 30, 31)
    assert gate.A1_population_moved
    assert gate.decision == "HOLD"


def test_a_drifted_feature_holds_the_release():
    gate = pre_release_gate(
        frame(1000, 0), frame(1000, 1, position_shift=10), frame(1000, 2), 30, 31
    )
    assert gate.psi_against_training["position"] >= 0.25
    assert gate.A2_features_drifted
    assert gate.decision == "HOLD"


def test_worst_psi_reads_active_days_per_day():
    gate = pre_release_gate(frame(1000, 0), frame(1000, 1), frame(1000, 2), 28, 31)
    others = [v for f, v in gate.psi_against_training.items() if f != "active_days"]
    assert gate.worst_psi == max([*others, gate.active_days_per_day_psi])


def held_gate() -> GateResult:
    return GateResult(0, 0, 1.0, 0, {}, {}, {}, 0.0, 0.3, False, True)


def test_a_held_release_keeps_the_sealed_label_closed():
    with pytest.raises(ReleaseHeld):
        open_sealed_month(None, None, frame(10, 0), held_gate())


def test_alarm_row_worked_by_hand():
    deployment = {"rule_precision_at_50": 0.98, "base_rate": 0.7362, "trial_40_10_precision": 0.92}
    row = alarm_row(
        "2026-05 to 2026-06", held_gate(), 0.388, 0.7362, deployment, {"shadow_lift_points": -3.3}
    )
    assert row["A2_features_drifted"] is True
    assert row["A3_base_rate_moved"] is True
    assert row["base_rate_move_points"] == 34.8
    # 0.98 - 0.7362 = 0.2438, above the 0.10 margin.
    assert row["A4_queue_below_margin"] is False
    assert row["queue_margin_points"] == 24.4
    assert row["A5_shadow_lift_not_positive"] is True
    assert row["A6_trial_costs_precision"] is True


def test_in_a_row():
    assert in_a_row([True, False, True, False], 2) is False
    assert in_a_row([False, True, True], 2) is True


def test_decisions_need_two_months_in_a_row():
    alarms = [
        {"A5_shadow_lift_not_positive": True, "A6_trial_costs_precision": True},
        {"A5_shadow_lift_not_positive": False, "A6_trial_costs_precision": True},
        {"A5_shadow_lift_not_positive": False, "A6_trial_costs_precision": False},
        {"A5_shadow_lift_not_positive": True, "A6_trial_costs_precision": True},
    ]
    assert decisions(alarms) == {"close_shadow_queue": False, "promote_trial": False}


def test_no_outcome_or_snapshot_column_is_a_feature():
    assert no_outcome_or_snapshot_features()


def test_the_planted_label_leak_is_caught():
    # The forest needs 400 pages per leaf. Training gets 1,800 pages.
    rng = np.random.default_rng(0)
    n = 3000
    data = frame(n, 3).assign(label_declining=rng.integers(0, 2, n))
    data["imp_next"] = rng.uniform(0, 1000, n)
    train, test = data.iloc[:1800], data.iloc[1800:]
    scores = rng.uniform(0, 1, len(test))
    result = leak_probe(train, test, scores)
    assert result["with_the_label"]["auc"] == 1.0
    assert result["five_features"]["auc"] < 0.6
    assert set(result) == {"five_features", "with_next_month_impressions", "with_the_label"}
    assert all(set(v) == {"precision_at_50", "auc"} for v in result.values())
