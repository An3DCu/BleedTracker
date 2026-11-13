import math

import numpy as np
import pandas as pd
import pytest

from bleedtracker.config import days_in_month
from bleedtracker.deployments import PAIRS, Ladder
from bleedtracker.leakage import leak_probe, no_outcome_or_snapshot_features
from bleedtracker.monitor import (
    GateResult,
    alarm_row,
    decisions,
    in_a_row,
    month_over_month,
    pre_release_gate,
    psi,
)
from bleedtracker.release import ReleaseHeld, month_over_month_alarms, open_sealed_month


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


def test_original_a1_misses_a_move_of_exactly_20_percent():
    # 1,200 pages against 1,000 is a move of exactly 20%. In floats, 1.2 - 1 is just
    # under 0.2.
    gate = pre_release_gate(frame(1000, 0), frame(1200, 1), frame(1000, 2), 30, 31)
    assert not gate.A1_population_moved


def test_fixed_a1_catches_a_move_of_exactly_20_percent():
    gate = pre_release_gate(
        frame(1000, 0), frame(1200, 1), frame(1000, 2), 30, 31, original_a1=False
    )
    assert gate.A1_population_moved
    assert gate.decision == "HOLD"


def test_fixed_a1_stays_quiet_just_under_20_percent():
    # 11,999 pages against 10,000 is a move of 19.99%.
    gate = pre_release_gate(
        frame(10000, 0), frame(11999, 1), frame(10000, 2), 30, 31, original_a1=False
    )
    assert not gate.A1_population_moved


def test_a7_stays_quiet_on_a_steady_month():
    result = month_over_month(frame(5000, 1), frame(5000, 2), 31, 31)
    assert not result["A7_drifted_since_last_month"]


def test_a7_fires_when_a_feature_moved_since_last_month():
    result = month_over_month(frame(5000, 1), frame(5000, 2, position_shift=10), 31, 31)
    assert result["A7_drifted_since_last_month"]
    assert result["worst_psi"] == result["psi_against_previous_month"]["position"]


def test_a7_reads_active_days_per_day():
    previous, score = frame(5000, 1), frame(5000, 2)
    result = month_over_month(previous, score, 30, 31)
    per_day = psi(previous.active_days / 30, score.active_days / 31)
    assert result["psi_against_previous_month"]["active_days"] == round(per_day, 4)
    assert per_day != psi(previous.active_days, score.active_days)


def test_a7_never_changes_the_gate_decision():
    # The scored month matches its training month. A2 stays quiet. The same clients
    # moved since the month before. A7 fires. The gate still releases.
    train = frame(5000, 0, position_shift=10)
    previous = frame(5000, 1)
    score = frame(5000, 2, position_shift=10)
    gate = pre_release_gate(train, score, previous, 31, 31)
    assert gate.decision == "RELEASE"
    assert month_over_month(previous, score, 31, 31)["A7_drifted_since_last_month"]


def test_a7_runs_on_the_four_forward_releases_for_held_out_clients_only():
    # Training clients move a lot every month. Held-out clients stay steady. Their live
    # days follow each month's length.
    frames = {}
    for i, pair in enumerate(PAIRS):
        days = days_in_month(pair[0])
        share = np.random.default_rng(30 + i).uniform(0.6, 1.0, 3000)
        held_out = frame(3000, 10 + i).assign(client_hash_id="held_out", active_days=share * days)
        training = frame(3000, 20 + i, position_shift=10 * i).assign(client_hash_id="training")
        frames[pair] = pd.concat([held_out, training], ignore_index=True).assign(label_declining=0)
    rows = month_over_month_alarms(frames, Ladder({"training"}, {"held_out"}))
    assert [row["release"] for row in rows] == [f"{a} to {b}" for a, b in PAIRS[1:]]
    assert not any(row["A7_drifted_since_last_month"] for row in rows)


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
