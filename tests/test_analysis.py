import numpy as np
import pandas as pd
import pytest

from bleedtracker.comparisons import (
    per_client_queue,
    precision_curve,
    random_draws,
    scorer_columns,
)
from bleedtracker.intervals import bootstrap_means, sealed_wilson, uncertainty
from bleedtracker.population import biggest_bleeds, month_table, triage_numbers
from bleedtracker.reference import new_name


def test_ctr_fix_flag_needs_size_reach_and_low_ctr():
    pool = pd.DataFrame(
        {
            "imp": [500, 499, 500, 500, 500],
            "position": [20.0, 5.0, 21.0, 5.0, 1.0],
            "ctr": [0.004, 0.004, 0.004, 0.005, 0.0],
            "momentum": 1.0,
        }
    ).assign(log_impressions=lambda f: np.log1p(f.imp))
    flagged = scorer_columns(pool).ctr_fix_flag > 0
    assert flagged.tolist() == [True, False, False, False, True]


def test_random_draws_repeat_with_the_same_seed():
    pool = pd.DataFrame({"label_declining": [1, 0] * 100})
    first, second = random_draws(pool), random_draws(pool)
    assert len(first) == 200
    np.testing.assert_array_equal(first, second)
    assert first.mean() == pytest.approx(0.5, abs=0.05)


def scored_pool() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 300
    return pd.DataFrame(
        {
            "client_hash_id": np.repeat(["a", "b", "c", "d"], [120, 100, 77, 3]),
            "action_score": rng.uniform(0, 5, n),
            "forest_score": rng.uniform(0, 1, n),
            "label_declining": rng.integers(0, 2, n),
        }
    )


def test_precision_curve_has_every_k_and_the_margin():
    curve = precision_curve(scored_pool())
    assert [row["K"] for row in curve] == [10, 25, 50, 75, 100, 150, 200, 250]
    first = curve[0]
    assert first["rule_margin_points"] == pytest.approx(
        (first["rule"] - first["base_rate"]) * 100, abs=0.06
    )


def test_per_client_queue_skips_clients_with_fewer_than_five_pages():
    assert per_client_queue(scored_pool())["clients"] == 3


def test_bootstrap_of_one_repeated_value_is_that_value():
    means = bootstrap_means([2.0, 2.0, 2.0], np.random.default_rng(0), draws=50)
    assert set(means) == {2.0}


def test_the_two_bootstraps_share_one_stream():
    rows = [{"forest_precision_at_50": 0.9, "rule_precision_at_50": 0.9}] + [
        {"forest_precision_at_50": f, "rule_precision_at_50": r}
        for f, r in [(0.78, 0.88), (0.98, 0.90), (0.94, 0.96), (1.0, 0.98)]
    ]
    shadow = [{"shadow_lift_points": 0.0}] + [
        {"shadow_lift_points": v} for v in [-6.4, 16.5, 65.6, -3.3]
    ]
    result = uncertainty(rows, shadow)
    # The four differences of my original ladder, in points.
    assert result["forest_minus_rule_points"] == [-10.0, 8.0, -2.0, 2.0]
    assert result["forest_minus_rule_mean_points"] == -0.5
    assert result["shadow_lift_mean_points"] == 18.1
    rng = np.random.default_rng(42)
    bootstrap_means(np.array([-0.1, 0.08, -0.02, 0.02]), rng)
    lift = bootstrap_means(np.array([-6.4, 16.5, 65.6, -3.3]), rng)
    assert result["shadow_lift_interval_90"] == [
        round(float(np.percentile(lift, 5)), 1),
        round(float(np.percentile(lift, 95)), 1),
    ]


def test_sealed_wilson_counts_hits_out_of_fifty():
    assert sealed_wilson(0.98) == [0.9152, 0.9955]


def frames_for_tables() -> dict:
    frame = pd.DataFrame(
        {
            "client_hash_id": ["a", "a", "b", "b"],
            "imp": [100.0, 200.0, 100.0, 400.0],
            "imp_next": [50.0, 300.0, 100.0, 100.0],
            "label_declining": [1, 0, 0, 1],
        }
    )
    return {("2026-01", "2026-02"): frame}


def test_month_table_worked_by_hand():
    row = month_table(frames_for_tables(), {"b"})[0]
    assert row["pair"] == "2026-01 to 2026-02"
    assert row["slice_base_rate"] == 0.5
    assert row["held_out_base_rate"] == 0.5
    # Ratios 0.5, 1.5, 1.0, 0.25 have a median of 0.75. One page of four grew.
    assert row["median_next_over_now"] == 0.75
    assert row["share_of_pages_that_grew"] == 0.25


def test_triage_numbers_worked_by_hand():
    pool = frames_for_tables()[("2026-01", "2026-02")]
    queue = pool.iloc[[3]].assign(slot_type="editor_slot")
    triage = triage_numbers(queue, pool)
    # The pool lost 50 + 300 = 350 impressions. The queued page lost 300 of them.
    assert triage["impressions_lost_by_pool"] == 350
    assert triage["impressions_in_front_of_editor"] == 300
    assert triage["queue_recall"] == 0.5


def test_biggest_bleeds_keep_no_page_impressions():
    pool = frames_for_tables()[("2026-01", "2026-02")].assign(
        momentum=[0.4, 1.0, 0.9, 0.2], action_score=[2.0, 0.0, 0.0, 3.0], forest_score=0.5
    )
    rows = biggest_bleeds(pool, count=2)
    assert all(set(r) == {"momentum", "rule_rank", "forest_rank"} for r in rows)
    assert rows[0] == {"momentum": 0.2, "rule_rank": 1, "forest_rank": 4}


def test_ablation_keys_get_new_feature_names():
    assert new_name("without_pos_mar") == "without_position"
    assert new_name("without_active_days") == "without_active_days"
    assert new_name("all_five") == "all_five"
