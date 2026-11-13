"""The margin, the lost-impressions share, the per-account queue and the paired bootstrap."""

import numpy as np
import pandas as pd
import pytest

from bleedtracker.measures import (
    editor_rank,
    forest_rank,
    paired_bootstrap,
    per_account_measures,
    queue_measures,
    slot_split,
    tie_fair_sum,
)


def test_queue_measures_worked_by_hand():
    # Pool of 4 pages. Two declined. Lost impressions 60, 0, 40 and 0 make 100.
    pool = pd.DataFrame(
        {
            "imp": [100.0, 50.0, 80.0, 10.0],
            "imp_next": [40.0, 70.0, 40.0, 10.0],
            "label_declining": [1, 0, 1, 0],
        }
    )
    m = queue_measures(pool.iloc[[0, 1]], pool)
    # Precision 1 of 2 against a base rate of 2 of 4 is a margin of 0 points.
    assert list(m)[0] == "margin_points"
    assert m == {
        "margin_points": 0.0,
        "precision": 0.5,
        "base_rate": 0.5,
        "lost_impressions_share": 0.6,
        "random_lost_impressions_share": 0.5,
    }


def test_tie_fair_sum_shares_the_tied_slot():
    # The top score takes 10. Two pages tie for the last slot. Each counts half of
    # 20 + 40.
    assert tie_fair_sum([3, 2, 2, 1], [10, 20, 40, 5], 2) == pytest.approx(40.0)


def test_slot_split_worked_by_hand():
    # One slot each leaves 47. Quotas are 42.3, 2.82 and 1.88. Floors give 45. The two
    # slots left go to the largest remainders, 0.88 (c) and 0.82 (b).
    split = slot_split(pd.Series({"a": 900, "b": 60, "c": 40}))
    assert split.to_dict() == {"a": 43, "b": 4, "c": 3}
    assert split.sum() == 50


def test_slot_split_gives_a_tiny_client_one_slot():
    split = slot_split(pd.Series({"big": 10_000, "tiny": 1}))
    assert split.to_dict() == {"big": 49, "tiny": 1}


def test_slot_split_breaks_a_remainder_tie_by_order():
    # One slot each leaves 47. Each quota is 15.67. Floors give 45. The remainders and
    # the pages tie. The two slots left go to the first two clients.
    split = slot_split(pd.Series({"x": 10, "y": 10, "z": 10}))
    assert split.to_dict() == {"x": 17, "y": 17, "z": 16}


def account_pool() -> pd.DataFrame:
    # Client a has 90 pages, client b has 10. Odd-numbered pages declined and lost 50
    # impressions each.
    rows = []
    for client, pages in [("a", 90), ("b", 10)]:
        for i in range(pages):
            rows.append(
                {
                    "client_hash_id": client,
                    "action_score": float(pages - i),
                    "forest_score": float(i),
                    "eligible": 1,
                    "imp": 100.0,
                    "imp_next": 100.0 - (i % 2) * 50,
                    "label_declining": i % 2,
                }
            )
    return pd.DataFrame(rows)


def test_per_account_measures_worked_by_hand():
    # One slot each leaves 48. Quotas are 43.2 and 4.8. Floors give 47. The last slot
    # goes to b's remainder of 0.8. Slots are 44 for a and 6 for b.
    m = per_account_measures(account_pool(), "action_score", eligible_only=True)
    assert m["slots_per_client"] == [6, 44]
    # a's top 44 pages hold 22 declines. b's top 6 hold 3. That's 25 of 50.
    assert m["precision"] == pytest.approx(25 / 50)
    # The pool lost 50 x 50 = 2,500 impressions. The queue holds 25 x 50.
    assert m["lost_impressions_share"] == pytest.approx(25 * 50 / 2500)


def test_per_account_refuses_a_client_with_too_few_pages():
    pool = account_pool()
    pool.loc[pool.client_hash_id == "b", "eligible"] = 0
    with pytest.raises(ValueError):
        per_account_measures(pool, "action_score", eligible_only=True)


def bootstrap_pool(forest_hits: bool) -> pd.DataFrame:
    # 200 pages. The rule can only take pages 0 to 49. The forest ranks pages 100 to
    # 199 first. Every resample holds far more than 50 of those 100 pages.
    i = np.arange(200)
    labels = i >= 100 if forest_hits else i < 50
    return pd.DataFrame(
        {
            "action_score": np.where(i < 50, 100.0 - i, 0.0),
            "forest_score": np.where(i >= 100, 1.0 - i / 1000, 0.0),
            "eligible": 1,
            "imp": 100.0,
            "label_declining": labels.astype(int),
        }
    )


def test_editor_rank_follows_the_editor_queue():
    pool = bootstrap_pool(True)
    rank = editor_rank(pool)
    assert rank[0] == 0 and rank[49] == 49
    assert np.isinf(rank[50:]).all()
    assert forest_rank(pool)[100] == 0


def test_paired_bootstrap_when_the_forest_always_wins():
    # On every resample, the forest's pages all declined and the rule's never did.
    result = paired_bootstrap(bootstrap_pool(True), draws=200)
    assert result == {"forest_minus_rule_points": 100.0, "interval_90": [100.0, 100.0]}


def test_paired_bootstrap_when_the_rule_always_wins():
    result = paired_bootstrap(bootstrap_pool(False), draws=200)
    assert result == {"forest_minus_rule_points": -100.0, "interval_90": [-100.0, -100.0]}
