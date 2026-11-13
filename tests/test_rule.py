import math

import pandas as pd
import pytest

from bleedtracker.rule import action_score, band_median_ctr, reason_code, signals


def train_frame() -> pd.DataFrame:
    # 1-3: CTRs 0.02, 0.04, 0.06 -> median 0.04. Position 3.0 is the band's top edge.
    # 4-10: position 3.5 is already in this band. CTRs 0.01, 0.03 -> median 0.02.
    # 21+: one page with CTR 0 -> median 0. No training page in 11-20.
    return pd.DataFrame(
        {
            "pos_mar": [1.0, 2.0, 3.0, 3.5, 10.0, 25.0],
            "ctr_mar": [0.02, 0.04, 0.06, 0.01, 0.03, 0.0],
        }
    )


def pages() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "imp": [999.0, 500.0, 300.0, 200.0, 400.0, 150.0],
            "momentum_in_month": [0.4, 1.0, 0.0, 0.6, 0.8, 5.0],
            "pos_mar": [2.0, 2.0, 25.0, 3.5, 20.0, 1.0],
            "ctr_mar": [0.01, 0.01, 0.0, 0.05, 0.001, 0.0],
        },
        index=["A", "B", "C", "D", "E", "F"],
    )


def test_band_medians_are_learned_from_the_rows_given():
    medians = band_median_ctr(train_frame())
    assert medians.to_dict() == pytest.approx({"1-3": 0.04, "4-10": 0.02, "21+": 0.0})


def test_signals():
    s = signals(pages(), band_median_ctr(train_frame()))
    # A: 1 - 0.4 / 0.8 = 0.5. D: 1 - 0.6 / 0.8 = 0.25. F: momentum 5 clips to 0.
    assert s.severity.tolist() == pytest.approx([0.5, 0.0, 1.0, 0.25, 0.0, 0.0])
    # E sits in 11-20, which has no learned median. C's CTR 0 is not below a median of 0.
    assert s.underclick.tolist() == [1, 1, 0, 0, 0, 1]
    # Positions 1 and 20 are both reachable. 25 is not.
    assert s.reachable.tolist() == [1, 1, 0, 1, 1, 1]


def test_action_score():
    score = action_score(pages(), band_median_ctr(train_frame()))
    expected = [
        0.5 * math.log(1000) * 2 * 1.5,  # A: sliding, underclicking, reachable
        0.0,  # B: flat
        1.0 * math.log(301) * 1 * 1,  # C: collapsed, deep
        0.25 * math.log(201) * 1 * 1.5,  # D: sliding, reachable
        0.0,  # E: momentum exactly 0.8
        0.0,  # F: growing
    ]
    assert score.tolist() == pytest.approx(expected)


def test_reason_codes():
    codes = reason_code(pages(), band_median_ctr(train_frame()))
    assert codes.to_dict() == {
        "A": "slipping_and_underclicking",
        "B": "holding_steady",
        "C": "slipping_deep_page",
        "D": "slipping_reachable_page",
        "E": "holding_steady",
        "F": "holding_steady",
    }
