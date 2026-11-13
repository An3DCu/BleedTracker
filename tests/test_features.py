import math
from datetime import date

import pytest

from bleedtracker.config import FEATURES, days_in_month, second_half_start
from bleedtracker.features import build_features, load_page_months

# The split dates written out by hand in my original analysis.
ORIGINAL_SECOND_HALF = {
    "2026-01": date(2026, 1, 17),
    "2026-02": date(2026, 2, 15),
    "2026-03": date(2026, 3, 17),
    "2026-04": date(2026, 4, 16),
    "2026-05": date(2026, 5, 17),
    "2026-06": date(2026, 6, 16),
}


@pytest.mark.parametrize("month", ORIGINAL_SECOND_HALF)
def test_second_half_matches_the_original_split(month):
    assert second_half_start(month) == ORIGINAL_SECOND_HALF[month]


def test_days_in_month():
    assert [days_in_month(m) for m in ["2026-01", "2026-02", "2026-04", "2028-02"]] == [
        31,
        28,
        30,
        29,
    ]


def test_slice_keeps_eligible_clients_and_pages_at_the_floor(con, tiny_source):
    pages = load_page_months(con, tiny_source, "2026-03")
    assert pages.content_hash_id.tolist() == ["p_late_only", "p_steady", "p_vanish"]


def test_page_month_sums_only_available_rows(con, tiny_source):
    pages = load_page_months(con, tiny_source, "2026-03").set_index("content_hash_id")
    late = pages.loc["p_late_only"]
    assert late.imp == 100
    assert late.clk == 5
    assert late.active_days == 1
    assert math.isnan(late.imp_early)
    assert late.imp_late == 100


def test_features_of_a_steady_page(con, tiny_source):
    f = build_features(load_page_months(con, tiny_source, "2026-03")).set_index("content_hash_id")
    steady = f.loc["p_steady"]
    assert steady.imp == 310
    assert steady.active_days == 31
    assert steady.pos_mar == 2.0
    assert steady.ctr_mar == pytest.approx(0.1)
    assert steady.log_imp_mar == pytest.approx(math.log(311))
    assert steady.momentum_in_month == pytest.approx(150 / 160)
    assert steady.spike_share == pytest.approx(10 / 310)
    assert steady.has_first_half_traffic == 1


def test_page_with_no_first_half_is_flat_and_flagged(con, tiny_source):
    for original in (True, False):
        f = build_features(load_page_months(con, tiny_source, "2026-03"), original)
        late = f.set_index("content_hash_id").loc["p_late_only"]
        assert late.momentum_in_month == 1.0
        assert late.has_first_half_traffic == 0


def test_original_momentum_scores_a_vanished_page_as_flat(con, tiny_source):
    f = build_features(load_page_months(con, tiny_source, "2026-03"), original_momentum=True)
    vanish = f.set_index("content_hash_id").loc["p_vanish"]
    assert vanish.momentum_in_month == 1.0
    assert vanish.has_first_half_traffic == 0


def test_fixed_momentum_scores_a_vanished_page_as_zero(con, tiny_source):
    f = build_features(load_page_months(con, tiny_source, "2026-03"), original_momentum=False)
    vanish = f.set_index("content_hash_id").loc["p_vanish"]
    assert vanish.momentum_in_month == 0.0
    assert vanish.has_first_half_traffic == 1
    assert vanish.pos_mar == 3.0
    assert vanish.spike_share == pytest.approx(60 / 110)


def test_no_missing_features(con, tiny_source):
    f = build_features(load_page_months(con, tiny_source, "2026-03"))
    assert not f[FEATURES].isna().any().any()
