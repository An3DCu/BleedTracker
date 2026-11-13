import pytest

from bleedtracker.metrics import margin_points, precision_at_k, wilson_interval


def test_precision_without_ties():
    assert precision_at_k([3, 2, 1], [1, 0, 1], k=2) == 0.5


def test_tied_pages_share_the_last_slots():
    # Above the cut: 5 and 4, one hit. One slot left for three pages tied at 3,
    # two of which decline: 2/3 of a hit. (1 + 2/3) / 3 = 5/9.
    assert precision_at_k([5, 4, 3, 3, 3, 1], [1, 0, 1, 1, 0, 1], k=3) == pytest.approx(5 / 9)


def test_row_order_does_not_change_the_result():
    scores, labels = [5, 4, 3, 3, 3, 1], [1, 0, 1, 1, 0, 1]
    assert precision_at_k(scores[::-1], labels[::-1], k=3) == pytest.approx(5 / 9)


def test_all_tied_gives_the_base_rate():
    assert precision_at_k([0] * 8, [1, 0, 0, 1, 0, 1, 0, 1], k=4) == 0.5


def test_too_few_pages_is_an_error():
    with pytest.raises(ValueError):
        precision_at_k([1, 2], [0, 1], k=3)


def test_margin_in_points():
    # The sealed month in my original analysis: 0.98 against a 73.6% base rate.
    assert margin_points(0.98, 0.736) == pytest.approx(24.4)


def test_wilson_matches_my_original_receipt():
    # 49 of 50 in the sealed month. The original receipts give [0.9152, 0.9955].
    low, high = wilson_interval(49, 50)
    assert round(low, 4) == 0.9152
    assert round(high, 4) == 0.9955


def test_wilson_bounds_are_where_the_score_test_is_on_the_edge():
    # Each bound L satisfies (p - L)^2 = z^2 * L * (1 - L) / n, without the closed form.
    hits, n, z = 30, 50, 1.645
    for bound in wilson_interval(hits, n, z):
        assert (hits / n - bound) ** 2 == pytest.approx(z * z * bound * (1 - bound) / n)
