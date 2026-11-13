from bleedtracker.features import build_features, load_page_months
from bleedtracker.label import attach_label, load_next_month


def labeled_march(con, source):
    frame = build_features(load_page_months(con, source, "2026-03"))
    return attach_label(frame, load_next_month(con, source, "2026-04")).set_index("content_hash_id")


def test_feature_frame_has_no_outcome(con, tiny_source):
    frame = build_features(load_page_months(con, tiny_source, "2026-03"))
    assert not {"imp_next", "seen_next_month", "label_declining"} & set(frame.columns)


def test_exactly_eighty_percent_is_not_a_decline(con, tiny_source):
    steady = labeled_march(con, tiny_source).loc["p_steady"]
    assert steady.imp_next == 248
    assert steady.label_declining == 0


def test_under_eighty_percent_is_a_decline(con, tiny_source):
    vanish = labeled_march(con, tiny_source).loc["p_vanish"]
    assert vanish.imp_next == 87
    assert vanish.label_declining == 1


def test_a_missing_page_counts_as_zero(con, tiny_source):
    late = labeled_march(con, tiny_source).loc["p_late_only"]
    assert late.seen_next_month == 0
    assert late.imp_next == 0
    assert late.label_declining == 1


def test_label_keeps_every_row(con, tiny_source):
    assert len(labeled_march(con, tiny_source)) == 3
