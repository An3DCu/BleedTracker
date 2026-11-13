import numpy as np
import pandas as pd

from bleedtracker.__main__ import main
from bleedtracker.check import compare, compare_with_reference
from bleedtracker.export import COLUMNS, queue_table
from bleedtracker.label import OUTCOME_COLUMNS
from bleedtracker.queue import build_queue


def test_matching_results_count_every_number():
    expected = {"a": 1, "b": {"c": 0.5, "d": [1.0, 2.0]}, "e": "RELEASE", "f": True}
    result = compare(expected, expected)
    assert result.all_match
    assert (result.numbers, result.matching) == (4, 4)
    assert result.summary() == "4 of 4 numbers match my original results."


def test_differences_name_the_value():
    got = {"a": 2, "b": {"d": [1.0]}, "e": "HOLD"}
    expected = {"a": 1, "b": {"c": 0.5, "d": [1.0, 2.0]}, "e": "RELEASE"}
    result = compare(got, expected)
    assert result.differences == [
        "a: original 1, rebuilt 2",
        "b.c: missing",
        "b.d: expected 2 items",
        "e: original 'RELEASE', rebuilt 'HOLD'",
    ]
    assert (result.numbers, result.matching) == (1, 0)


def test_extra_rebuilt_values_are_not_checked():
    result = compare({"a": 1, "extra": 9}, {"a": 1})
    assert result.all_match
    assert result.numbers == 1


def test_the_reference_description_is_not_a_result():
    assert compare_with_reference({"a": 1}, {"about": "text", "a": 1}).all_match


def test_ints_and_floats_with_the_same_value_match():
    assert compare({"a": 218}, {"a": 218.0}).all_match


def pool() -> pd.DataFrame:
    rows = [
        ("a", 5.0, 0.1, False, True),
        ("b", 3.0, 0.2, False, True),
        ("c", 0.0, 0.9, False, True),
        ("d", 0.0, 0.3, False, True),
        ("e", 4.0, 0.5, True, True),
        ("f", 0.0, 0.8, False, False),
    ]
    df = pd.DataFrame(
        rows, columns=["content_hash_id", "action_score", "forest_score", "rising", "first_half"]
    )
    return df.assign(
        client_hash_id="c1",
        imp=1000.0,
        imp_next=500.0,
        label_declining=1,
        seen_next_month=1,
        position=5.0,
        ctr=0.01,
        log_impressions=np.log1p(1000.0),
        momentum=df.rising.map({True: 1.5, False: 0.5}),
        has_first_half_traffic=df.first_half.astype(int),
        active_days=31,
        spike_share=0.1,
        reason_code="slipping_and_underclicking",
    )


def table() -> pd.DataFrame:
    queue, gated = build_queue(pool(), 31, editor_slots=2, shadow_items=1)
    return queue_table(queue, gated)


def test_queue_table_has_every_page_and_no_outcome():
    t = table()
    assert list(t.columns) == COLUMNS
    assert len(t) == 6
    assert not set(OUTCOME_COLUMNS) & set(t.columns)


def test_queued_pages_come_first():
    t = table()
    assert t.content_hash_id.tolist()[:3] == ["a", "b", "c"]
    assert t.slot.tolist()[:3] == [1, 2, 3]


def test_pages_without_a_slot_get_their_disposition():
    t = table().set_index("content_hash_id")
    assert t.loc["e", "disposition"] == "monitor_only: was rising"
    assert t.loc["e", "reason_code"] == "durable_riser"
    assert t.loc["f", "disposition"] == "monitor_only: no first-half traffic"
    assert t.loc["d", "action"] == "monitor"


def reasons_table(original: bool) -> pd.DataFrame:
    # Pages a and b slide and get a positive score. With one editor slot, b slides
    # below the cut. Pages c and d score zero and hold steady.
    reasons = {"a": "slipping_and_underclicking", "b": "slipping_reachable_page"}
    p = pool().assign(reason_code=lambda d: d.content_hash_id.map(reasons).fillna("holding_steady"))
    queue, gated = build_queue(p, 31, editor_slots=1, shadow_items=1)
    return queue_table(queue, gated, original_reasons=original).set_index("content_hash_id")


def test_original_reasons_call_a_page_below_the_cut_holding_steady():
    t = reasons_table(original=True)
    assert pd.isna(t.loc["b", "slot"])
    assert t.loc["b", "reason_code"] == "holding_steady"
    assert t.loc["b", "action"] == "monitor"


def test_fixed_reasons_keep_the_rule_reason_below_the_cut():
    t = reasons_table(original=False)
    assert t.loc["b", "reason_code"] == "slipping_reachable_page"
    assert t.loc["b", "action"] == "refresh_now"


def test_fixed_reasons_leave_queued_gated_and_steady_pages_alone():
    original, fixed = reasons_table(original=True), reasons_table(original=False)
    for page in ["a", "c", "d", "e", "f"]:
        assert fixed.loc[page, "reason_code"] == original.loc[page, "reason_code"]


def test_reproduce_without_a_token_exits_with_2(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)
    assert main(["reproduce"]) == 2
    assert "No Hugging Face token" in capsys.readouterr().out
