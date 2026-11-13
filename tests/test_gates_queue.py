import pandas as pd

from bleedtracker.gates import ACTIONS, PLAYBOOK, apply_gates
from bleedtracker.queue import build_queue, page_disposition, queue_stats


def gate_pool(**overrides) -> pd.DataFrame:
    base = {
        "has_first_half_traffic": 1,
        "momentum": 0.5,
        "spike_share": 0.1,
        "active_days": 31,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def gates(days: int = 31, original: bool = True, **overrides) -> pd.Series:
    return apply_gates(gate_pool(**overrides), days, original).iloc[0]


def test_momentum_of_exactly_1_2_is_a_riser():
    assert gates(momentum=1.2).gate_riser == 1
    assert gates(momentum=1.2).eligible == 0
    assert gates(momentum=1.19).eligible == 1


def test_no_first_half_is_out_and_never_counted_as_a_riser():
    g = gates(has_first_half_traffic=0, momentum=1.5)
    assert g.gate_no_first_half == 1
    assert g.gate_riser == 0
    assert g.eligible == 0


def test_spike_of_exactly_a_quarter_means_verify_first():
    assert gates(spike_share=0.25).gate_verify == 1
    assert gates(spike_share=0.24).gate_verify == 0


def test_original_coverage_gate_uses_a_share_of_31_days():
    # 23 live days in February give 23/28 = 0.82. That is above 25/31 = 0.81.
    # The original passes it.
    assert gates(days=28, original=True, active_days=23).gate_verify == 0
    # Counted in days, 23 is under 25.
    assert gates(days=28, original=False, active_days=23).gate_verify == 1


def test_coverage_in_a_31_day_month_is_the_same_either_way():
    for original in (True, False):
        assert gates(days=31, original=original, active_days=24).gate_verify == 1
        assert gates(days=31, original=original, active_days=25).gate_verify == 0


def test_playbook_has_one_action_per_reason_code():
    assert PLAYBOOK.reason_code.is_unique
    assert set(ACTIONS) == set(PLAYBOOK.reason_code)


def scored_pool() -> pd.DataFrame:
    rows = [
        # page, score, forest, imp, rising, first half, spike, declined
        ("a", 5.0, 0.1, 1000, False, True, 0.1, 1),
        ("b", 4.0, 0.2, 500, False, True, 0.1, 0),
        ("c", 4.0, 0.3, 900, False, True, 0.3, 1),  # ties with b, bigger page; spike
        ("d", 9.0, 0.9, 800, True, True, 0.1, 1),  # rising: gated out
        ("e", 3.0, 0.5, 700, False, True, 0.1, 1),
        ("f", 0.0, 0.8, 600, False, True, 0.1, 1),
        ("g", 0.0, 0.7, 650, False, True, 0.1, 0),
        ("h", 0.0, 0.95, 400, False, False, 0.1, 1),  # no first half: gated out
        ("i", 0.0, 0.1, 300, False, True, 0.1, 0),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "content_hash_id",
            "action_score",
            "forest_score",
            "imp",
            "rising",
            "first_half",
            "spike_share",
            "label_declining",
        ],
    )
    return df.assign(
        client_hash_id="c1",
        momentum=df.rising.map({True: 1.5, False: 0.5}),
        has_first_half_traffic=df.first_half.astype(int),
        active_days=31,
        imp_next=df.imp * 0.5,
        reason_code="slipping_and_underclicking",
    )


def test_editor_slots_take_the_top_scores_and_ties_go_to_the_bigger_page():
    queue, _ = build_queue(scored_pool(), 31, editor_slots=3, shadow_items=2)
    editor = queue[queue.slot_type == "editor_slot"]
    assert editor.content_hash_id.tolist() == ["a", "c", "b"]


def test_shadow_items_come_from_eligible_pages_the_rule_scores_zero():
    queue, _ = build_queue(scored_pool(), 31, editor_slots=3, shadow_items=2)
    shadow = queue[queue.slot_type == "shadow"]
    assert shadow.content_hash_id.tolist() == ["f", "g"]
    assert set(shadow.action) == {"investigate_quiet_risk"}


def test_a_spike_page_keeps_its_slot_and_gets_verify_first():
    queue, _ = build_queue(scored_pool(), 31, editor_slots=3, shadow_items=2)
    c = queue.set_index("content_hash_id").loc["c"]
    assert c.reason_code == "measurement_suspicion"
    assert c.action == "verify_then_review"


def test_slots_are_numbered_editor_first():
    queue, _ = build_queue(scored_pool(), 31, editor_slots=3, shadow_items=2)
    assert queue.slot.tolist() == [1, 2, 3, 4, 5]
    assert queue.slot_type.tolist() == ["editor_slot"] * 3 + ["shadow"] * 2


def test_the_trial_counts_forest_items_as_editor_slots():
    queue, _ = build_queue(scored_pool(), 31, editor_slots=2, shadow_items=2, shadow_is_editor=True)
    assert (queue.slot_type == "editor_slot").sum() == 4


def test_queue_stats():
    queue, pool = build_queue(scored_pool(), 31, editor_slots=3, shadow_items=2)
    editor = queue[queue.slot_type == "editor_slot"]
    stats = queue_stats(editor, pool)
    # Two of three declined, stored at 4 decimals like the original.
    assert stats["precision"] == 0.6667
    # Each page loses half its impressions: (1000 + 900 + 500) / 2.
    assert stats["impressions_lost"] == 1200


def test_every_page_is_accounted_for():
    queue, pool = build_queue(scored_pool(), 31, editor_slots=3, shadow_items=2)
    disposition = page_disposition(queue, pool)
    assert disposition == {
        "editor_slots": 3,
        "shadow_items": 2,
        "rising": 1,
        "no_first_half_traffic": 1,
        "eligible_not_sliding": 1,
        "eligible_sliding_below_the_cut": 1,
    }
    assert sum(disposition.values()) == len(pool)
