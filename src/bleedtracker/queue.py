"""The queue an editor gets. The rule fills 50 slots and the forest fills 10 shadow items."""

import numpy as np
import pandas as pd

from bleedtracker.gates import ACTIONS, apply_gates

EDITOR_SLOTS = 50
SHADOW_ITEMS = 10


def build_queue(
    scored: pd.DataFrame,
    days_in_month: int,
    editor_slots: int = EDITOR_SLOTS,
    shadow_items: int = SHADOW_ITEMS,
    shadow_is_editor: bool = False,
    original_coverage: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the queue from a pool scored by the rule and the forest.

    The rule fills the editor slots from eligible pages it scores above zero. Ties go
    to the bigger page, then to row order. The forest fills the shadow items from
    eligible pages the rule scores zero. With `shadow_is_editor`, those items are
    editor slots too. The trial with 10 forest slots is built this way.

    Returns the queue and the pool with its gate flags.
    """
    d = apply_gates(scored.reset_index(drop=True), days_in_month, original_coverage)
    d["row_id"] = np.arange(len(d))
    d["forest_rank_pct"] = (d.forest_score.rank(pct=True) * 100).round(1)

    eligible = d[d.eligible == 1]
    rule_part = (
        eligible[eligible.action_score > 0]
        .sort_values(["action_score", "imp"], ascending=False)
        .head(editor_slots)
        .copy()
    )
    rule_part["queue"] = "A_rule"
    rule_part["slot_type"] = "editor_slot"

    forest_part = (
        eligible[eligible.action_score == 0]
        .sort_values(["forest_score", "imp"], ascending=False)
        .head(shadow_items)
        .copy()
    )
    forest_part["queue"] = "B_forest"
    forest_part["reason_code"] = "quiet_risk_model_only"
    forest_part["slot_type"] = "editor_slot" if shadow_is_editor else "shadow"

    q = pd.concat([rule_part, forest_part], ignore_index=True)
    q["reason_code"] = np.where(q.gate_verify == 1, "measurement_suspicion", q.reason_code)
    q["action"] = q.reason_code.map(ACTIONS)
    q = q.sort_values(
        ["queue", "action_score", "forest_score"], ascending=[True, False, False]
    ).reset_index(drop=True)
    q.insert(0, "slot", np.arange(1, len(q) + 1))

    if q.action.isna().any() or not q.row_id.is_unique:
        raise ValueError("Every queue item needs one action and a page of its own.")
    if (q.eligible == 0).any():
        raise ValueError("A gated page took a slot.")
    return q, d


def queue_stats(queue: pd.DataFrame, pool: pd.DataFrame) -> dict:
    """Precision, hits and the impressions a queue's pages went on to lose."""
    lost = (queue.imp - queue.imp_next).clip(lower=0)
    return {
        "items": len(queue),
        "precision": round(float(queue.label_declining.mean()), 4),
        "base_rate": round(float(pool.label_declining.mean()), 4),
        "hits": int(queue.label_declining.sum()),
        "impressions_lost": int(lost.sum()),
        "clients": int(queue.client_hash_id.nunique()),
    }


def page_disposition(queue: pd.DataFrame, pool: pd.DataFrame) -> dict:
    """Where every page in the pool went."""
    editor = int((queue.slot_type == "editor_slot").sum())
    shadow = int((queue.slot_type == "shadow").sum())
    eligible = pool[pool.eligible == 1]
    return {
        "editor_slots": editor,
        "shadow_items": shadow,
        "rising": int(pool.gate_riser.sum()),
        "no_first_half_traffic": int(pool.gate_no_first_half.sum()),
        "eligible_not_sliding": int((eligible.action_score == 0).sum()) - shadow,
        "eligible_sliding_below_the_cut": int((eligible.action_score > 0).sum()) - editor,
    }
