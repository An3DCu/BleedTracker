"""The anchor deployment and the four forward deployments. The last one is the sealed month.

Every deployment uses the clients split on the anchor month. The anchor trains and
judges on March, on different clients. The forward ones train on month t and judge
the held-out clients on month t + 1. There the forest meets clients it never saw, in
a month that hadn't happened when it was fitted.
"""

from dataclasses import dataclass, field

import pandas as pd
from sklearn.metrics import roc_auc_score

from bleedtracker.config import days_in_month
from bleedtracker.metrics import TOP_K
from bleedtracker.model import deploy, split_clients
from bleedtracker.queue import (
    EDITOR_SLOTS,
    SHADOW_ITEMS,
    build_queue,
    page_disposition,
    queue_stats,
)

PAIRS = [
    ("2026-01", "2026-02"),
    ("2026-02", "2026-03"),
    ("2026-03", "2026-04"),
    ("2026-04", "2026-05"),
    ("2026-05", "2026-06"),
]
ANCHOR_PAIR = ("2026-03", "2026-04")
SEALED_PAIR = ("2026-05", "2026-06")
TRIAL_SPLIT = (40, 10)


@dataclass(frozen=True)
class Plan:
    """Which labelled month a deployment trains on and which one it judges."""

    name: str
    train_pair: tuple[str, str]
    test_pair: tuple[str, str]


def plans() -> list[Plan]:
    """The anchor first, then the four forward deployments in time order."""
    result = [Plan(f"anchor {ANCHOR_PAIR[0]} to {ANCHOR_PAIR[1]}", ANCHOR_PAIR, ANCHOR_PAIR)]
    for train_pair, test_pair in zip(PAIRS, PAIRS[1:], strict=False):
        kind = "sealed" if test_pair == SEALED_PAIR else "forward"
        result.append(Plan(f"{kind} {test_pair[0]} to {test_pair[1]}", train_pair, test_pair))
    return result


@dataclass
class Ladder:
    """Everything the five deployments produce.

    The client sets and the queues hold page and client ids. They stay out of the printed form.
    An error message never shows them.
    """

    train_clients: set = field(repr=False)
    test_clients: set = field(repr=False)
    rows: list[dict] = field(default_factory=list)
    gate_effects: list[dict] = field(default_factory=list)
    shadow_queue: list[dict] = field(default_factory=list)
    queues: dict = field(default_factory=dict, repr=False)


def run_deployments(frames: dict, original_coverage: bool = True) -> Ladder:
    """Run every deployment and collect its numbers."""
    train_clients, test_clients = split_clients(frames[ANCHOR_PAIR])
    ladder = Ladder(train_clients, test_clients)
    for plan in plans():
        train = frames[plan.train_pair]
        test = frames[plan.test_pair]
        deployment = deploy(
            train[train.client_hash_id.isin(train_clients)],
            test[test.client_hash_id.isin(test_clients)],
        )
        days = days_in_month(plan.test_pair[0])
        queue, pool = build_queue(deployment.test, days, original_coverage=original_coverage)
        trial, _ = build_queue(
            deployment.test,
            days,
            editor_slots=TRIAL_SPLIT[0],
            shadow_items=TRIAL_SPLIT[1],
            shadow_is_editor=True,
            original_coverage=original_coverage,
        )
        ladder.queues[plan.name] = (queue, pool)
        ladder.rows.append(ladder_row(plan.name, queue, pool, trial))
        ladder.gate_effects.append(gate_row(plan.name, queue, pool))
        ladder.shadow_queue.append(shadow_row(plan.name, queue, pool))
    return ladder


def ladder_row(name: str, queue: pd.DataFrame, pool: pd.DataFrame, trial: pd.DataFrame) -> dict:
    editor = queue[queue.slot_type == "editor_slot"]
    shadow = queue[queue.slot_type == "shadow"]
    forest_top = pool.nlargest(TOP_K, "forest_score")
    rule = queue_stats(editor, pool)
    forest = queue_stats(forest_top, pool)
    return {
        "deployment": name,
        "clients": int(pool.client_hash_id.nunique()),
        "pages": len(pool),
        "base_rate": round(float(pool.label_declining.mean()), 4),
        "rule_precision_at_50": rule["precision"],
        "forest_precision_at_50": forest["precision"],
        "margin_points": round(100 * (rule["precision"] - pool.label_declining.mean()), 1),
        "rule_auc": round(float(roc_auc_score(pool.label_declining, pool.action_score)), 4),
        "forest_auc": round(float(roc_auc_score(pool.label_declining, pool.forest_score)), 4),
        "trial_40_10_precision": queue_stats(trial, pool)["precision"],
        "shadow_precision": queue_stats(shadow, pool)["precision"],
        "impressions_lost_rule_50": rule["impressions_lost"],
        "impressions_lost_forest_50": forest["impressions_lost"],
    }


def gate_row(name: str, queue: pd.DataFrame, pool: pd.DataFrame) -> dict:
    """What the gates took out, measured against the queues built without them."""
    editor = queue[queue.slot_type == "editor_slot"]
    shadow = queue[queue.slot_type == "shadow"]
    ungated_editor = pool[pool.action_score > 0].nlargest(EDITOR_SLOTS, "action_score")
    ungated_shadow = pool[pool.action_score == 0].nlargest(SHADOW_ITEMS, "forest_score")
    return {
        "deployment": name,
        "gated_out_percent": round(100 * (1 - pool.eligible.mean()), 2),
        "editor_items_swapped": int((~ungated_editor.row_id.isin(editor.row_id)).sum()),
        "shadow_items_swapped": int((~ungated_shadow.row_id.isin(shadow.row_id)).sum()),
        "verify_first_items": int((editor.reason_code == "measurement_suspicion").sum()),
    }


def shadow_row(name: str, queue: pd.DataFrame, pool: pd.DataFrame) -> dict:
    """The shadow queue against the eligible pages the rule scores zero."""
    shadow = queue[queue.slot_type == "shadow"]
    zero = pool[(pool.action_score == 0) & (pool.eligible == 1)]
    return {
        "deployment": name,
        "eligible_rule_zero_pages": len(zero),
        "eligible_rule_zero_base_rate": round(float(zero.label_declining.mean()), 4),
        "shadow_precision": round(float(shadow.label_declining.mean()), 4),
        "shadow_lift_points": round(
            (shadow.label_declining.mean() - zero.label_declining.mean()) * 100, 1
        ),
        "eligible_rule_zero_declines": int(zero.label_declining.sum()),
        "share_of_all_declines": round(
            float(zero.label_declining.sum() / pool.label_declining.sum()), 4
        ),
    }


def sealed_summary(queue: pd.DataFrame, pool: pd.DataFrame) -> dict:
    """What the sealed month's queue asked an editor to do."""
    editor = queue[queue.slot_type == "editor_slot"]
    shadow = queue[queue.slot_type == "shadow"]
    return {
        "editor_slots": len(editor),
        "shadow_items": len(shadow),
        "action_mix": queue.action.value_counts().to_dict(),
        "reason_mix": queue.reason_code.value_counts().to_dict(),
        "distinct_clients_editor": int(editor.client_hash_id.nunique()),
        "editor_precision": round(float(editor.label_declining.mean()), 4),
        "shadow_precision": round(float(shadow.label_declining.mean()), 4),
        "verify_first_items": int((editor.reason_code == "measurement_suspicion").sum()),
        "editor_median_impressions": int(editor.imp.median()),
        "shadow_median_impressions": int(shadow.imp.median()),
        "page_disposition": page_disposition(queue, pool),
    }
