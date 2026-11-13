"""The sealed month's queue as a CSV, the file an editor would have been handed.

It lists every held-out page, the queued ones first. It carries page and client ids.
It stays on the machine that ran it. Git ignores every CSV in the repo.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from bleedtracker.gates import ACTIONS, PLAYBOOK
from bleedtracker.label import OUTCOME_COLUMNS

COLUMNS = [
    "slot",
    "slot_type",
    "queue",
    "rule_rank",
    "disposition",
    "reason_code",
    "archetype",
    "action",
    "guardrail",
    "action_score",
    "forest_rank_pct",
    "client_hash_id",
    "content_hash_id",
    "impressions",
    "position",
    "ctr",
    "momentum",
    "active_days",
    "spike_share",
    "gate_no_first_half",
    "gate_riser",
    "gate_verify",
]


def queue_table(
    queue: pd.DataFrame, pool: pd.DataFrame, original_reasons: bool = True
) -> pd.DataFrame:
    """Every page in the pool with its slot, reason, action and guardrail.

    My original analysis called every eligible page without a slot "holding_steady".
    A page that was sliding below the cut got that reason too. `original_reasons`
    keeps that. Without it, such a page keeps the reason the rule gave it.
    """
    rule_reason = pool.reason_code
    table = pool.drop(columns=["reason_code"]).copy()
    table["disposition"] = np.select(
        [table.gate_no_first_half == 1, table.gate_riser == 1, table.action_score > 0],
        ["monitor_only: no first-half traffic", "monitor_only: was rising", "eligible: sliding"],
        default="eligible: not sliding",
    )
    queued = queue[["row_id", "slot", "slot_type", "queue", "reason_code", "action"]]
    table = table.merge(queued, on="row_id", how="left")
    eligible_reason = "holding_steady" if original_reasons else rule_reason.to_numpy()
    unqueued = np.where(
        table.gate_no_first_half == 1,
        "new_or_no_first_half_traffic",
        np.where(table.gate_riser == 1, "durable_riser", eligible_reason),
    )
    table["reason_code"] = table.reason_code.fillna(pd.Series(unqueued, index=table.index))
    table["action"] = table.action.fillna(table.reason_code.map(ACTIONS))
    table = table.merge(
        PLAYBOOK[["reason_code", "archetype", "guardrail"]], on="reason_code", how="left"
    )
    table["rule_rank"] = table.action_score.rank(ascending=False, method="first").astype(int)
    table = table.rename(columns={"imp": "impressions"})
    table = table.sort_values(["slot", "rule_rank"], na_position="last")
    return table[COLUMNS]


def write_queue_csv(
    queue: pd.DataFrame, pool: pd.DataFrame, path: Path, original_reasons: bool = True
) -> int:
    """Write the queue table and return how many rows it has."""
    table = queue_table(queue, pool, original_reasons=original_reasons)
    if set(OUTCOME_COLUMNS) & set(table.columns):
        raise ValueError("The queue must not carry the outcome it is judged by.")
    path.parent.mkdir(parents=True, exist_ok=True)
    table.round(6).to_csv(path, index=False)
    return len(table)
