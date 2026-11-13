"""Picking rule v2's slide line and riser gate.

The pick sees the training clients of the four open month pairs. It never sees a
held-out client or the sealed month. Each month pair counts equally.

1. The slide line. Every pair of lines is scored by the mean tie-fair precision of
   rule v2's ranking over K = 50, 100, 150, 200 and 250. A tie goes to the pair
   closest to 0.8 and 1.20. The riser gate can't move this score. A page it removes
   has momentum of at least 1.05. The rule already scores such a page zero.
2. The riser gate. With the picked slide line, each gate is scored by the precision
   of the shadow queue's 10 items. Forest v2 is trained on each month's training
   pages and scores those same pages. This step is in-sample. A tie goes to the gate
   closest to 1.20. On these months the forest ranks rising pages too low for the
   shadow queue. This score can't move the gate either.

Rule v2's results on the open months are partly in-sample.
"""

import numpy as np
import pandas as pd

from bleedtracker.config import FEATURES_V2, days_in_month
from bleedtracker.features import per_day
from bleedtracker.gates import RISER_GATE, apply_gates
from bleedtracker.metrics import precision_at_k
from bleedtracker.model import make_forest
from bleedtracker.queue import build_queue
from bleedtracker.rule import SLIP_GATE, action_score, band_median_ctr, reason_code

SLIDE_LINES = [round(x, 2) for x in np.arange(0.60, 0.951, 0.05)]
RISER_GATES = [round(x, 2) for x in np.arange(1.05, 1.501, 0.05)]
PICK_KS = [50, 100, 150, 200, 250]


def pair_score(frame: pd.DataFrame, month: str, slide_line: float, riser_gate: float) -> float:
    """Mean tie-fair precision of rule v2's ranking of the eligible pages over PICK_KS."""
    medians = band_median_ctr(frame)
    scored = frame.assign(action_score=action_score(frame, medians, slide_line))
    pool = apply_gates(scored, days_in_month(month), original_coverage=False, riser_gate=riser_gate)
    eligible = pool[pool.eligible == 1]
    return float(
        np.mean(
            [precision_at_k(eligible.action_score, eligible.label_declining, k) for k in PICK_KS]
        )
    )


def distance(slide_line: float, riser_gate: float) -> float:
    """How far a pair of lines sits from the original 0.8 and 1.20."""
    return abs(slide_line - SLIP_GATE) + abs(riser_gate - RISER_GATE)


def pick(scores: list[dict]) -> dict:
    """The best row. A tie goes to the row closest to the original lines."""
    best = max(round(row["score"], 10) for row in scores)
    tied = [row for row in scores if round(row["score"], 10) == best]
    return min(tied, key=lambda row: distance(row["slide_line"], row["riser_gate"]))


def training_months(frames: dict, train_clients: set) -> dict:
    """Each open feature month's training pages and their per-day features."""
    months = {}
    for (feature_month, _), frame in frames.items():
        train = frame[frame.client_hash_id.isin(train_clients)]
        months[feature_month] = per_day(train, feature_month).reset_index(drop=True)
    return months


def score_grid(frames: dict, train_clients: set) -> list[dict]:
    """Step 1. Score every pair of lines on the training clients of each open month pair.

    `frames` maps each open month pair to its labeled frame with the fixed features.
    """
    months = training_months(frames, train_clients)
    rows = []
    for slide_line in SLIDE_LINES:
        for riser_gate in RISER_GATES:
            per_month = [
                pair_score(frame, month, slide_line, riser_gate) for month, frame in months.items()
            ]
            rows.append(
                {
                    "slide_line": slide_line,
                    "riser_gate": riser_gate,
                    "score": round(float(np.mean(per_month)), 4),
                }
            )
    return rows


def shadow_precision(scored: pd.DataFrame, month: str, riser_gate: float) -> float:
    """Precision of the shadow queue's items built from a pool scored by rule v2 and forest v2."""
    queue, _ = build_queue(
        scored, days_in_month(month), original_coverage=False, riser_gate=riser_gate
    )
    return float(queue[queue.queue == "B_forest"].label_declining.mean())


def score_riser_gates(frames: dict, train_clients: set, slide_line: float) -> list[dict]:
    """Step 2. Score every riser gate by the shadow queue's precision, in-sample."""
    scored_months = {}
    for month, frame in training_months(frames, train_clients).items():
        medians = band_median_ctr(frame)
        forest = make_forest().fit(frame[FEATURES_V2], frame.label_declining)
        scored_months[month] = frame.assign(
            action_score=action_score(frame, medians, slide_line),
            reason_code=reason_code(frame, medians, slide_line),
            forest_score=forest.predict_proba(frame[FEATURES_V2])[:, 1],
        )
    rows = []
    for riser_gate in RISER_GATES:
        per_month = [
            shadow_precision(scored, month, riser_gate) for month, scored in scored_months.items()
        ]
        rows.append(
            {
                "slide_line": slide_line,
                "riser_gate": riser_gate,
                "score": round(float(np.mean(per_month)), 4),
            }
        )
    return rows


def lines_report(
    slide_scores: list[dict], riser_scores: list[dict], frames: dict, train_clients: set
) -> dict:
    """The file `pick-lines` writes. Aggregates only."""
    slide = pick(slide_scores)
    riser = pick(riser_scores)
    return {
        "about": (
            "Rule v2's slide line and riser gate, picked by python -m bleedtracker pick-lines "
            "on the training clients of the four open month pairs. Each month pair counts "
            "equally. Step 1 scores every pair of lines by the mean tie-fair precision of "
            "rule v2's ranking over K = 50, 100, 150, 200 and 250. The riser gate can't move "
            "that score. Step 2 keeps the picked slide line and scores each riser gate by the "
            "precision of the shadow queue's 10 items. Forest v2 is trained and judged on the "
            "same pages in step 2. On these months step 2 can't move the gate either. Post-hoc."
        ),
        "month_pairs": [f"{a} to {b}" for a, b in frames],
        "training_pages": {
            f"{a} to {b}": int(frame.client_hash_id.isin(train_clients).sum())
            for (a, b), frame in frames.items()
        },
        "picked": {"slide_line": slide["slide_line"], "riser_gate": riser["riser_gate"]},
        "step_1_picked": slide,
        "step_1_original_lines": next(
            row
            for row in slide_scores
            if row["slide_line"] == SLIP_GATE and row["riser_gate"] == RISER_GATE
        ),
        "step_2_picked": riser,
        "step_2_original_gate": next(
            row for row in riser_scores if row["riser_gate"] == RISER_GATE
        ),
        "step_1_grid": slide_scores,
        "step_2_grid": riser_scores,
    }
