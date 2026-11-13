"""My upgrades, run next to my original analysis.

v1 is my original analysis, reproduced. v2 adds the four fixes, the per-day features
and rule v2's lines. Every result here is post-hoc. v2's results on the open months
are partly in-sample. Its lines were picked on those months' training clients.
"""

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from bleedtracker.check import Comparison, compare
from bleedtracker.config import FEATURES_V2, days_in_month
from bleedtracker.deployments import SEALED_PAIR, Ladder, plans, run_deployments
from bleedtracker.export import queue_table
from bleedtracker.features import per_day
from bleedtracker.measures import (
    BOOTSTRAP_DRAWS,
    paired_bootstrap,
    per_account_measures,
    queue_measures,
)
from bleedtracker.metrics import TOP_K
from bleedtracker.model import SEED
from bleedtracker.monitor import A7_PSI
from bleedtracker.queue import build_queue
from bleedtracker.release import (
    OPEN_PAIRS,
    gate_sealed_month,
    held_out_alarms,
    load_open_frames,
    load_sealed_features,
    month_over_month_alarms,
    open_sealed_month,
)
from bleedtracker.rule import V2_RISER_GATE, V2_SLIDE_LINE

UPGRADES_FILE = Path(__file__).resolve().parents[2] / "reports" / "upgrades.json"
NOT_CHECKED = {"environment"}
ABOUT = (
    "My upgrades, written by python -m bleedtracker upgrades. v1 is my original analysis. "
    "It is rebuilt and checked against reports/results.json before any upgrade runs. v2 adds "
    "the four fixes, the per-day features and rule v2's lines. Every result is post-hoc. "
    "v2's results on the open months are partly in-sample. Its lines were picked on those "
    "months' training clients."
)


def load_upgrades(path: Path = UPGRADES_FILE) -> dict:
    """Read the committed upgrades file."""
    return json.loads(path.read_text(encoding="utf-8"))


def check_originals(rebuilt: dict, committed: dict) -> Comparison:
    """Compare a fresh rebuild of my original results with the committed file."""
    rebuilt = json.loads(json.dumps(rebuilt))
    return compare(
        {k: v for k, v in rebuilt.items() if k not in NOT_CHECKED},
        {k: v for k, v in committed.items() if k not in NOT_CHECKED},
    )


def load_v2_frames(con, source) -> tuple[dict, pd.DataFrame, object]:
    """Every month with v2's features. The sealed month opens only through its own gate.

    The gate runs on v2's features with the fixed A1. Returns the frames, the sealed
    month's features and the gate.
    """
    frames = {
        pair: per_day(frame, pair[0])
        for pair, frame in load_open_frames(con, source, original_momentum=False).items()
    }
    sealed = per_day(load_sealed_features(con, source, original_momentum=False), SEALED_PAIR[0])
    gate = gate_sealed_month(frames, sealed, original_a1=False)
    if gate.decision == "RELEASE":
        frames[SEALED_PAIR] = open_sealed_month(con, source, sealed, gate)
    return frames, sealed, gate


def empty_second_half_pages(frames: dict, sealed: pd.DataFrame) -> dict:
    """Pages in each feature month's slice whose rows stopped before the second half."""
    months = {pair[0]: frame for pair, frame in frames.items() if pair in OPEN_PAIRS}
    months[SEALED_PAIR[0]] = sealed
    return {
        month: int(((frame.has_first_half_traffic == 1) & frame.imp_late.isna()).sum())
        for month, frame in months.items()
    }


def coverage_effect(ladder: Ladder) -> list[dict]:
    """Editor slots marked "verify first" under my original coverage gate and the fixed one."""
    rows = []
    for plan in plans():
        queue, pool = ladder.queues[plan.name]
        fixed, _ = build_queue(pool, days_in_month(plan.test_pair[0]), original_coverage=False)
        rows.append(
            {
                "deployment": plan.name,
                "original": verify_first(queue),
                "fixed": verify_first(fixed),
            }
        )
    return rows


def verify_first(queue: pd.DataFrame) -> int:
    """How many editor slots ask to verify first."""
    editor = queue[queue.slot_type == "editor_slot"]
    return int((editor.reason_code == "measurement_suspicion").sum())


def a1_effect(frames: dict, ladder: Ladder, original_alarms: list[dict]) -> list[dict]:
    """A1 on each forward release under my original compare and the rounded one."""
    fixed = held_out_alarms(frames, ladder, original_a1=False)
    return [
        {
            "release": before["release"],
            "original": before["A1_population_moved"],
            "fixed": after["A1_population_moved"],
        }
        for before, after in zip(original_alarms, fixed, strict=True)
    ]


def csv_reason_effect(ladder: Ladder) -> dict:
    """Reason codes in the sealed queue CSV under my original reasons and the fixed ones."""
    queue, pool = ladder.queues[plans()[-1].name]
    counts = {}
    for name, original in [("original", True), ("fixed", False)]:
        table = queue_table(queue, pool, original_reasons=original)
        counts[name] = {k: int(v) for k, v in table.reason_code.value_counts().items()}
    return counts


def version_entry(queue: pd.DataFrame, pool: pd.DataFrame, draws: int) -> dict:
    """The rule, the forest, both per-account queues and the bootstrap for one version."""
    editor = queue[queue.slot_type == "editor_slot"]
    return {
        "rule": queue_measures(editor, pool),
        "forest": queue_measures(pool.nlargest(TOP_K, "forest_score"), pool),
        "per_account_rule": per_account_measures(pool, "action_score", eligible_only=True),
        "per_account_forest": per_account_measures(pool, "forest_score", eligible_only=False),
        "bootstrap": paired_bootstrap(pool, draws),
    }


def deployment_entries(v1: Ladder, v2: Ladder, draws: int) -> list[dict]:
    """One entry per deployment. v1 and v2 sit side by side."""
    entries = []
    for plan in plans():
        entry = {
            "deployment": plan.name,
            "post_hoc": True,
            "in_sample": {"v1": False, "v2": plan.test_pair in OPEN_PAIRS},
            "v1": version_entry(*v1.queues[plan.name], draws),
        }
        if plan.name in v2.queues:
            entry["v2"] = version_entry(*v2.queues[plan.name], draws)
        else:
            entry["v2"] = "HOLD"
        entries.append(entry)
    return entries


def settings() -> dict:
    """The settings every upgrade ran with."""
    return {
        "v2_features": FEATURES_V2,
        "v2_slide_line": V2_SLIDE_LINE,
        "v2_riser_gate": V2_RISER_GATE,
        "a7_psi": A7_PSI,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "seed": SEED,
        "top_k": TOP_K,
    }


def build_upgrades(
    con,
    source,
    run,
    original_alarms: list[dict],
    draws: int = BOOTSTRAP_DRAWS,
    progress: Callable[[str], None] = print,
) -> tuple[dict, tuple | None]:
    """Run every upgrade. Returns the report and v2's sealed queue and pool, if released."""
    progress("Reading every month with v2's features")
    frames_v2, sealed_v2, gate_v2 = load_v2_frames(con, source)
    progress(f"Pre-release gate on v2's sealed month: {gate_v2.decision}")
    progress("Running the five deployments with v2")
    ladder_v2 = run_deployments(
        frames_v2,
        original_coverage=False,
        features=FEATURES_V2,
        slide_line=V2_SLIDE_LINE,
        riser_gate=V2_RISER_GATE,
    )
    progress("Counting what each fix changed on its own")
    fixes = {
        "post_hoc": True,
        "in_sample": False,
        "empty_second_half_pages": empty_second_half_pages(frames_v2, sealed_v2),
        "coverage_verify_first": coverage_effect(run.ladder),
        "a1": a1_effect(run.frames, run.ladder, original_alarms),
        "queue_csv_reasons": csv_reason_effect(run.ladder),
    }
    progress(f"Measuring every queue and running the bootstraps ({draws:,} draws each)")
    report = {
        "about": ABOUT,
        "settings": settings(),
        "fixes": fixes,
        "sealed_gate_v2": {"post_hoc": True, "in_sample": False, **gate_v2.summary()},
        "deployments": deployment_entries(run.ladder, ladder_v2, draws),
        "a7": {
            "post_hoc": True,
            "in_sample": False,
            "releases": month_over_month_alarms(run.frames, run.ladder),
        },
    }
    sealed_name = plans()[-1].name
    sealed_queue = ladder_v2.queues.get(sealed_name)
    return report, sealed_queue
