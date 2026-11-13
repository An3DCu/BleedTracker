"""Every result, built from the warehouse in one pass.

The sections and their names follow `reference/original.json`. A few sections carry
extra values the original never stored. Those are kept and left unchecked.
"""

import json
import sys
from collections.abc import Callable
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score

from bleedtracker.comparisons import per_client_queue, precision_curve, sealed_comparison
from bleedtracker.config import MIN_IMPRESSIONS
from bleedtracker.deployments import (
    ANCHOR_PAIR,
    SEALED_PAIR,
    plans,
    run_deployments,
    sealed_summary,
)
from bleedtracker.gates import COVERAGE_DAYS, RISER_GATE, SPIKE_GATE
from bleedtracker.intervals import uncertainty
from bleedtracker.label import without_outcome
from bleedtracker.metrics import TOP_K, precision_at_k
from bleedtracker.model import SEED
from bleedtracker.monitor import PRE_REGISTERED, decisions
from bleedtracker.pipeline import anchor_deployment
from bleedtracker.population import (
    biggest_bleeds,
    data_facts,
    decision_envelope,
    month_table,
    population_audit,
    snapshot_audit,
    triage_numbers,
)
from bleedtracker.release import (
    OPEN_PAIRS,
    feature_worth,
    held_out_alarms,
    leak_probes,
    prepare_frames,
)
from bleedtracker.rule import reason_code

RESULTS_FILE = Path(__file__).resolve().parents[2] / "reports" / "results.json"
ANCHOR_K = [10, 25, 50, 100, 250]
ABOUT = (
    "Rebuilt by python -m bleedtracker reproduce from warehouse build v20260703. "
    "Rates are rounded to 4 decimals. My original results were stored the same way."
)


def anchor_summary(frame: pd.DataFrame) -> dict:
    """The first deployment, with the numbers my original checked it by."""
    deployment, train_clients, test_clients = anchor_deployment(frame)
    test = deployment.test
    zero = test[test.action_score == 0]
    return {
        "feature_month": ANCHOR_PAIR[0],
        "label_month": ANCHOR_PAIR[1],
        "slice_pages": len(frame),
        "slice_clients": int(frame.client_hash_id.nunique()),
        "slice_base_rate": round(float(frame.label_declining.mean()), 4),
        "train_pages": len(deployment.train),
        "train_clients": len(train_clients),
        "test_pages": len(test),
        "test_clients": len(test_clients),
        "test_base_rate": round(float(test.label_declining.mean()), 4),
        "rule_precision_at_50": round(precision_at_k(test.action_score, test.label_declining), 4),
        "forest_precision_at_50": round(precision_at_k(test.forest_score, test.label_declining), 4),
        "forest_auc": round(float(roc_auc_score(test.label_declining, test.forest_score)), 4),
        "rule_zero_pages": len(zero),
        "rule_zero_declines": int(zero.label_declining.sum()),
        "band_median_ctr_percent": {
            band: round(100 * float(v), 4) for band, v in deployment.medians.items()
        },
        "reason_codes_in_slice": reason_code(frame, deployment.medians).value_counts().to_dict(),
        "rule_precision_at_k": {
            str(k): round(precision_at_k(test.action_score, test.label_declining, k), 4)
            for k in ANCHOR_K
        },
    }


def settings() -> dict:
    """The seed, the queue size and the gate settings."""
    return {
        "seed": SEED,
        "top_k": TOP_K,
        "gates": {
            "riser": RISER_GATE,
            "spike_share": SPIKE_GATE,
            "min_active_days": COVERAGE_DAYS,
            "min_impressions": MIN_IMPRESSIONS,
        },
    }


def load_results(path: Path = RESULTS_FILE) -> dict:
    """Read the committed results file."""
    return json.loads(path.read_text(encoding="utf-8"))


def environment() -> dict:
    """The versions this run used."""
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "duckdb": duckdb.__version__,
        "matplotlib": matplotlib.__version__,
        "warehouse_build": "v20260703",
        "seed": SEED,
        "top_k": TOP_K,
    }


class Run:
    """The months, the gate and the ladder from one read of the warehouse.

    Its printed form names no client or page.
    """

    def __init__(self, frames: dict, gate, ladder):
        self.frames, self.gate, self.ladder = frames, gate, ladder

    def __repr__(self) -> str:
        return "Run(frames, gate, ladder)"


def prepare_run(con, source, progress: Callable[[str], None] = print) -> Run:
    """Read every month through the gate and run the five deployments."""
    progress("Reading the four open months and the sealed month's features")
    frames, gate = prepare_frames(con, source)
    progress(f"Pre-release gate on the sealed month: {gate.decision}")
    progress("Running the five deployments")
    return Run(frames, gate, run_deployments(frames))


def build_results(con, source, run, progress: Callable[[str], None] = print) -> dict:
    """Rebuild every section of my original results from one run."""
    frames, gate, ladder = run.frames, run.gate, run.ladder
    sealed_name = plans()[-1].name
    queue, pool = ladder.queues[sealed_name]
    sealed_features = without_outcome(frames[SEALED_PAIR])

    progress("Rebuilding the first deployment")
    results = {"about": ABOUT, "anchor": anchor_summary(frames[ANCHOR_PAIR])}
    results["deployments"] = ladder.rows
    results["gate_effects"] = ladder.gate_effects
    results["shadow_queue"] = ladder.shadow_queue
    results["sealed_queue"] = sealed_summary(queue, pool)

    progress("Running the alarms and the leak probes")
    alarms = held_out_alarms(frames, ladder)
    probes = leak_probes(frames, ladder)
    results["pre_registered_thresholds"] = PRE_REGISTERED
    results["sealed_gate"] = gate.summary()
    results["alarms"] = alarms
    results["decisions"] = decisions(alarms)
    results["leak_probe_open"] = probes["open"]
    results["leak_probe_sealed"] = probes["sealed"]

    progress("Comparing the rule with simpler rules and putting intervals on the results")
    results["sealed_comparison"] = sealed_comparison(pool)
    results["sealed_precision_at_k"] = precision_curve(pool)
    results["per_client_queue"] = per_client_queue(pool)
    results["uncertainty"] = uncertainty(ladder.rows, ladder.shadow_queue)
    results["feature_worth"] = feature_worth(frames, ladder)

    progress("Counting the data facts and auditing the population")
    results["population_audit"] = population_audit(sealed_features, ladder.test_clients)
    results["population_audit"] |= snapshot_audit(con, source, sealed_features)
    open_frames = {pair: frames[pair] for pair in OPEN_PAIRS}
    results["decision_envelope"] = decision_envelope(sealed_features, open_frames)
    results["data"] = data_facts(con, source, SEALED_PAIR[0], sealed_features)
    held_out = pool.client_hash_id.value_counts(normalize=True)
    results["limitations"] = {
        "sealed_pool_largest_client_share": round(float(held_out.iloc[0]), 4),
        "month_table": month_table(frames, ladder.test_clients),
        **triage_numbers(queue, pool),
        "biggest_bleeds": biggest_bleeds(pool),
    }
    results["settings"] = settings()
    results["environment"] = environment()
    return results
