"""Reading the months in the order a release allows.

The four open months are read with their labels. The sealed month is read as
features only. Its label is joined only after the pre-release gate says "release".
`open_sealed_month` is the one place that joins it.
"""

import pandas as pd

from bleedtracker.comparisons import ablation, momentum_alone
from bleedtracker.config import days_in_month
from bleedtracker.deployments import ANCHOR_PAIR, PAIRS, SEALED_PAIR, Ladder
from bleedtracker.features import build_features, load_page_months
from bleedtracker.label import attach_label, load_next_month, without_outcome
from bleedtracker.leakage import leak_probe
from bleedtracker.model import Deployment, deploy, split_clients
from bleedtracker.monitor import GateResult, alarm_row, month_over_month, pre_release_gate
from bleedtracker.pipeline import labelled_month
from bleedtracker.warehouse import Source

OPEN_PAIRS = [pair for pair in PAIRS if pair != SEALED_PAIR]
TRAINING_PAIR_FOR_SEALED = PAIRS[PAIRS.index(SEALED_PAIR) - 1]


class ReleaseHeld(RuntimeError):
    """Raised when the sealed month is opened after the gate said "hold"."""


def load_open_frames(con, source: Source, original_momentum: bool = True) -> dict:
    """The four open months, each with its label."""
    return {pair: labelled_month(con, source, *pair, original_momentum) for pair in OPEN_PAIRS}


def load_sealed_features(con, source: Source, original_momentum: bool = True) -> pd.DataFrame:
    """The sealed feature month, with no outcome column."""
    return build_features(load_page_months(con, source, SEALED_PAIR[0]), original_momentum)


def gate_sealed_month(open_frames: dict, sealed_features: pd.DataFrame) -> GateResult:
    """Run the pre-release gate on the sealed month, from the open months only."""
    train_clients, test_clients = split_clients(open_frames[ANCHOR_PAIR])
    before = open_frames[TRAINING_PAIR_FOR_SEALED]
    return pre_release_gate(
        train=before[before.client_hash_id.isin(train_clients)],
        score=sealed_features[sealed_features.client_hash_id.isin(test_clients)],
        previous=before[before.client_hash_id.isin(test_clients)],
        train_days=days_in_month(TRAINING_PAIR_FOR_SEALED[0]),
        score_days=days_in_month(SEALED_PAIR[0]),
    )


def open_sealed_month(
    con, source: Source, sealed_features: pd.DataFrame, gate: GateResult
) -> pd.DataFrame:
    """Join the sealed month's label. Only allowed after the gate says "release"."""
    if gate.decision != "RELEASE":
        raise ReleaseHeld(
            "The pre-release gate held the sealed month. Its label stays closed. "
            f"A1 population moved is {gate.A1_population_moved}. "
            f"A2 features drifted is {gate.A2_features_drifted}."
        )
    return attach_label(sealed_features, load_next_month(con, source, SEALED_PAIR[1]))


def prepare_frames(con, source: Source, original_momentum: bool = True):
    """Every labelled month, with the sealed one opened through the gate."""
    frames = load_open_frames(con, source, original_momentum)
    sealed_features = load_sealed_features(con, source, original_momentum)
    gate = gate_sealed_month(frames, sealed_features)
    frames[SEALED_PAIR] = open_sealed_month(con, source, sealed_features, gate)
    return frames, gate


def forward_deployment(frames: dict, ladder: Ladder, test_pair: tuple) -> Deployment:
    """Refit the forward deployment that judges `test_pair`."""
    train = frames[PAIRS[PAIRS.index(test_pair) - 1]]
    test = frames[test_pair]
    return deploy(
        train[train.client_hash_id.isin(ladder.train_clients)],
        test[test.client_hash_id.isin(ladder.test_clients)],
    )


def leak_probes(frames: dict, ladder: Ladder) -> dict:
    """The planted leaks on the last open deployment and on the sealed one."""
    result = {}
    for name, test_pair in [("open", TRAINING_PAIR_FOR_SEALED), ("sealed", SEALED_PAIR)]:
        d = forward_deployment(frames, ladder, test_pair)
        result[name] = leak_probe(d.train, d.test, d.test.forest_score)
    return result


def feature_worth(frames: dict, ladder: Ladder) -> dict:
    """The ablation and momentum on its own, on the last open month and the sealed one."""
    result = {}
    for name, test_pair in [("open", TRAINING_PAIR_FOR_SEALED), ("sealed", SEALED_PAIR)]:
        d = forward_deployment(frames, ladder, test_pair)
        result[name] = {
            "ablation_auc": ablation(d.train, d.test, d.test.forest_score),
            "momentum_alone": momentum_alone(d.train, d.test),
            "train_base_rate": round(float(d.train.label_declining.mean()), 4),
            "judged_base_rate": round(float(d.test.label_declining.mean()), 4),
        }
    return result


def held_out_alarms(frames: dict, ladder: Ladder) -> list[dict]:
    """The six alarms for each forward release, in time order."""
    rows = []
    for i, (before_pair, now_pair) in enumerate(zip(PAIRS, PAIRS[1:], strict=False)):
        before = frames[before_pair]
        now = frames[now_pair]
        train = before[before.client_hash_id.isin(ladder.train_clients)]
        score = now[now.client_hash_id.isin(ladder.test_clients)]
        previous = before[before.client_hash_id.isin(ladder.test_clients)]
        gate = pre_release_gate(
            train,
            without_outcome(score),
            previous,
            days_in_month(before_pair[0]),
            days_in_month(now_pair[0]),
        )
        rows.append(
            alarm_row(
                release=f"{now_pair[0]} to {now_pair[1]}",
                gate=gate,
                base_rate_before=previous.label_declining.mean(),
                base_rate_now=score.label_declining.mean(),
                deployment=ladder.rows[i + 1],
                shadow=ladder.shadow_queue[i + 1],
            )
        )
    return rows


def month_over_month_alarms(frames: dict, ladder: Ladder) -> list[dict]:
    """A7 for each forward release, in time order.

    It compares the held-out clients with themselves one month earlier.
    """
    rows = []
    for before_pair, now_pair in zip(PAIRS, PAIRS[1:], strict=False):
        before = frames[before_pair]
        now = frames[now_pair]
        result = month_over_month(
            before[before.client_hash_id.isin(ladder.test_clients)],
            without_outcome(now[now.client_hash_id.isin(ladder.test_clients)]),
            days_in_month(before_pair[0]),
            days_in_month(now_pair[0]),
        )
        rows.append({"release": f"{now_pair[0]} to {now_pair[1]}", **result})
    return rows
