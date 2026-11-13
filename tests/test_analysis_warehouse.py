"""Comparisons, intervals, the data facts and the limitations, checked against my original results.

Runs only with a token: pytest -m warehouse
"""

import pytest

from bleedtracker.comparisons import per_client_queue, precision_curve, sealed_comparison
from bleedtracker.deployments import SEALED_PAIR, plans
from bleedtracker.intervals import uncertainty
from bleedtracker.label import without_outcome
from bleedtracker.population import (
    biggest_bleeds,
    data_facts,
    decision_envelope,
    month_table,
    population_audit,
    snapshot_audit,
    triage_numbers,
)
from bleedtracker.reference import load_reference
from bleedtracker.release import OPEN_PAIRS, feature_worth

pytestmark = pytest.mark.warehouse

REFERENCE = load_reference()


@pytest.fixture(scope="module")
def sealed(warehouse_run):
    return warehouse_run.ladder.queues[plans()[-1].name]


@pytest.fixture(scope="module")
def sealed_features(warehouse_run):
    return without_outcome(warehouse_run.frames[SEALED_PAIR])


def test_sealed_comparison(sealed):
    _, pool = sealed
    for row, expected in zip(sealed_comparison(pool), REFERENCE["sealed_comparison"], strict=True):
        assert {k: row[k] for k in expected} == expected, row["scorer"]


def test_precision_curve(sealed):
    assert precision_curve(sealed[1]) == REFERENCE["sealed_precision_at_k"]


def test_per_client_queue(sealed):
    assert per_client_queue(sealed[1]) == REFERENCE["per_client_queue"]


def test_uncertainty(warehouse_run):
    ladder = warehouse_run.ladder
    assert uncertainty(ladder.rows, ladder.shadow_queue) == REFERENCE["uncertainty"]


def test_feature_worth(warehouse_run):
    worth = feature_worth(warehouse_run.frames, warehouse_run.ladder)
    for name, expected in REFERENCE["feature_worth"].items():
        assert {k: worth[name][k] for k in expected} == expected, name


def test_population_audit(warehouse_run, sealed_features):
    audit = population_audit(sealed_features, warehouse_run.ladder.test_clients)
    audit |= snapshot_audit(warehouse_run.con, warehouse_run.source, sealed_features)
    assert audit == REFERENCE["population_audit"]


def test_decision_envelope(warehouse_run, sealed_features):
    open_frames = {pair: warehouse_run.frames[pair] for pair in OPEN_PAIRS}
    assert decision_envelope(sealed_features, open_frames) == REFERENCE["decision_envelope"]


def test_data_facts(warehouse_run, sealed_features):
    facts = data_facts(warehouse_run.con, warehouse_run.source, SEALED_PAIR[0], sealed_features)
    assert facts == REFERENCE["data"]


def test_limitations(warehouse_run, sealed):
    queue, pool = sealed
    expected = REFERENCE["limitations"]
    held_out = pool.client_hash_id.value_counts(normalize=True)
    assert round(float(held_out.iloc[0]), 4) == expected["sealed_pool_largest_client_share"]
    assert (
        month_table(warehouse_run.frames, warehouse_run.ladder.test_clients)
        == expected["month_table"]
    )
    triage = triage_numbers(queue, pool)
    assert {k: expected[k] for k in triage} == triage
    assert biggest_bleeds(pool) == expected["biggest_bleeds"]
