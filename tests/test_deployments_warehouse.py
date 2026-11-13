"""The five deployments rebuilt from the warehouse and checked against my original results.

Runs only with a token: pytest -m warehouse
"""

import pytest

from bleedtracker.deployments import SEALED_PAIR, plans, sealed_summary
from bleedtracker.reference import load_reference

pytestmark = pytest.mark.warehouse

REFERENCE = load_reference()


@pytest.mark.parametrize("section", ["deployments", "gate_effects", "shadow_queue"])
def test_section_matches(warehouse_run, section):
    ladder = warehouse_run.ladder
    got = {
        "deployments": ladder.rows,
        "gate_effects": ladder.gate_effects,
        "shadow_queue": ladder.shadow_queue,
    }[section]
    for row, expected in zip(got, REFERENCE[section], strict=True):
        assert row == expected, row["deployment"]


def test_sealed_queue_matches(warehouse_run):
    sealed_name = plans()[-1].name
    assert sealed_name.endswith(f"{SEALED_PAIR[0]} to {SEALED_PAIR[1]}")
    queue, pool = warehouse_run.ladder.queues[sealed_name]
    summary = sealed_summary(queue, pool)
    assert {k: summary[k] for k in REFERENCE["sealed_queue"]} == REFERENCE["sealed_queue"]
