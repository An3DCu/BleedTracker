"""The five deployments rebuilt from the warehouse and checked against my original results.

Runs only with a token: pytest -m warehouse
"""

import pytest

from bleedtracker.deployments import (
    SEALED_PAIR,
    load_frames,
    plans,
    run_deployments,
    sealed_summary,
)
from bleedtracker.reference import load_reference
from bleedtracker.warehouse import MissingTokenError, Source, connect, read_token

pytestmark = pytest.mark.warehouse

REFERENCE = load_reference()


@pytest.fixture(scope="module")
def ladder():
    try:
        token = read_token()
    except MissingTokenError:
        pytest.skip("No Hugging Face token.")
    return run_deployments(load_frames(connect(token), Source()))


@pytest.mark.parametrize("section", ["deployments", "gate_effects", "shadow_queue"])
def test_section_matches(ladder, section):
    got = {
        "deployments": ladder.rows,
        "gate_effects": ladder.gate_effects,
        "shadow_queue": ladder.shadow_queue,
    }[section]
    for row, expected in zip(got, REFERENCE[section], strict=True):
        assert row == expected, row["deployment"]


def test_sealed_queue_matches(ladder):
    sealed_name = plans()[-1].name
    assert sealed_name.endswith(f"{SEALED_PAIR[0]} to {SEALED_PAIR[1]}")
    queue, pool = ladder.queues[sealed_name]
    assert sealed_summary(queue, pool) == REFERENCE["sealed_queue"]
