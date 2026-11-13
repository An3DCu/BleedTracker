"""The pre-release gate, the alarms and the leakage audit, checked against my original results.

Runs only with a token: pytest -m warehouse
"""

import pytest

from bleedtracker.deployments import PAIRS
from bleedtracker.leakage import windows_do_not_overlap
from bleedtracker.monitor import PRE_REGISTERED, decisions
from bleedtracker.reference import load_reference
from bleedtracker.release import held_out_alarms, leak_probes

pytestmark = pytest.mark.warehouse

REFERENCE = load_reference()


def test_thresholds_are_the_pre_registered_ones():
    assert REFERENCE["pre_registered_thresholds"] == PRE_REGISTERED


def test_sealed_gate(warehouse_run):
    assert warehouse_run.gate.summary() == REFERENCE["sealed_gate"]


def test_alarms_and_decisions(warehouse_run):
    alarms = held_out_alarms(warehouse_run.frames, warehouse_run.ladder)
    for row, expected in zip(alarms, REFERENCE["alarms"], strict=True):
        assert row == expected, row["release"]
    assert decisions(alarms) == REFERENCE["decisions"]


def test_leak_probes(warehouse_run):
    probes = leak_probes(warehouse_run.frames, warehouse_run.ladder)
    assert probes["open"] == REFERENCE["leak_probe_open"]
    assert probes["sealed"] == REFERENCE["leak_probe_sealed"]


@pytest.mark.parametrize("pair", PAIRS)
def test_feature_and_label_months_never_overlap(warehouse_run, pair):
    assert windows_do_not_overlap(warehouse_run.con, warehouse_run.source, *pair)
