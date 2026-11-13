"""The pre-release gate, the alarms and the leakage audit, checked against my original results.

Runs only with a token: pytest -m warehouse
"""

import pytest

from bleedtracker.deployments import PAIRS
from bleedtracker.leakage import windows_do_not_overlap
from bleedtracker.monitor import PRE_REGISTERED, decisions
from bleedtracker.reference import load_reference
from bleedtracker.release import held_out_alarms, leak_probes, month_over_month_alarms

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


def test_a7_on_the_sealed_release_matches_my_original_month_over_month_psi(warehouse_run):
    # My original gate printed each feature's PSI against the month before. A7 reads the
    # same numbers. Only active days is read per day. Position's 0.4224 sets it off.
    rows = month_over_month_alarms(warehouse_run.frames, warehouse_run.ladder)
    sealed = rows[-1]
    original = REFERENCE["sealed_gate"]["psi_against_previous_month"]
    for feature in ["log_impressions", "position", "ctr", "momentum"]:
        assert sealed["psi_against_previous_month"][feature] == original[feature]
    assert sealed["A7_drifted_since_last_month"]
    assert len(rows) == 4
