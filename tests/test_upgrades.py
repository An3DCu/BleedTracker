"""The upgrades command's pieces."""

import json
import re

import numpy as np
import pandas as pd

from bleedtracker.upgrades import (
    UPGRADES_FILE,
    check_originals,
    empty_second_half_pages,
    load_upgrades,
    verify_first,
)


def test_check_originals_skips_the_environment():
    committed = {"a": 1.0, "environment": {"python": "3.14.7"}}
    rebuilt = {"a": 1.0, "environment": {"python": "3.14.9"}}
    comparison = check_originals(rebuilt, committed)
    assert comparison.all_match
    assert comparison.numbers == 1


def test_check_originals_finds_a_difference():
    comparison = check_originals({"a": 1.5}, {"a": 1.0})
    assert not comparison.all_match


def test_empty_second_half_pages_counts_by_month():
    # One page stopped before the second half. One kept going. One had no first half.
    month = pd.DataFrame({"has_first_half_traffic": [1, 1, 0], "imp_late": [np.nan, 50.0, 80.0]})
    frames = {("2026-01", "2026-02"): month}
    sealed = month.iloc[[0, 0, 1]]
    counts = empty_second_half_pages(frames, sealed)
    assert counts == {"2026-01": 1, "2026-05": 2}


def test_verify_first_counts_editor_slots_only():
    queue = pd.DataFrame(
        {
            "slot_type": ["editor_slot", "editor_slot", "shadow"],
            "reason_code": ["measurement_suspicion", "holding_steady", "measurement_suspicion"],
        }
    )
    assert verify_first(queue) == 1


def test_every_section_of_the_committed_file_is_post_hoc():
    upgrades = load_upgrades()
    sections = [upgrades["fixes"], upgrades["sealed_gate_v2"], upgrades["a7"]]
    sections += upgrades["deployments"]
    assert all(section["post_hoc"] for section in sections)


def test_the_margin_comes_first_in_every_queue():
    for entry in load_upgrades()["deployments"]:
        for version in ["v1", "v2"]:
            for queue in ["rule", "forest", "per_account_rule", "per_account_forest"]:
                assert list(entry[version][queue])[0] == "margin_points"


def test_only_v2_on_the_open_months_is_in_sample():
    for entry in load_upgrades()["deployments"]:
        assert entry["in_sample"]["v1"] is False
        assert entry["in_sample"]["v2"] == (not entry["deployment"].startswith("sealed"))


def test_the_committed_file_carries_no_ids():
    text = UPGRADES_FILE.read_text(encoding="utf-8")
    assert not re.search(r"(client|content)_[0-9a-f]{6,}|[0-9a-f]{32,}", text)
    assert json.loads(text)["originals_checked"]["numbers"] > 0
