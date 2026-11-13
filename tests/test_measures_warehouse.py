"""The new measures on the warehouse deployments. They're checked against my original results.

Runs only with a token: pytest -m warehouse
"""

import pytest

from bleedtracker.deployments import plans
from bleedtracker.measures import paired_bootstrap, per_account_measures, queue_measures
from bleedtracker.reference import load_reference

pytestmark = pytest.mark.warehouse

REFERENCE = load_reference()


def test_the_bootstrap_starts_from_my_original_four_differences(warehouse_run):
    # The observed gap in each forward deployment is the forest minus the rule at 50.
    forward = [plan.name for plan in plans()[1:]]
    observed = [
        paired_bootstrap(warehouse_run.ladder.queues[name][1], draws=50)["forest_minus_rule_points"]
        for name in forward
    ]
    assert observed == REFERENCE["uncertainty"]["forest_minus_rule_points"]


def test_the_sealed_queue_holds_my_original_share_of_lost_impressions(warehouse_run):
    queue, pool = warehouse_run.ladder.queues[plans()[-1].name]
    editor = queue[queue.slot_type == "editor_slot"]
    measured = queue_measures(editor, pool)
    assert (
        measured["lost_impressions_share"] == REFERENCE["limitations"]["impressions_share_covered"]
    )
    assert measured["margin_points"] == warehouse_run.ladder.rows[-1]["margin_points"]


def test_the_per_account_queue_runs_in_every_deployment(warehouse_run):
    for _, pool in warehouse_run.ladder.queues.values():
        rule = per_account_measures(pool, "action_score", eligible_only=True)
        forest = per_account_measures(pool, "forest_score", eligible_only=False)
        assert sum(rule["slots_per_client"]) == 50
        assert min(rule["slots_per_client"]) >= 1
        assert 0 <= forest["precision"] <= 1
