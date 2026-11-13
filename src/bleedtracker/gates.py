"""The gates and the playbook.

Gates run on candidates before any slot is handed out. They never change a score.
A page that was rising, or that had no first-half traffic, never gets a slot.
A page whose month rests on one spike day, or that was live on too few days,
keeps its slot and gets "verify first" as its action.
"""

import pandas as pd

RISER_GATE = 1.20
SPIKE_GATE = 0.25
COVERAGE_DAYS = 25
ORIGINAL_COVERAGE_MONTH = 31

PLAYBOOK = pd.DataFrame(
    [
        (
            "slipping_and_underclicking",
            "rule",
            "A visible page that is losing clicks it already earns.",
            "rewrite_title_and_refresh",
            "A rewrite can't fix a page that answers the wrong question. "
            "Check the search queries first.",
        ),
        (
            "slipping_reachable_page",
            "rule",
            "A page an edit could still lift, sliding inside the month.",
            "refresh_now",
            "A slide can be seasonal. Check last year and the client's calendar.",
        ),
        (
            "slipping_deep_page",
            "rule",
            "A page past position 20, sliding, unlikely to come back with an edit.",
            "expand_or_consolidate",
            "Merging can't be undone. Never merge without a person who owns the page.",
        ),
        (
            "quiet_risk_model_only",
            "model",
            "A flat page the rule can't score that the forest ranks high.",
            "investigate_quiet_risk",
            "The forest gives no reason. A person supplies one or the item is dropped.",
        ),
        (
            "measurement_suspicion",
            "gate",
            "One day, or a short live window, carries the month.",
            "verify_then_review",
            "Check the tracking and any campaign before touching the content.",
        ),
        (
            "durable_riser",
            "gate",
            "A page that was rising when the month closed.",
            "monitor_only",
            "Never acted on. Rising pages declined less often than the base rate.",
        ),
        (
            "new_or_no_first_half_traffic",
            "gate",
            "A page with no first-half traffic. Its momentum is filled in.",
            "monitor_only",
            "Never acted on. Its slide signal is made up.",
        ),
        (
            "holding_steady",
            "rule",
            "A page that isn't sliding and that the shadow queue didn't pick.",
            "monitor",
            "The queue isn't a to-do list for every page.",
        ),
    ],
    columns=["reason_code", "supplied_by", "archetype", "action", "guardrail"],
)
ACTIONS = dict(zip(PLAYBOOK.reason_code, PLAYBOOK.action, strict=True))


def apply_gates(
    pool: pd.DataFrame,
    days_in_month: int,
    original_coverage: bool = True,
    riser_gate: float = RISER_GATE,
):
    """Add the gate flags and the `eligible` column to a scored pool.

    My original analysis failed a page on coverage when its live days, as a share of
    its own month, were under 25/31. `original_coverage` keeps that. Without it, a
    page fails when it was live on fewer than 25 days. A page whose momentum reaches
    `riser_gate` was rising.
    """
    d = pool.copy()
    d["active_share"] = d.active_days / days_in_month
    if original_coverage:
        low_coverage = d.active_share < COVERAGE_DAYS / ORIGINAL_COVERAGE_MONTH
    else:
        low_coverage = d.active_days < COVERAGE_DAYS
    d["gate_no_first_half"] = (d.has_first_half_traffic == 0).astype(int)
    d["gate_riser"] = ((d.momentum >= riser_gate) & (d.gate_no_first_half == 0)).astype(int)
    d["gate_verify"] = ((d.spike_share >= SPIKE_GATE) | low_coverage).astype(int)
    d["eligible"] = ((d.gate_no_first_half == 0) & (d.gate_riser == 0)).astype(int)
    return d
