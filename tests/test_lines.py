"""Per-day features, the slide line and riser gate as settings, forest v2 and the pick."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bleedtracker.config import FEATURES_V2
from bleedtracker.features import per_day
from bleedtracker.gates import apply_gates
from bleedtracker.lines import (
    RISER_GATES,
    SLIDE_LINES,
    lines_report,
    pair_score,
    pick,
    score_grid,
    score_riser_gates,
    shadow_precision,
)
from bleedtracker.model import deploy
from bleedtracker.rule import V2_RISER_GATE, V2_SLIDE_LINE, signals


def halves(early, late, active_days=31, flag=1) -> dict:
    return {
        "imp_early": early,
        "imp_late": late,
        "active_days": active_days,
        "has_first_half_traffic": flag,
    }


def test_per_day_in_a_31_day_month():
    # March splits 16/15. 160 impressions over 16 days and 150 over 15 are both
    # 10 a day. The page is flat.
    f = per_day(pd.DataFrame([halves(160.0, 150.0)]), "2026-03")
    assert f.momentum[0] == pytest.approx(1.0)
    assert f.active_share[0] == pytest.approx(1.0)


def test_per_day_in_a_30_day_month():
    # April splits 15/15. 150 then 75 impressions is half as many a day.
    f = per_day(pd.DataFrame([halves(150.0, 75.0, active_days=15)]), "2026-04")
    assert f.momentum[0] == pytest.approx(0.5)
    assert f.active_share[0] == pytest.approx(0.5)


def test_a_page_that_vanished_mid_month_scores_zero():
    f = per_day(pd.DataFrame([halves(160.0, np.nan)]), "2026-03")
    assert f.momentum[0] == 0.0
    assert f.has_first_half_traffic[0] == 1


def test_a_page_with_no_first_half_keeps_momentum_1():
    f = per_day(pd.DataFrame([halves(np.nan, 100.0, flag=0)]), "2026-03")
    assert f.momentum[0] == 1.0


def test_per_day_refuses_the_original_first_half_flag():
    # My original code flagged a vanished page as having no first half.
    with pytest.raises(ValueError):
        per_day(pd.DataFrame([halves(160.0, np.nan, flag=0)]), "2026-03")


def test_the_slide_line_is_a_setting():
    page = pd.DataFrame({"momentum": [0.6], "position": [5.0], "ctr": [0.01]})
    medians = pd.Series({"3 to 10": 0.02})
    # severity = 1 - momentum / slide line
    assert signals(page, medians).severity[0] == pytest.approx(0.25)
    assert signals(page, medians, slide_line=0.75).severity[0] == pytest.approx(0.2)
    assert signals(page, medians, slide_line=0.6).severity[0] == 0.0


def test_the_riser_gate_is_a_setting():
    pool = pd.DataFrame(
        {
            "momentum": [1.3],
            "has_first_half_traffic": [1],
            "active_days": [31],
            "spike_share": [0.1],
        }
    )
    assert apply_gates(pool, 31).gate_riser[0] == 1
    assert apply_gates(pool, 31, riser_gate=1.35).gate_riser[0] == 0


def test_forest_v2_trains_on_the_feature_list_it_is_given():
    rng = np.random.default_rng(0)
    n = 1200
    momentum = rng.uniform(0, 2, n)
    frame = pd.DataFrame(
        {
            "imp": rng.integers(100, 5000, n).astype(float),
            "active_share": rng.uniform(0, 1, n),
            "position": rng.uniform(1, 40, n),
            "ctr": rng.uniform(0, 0.05, n),
            "momentum": momentum,
            "label_declining": (momentum < 0.8).astype(int),
        }
    ).assign(log_impressions=lambda f: np.log1p(f.imp))
    d = deploy(frame.iloc[:800], frame.iloc[800:], features=FEATURES_V2, slide_line=0.7)
    assert list(d.forest.feature_names_in_) == FEATURES_V2
    assert d.test.forest_score.between(0, 1).all()


def test_the_grid_has_80_pairs_around_the_original_lines():
    assert len(SLIDE_LINES) == 8 and SLIDE_LINES[0] == 0.6 and SLIDE_LINES[-1] == 0.95
    assert len(RISER_GATES) == 10 and RISER_GATES[0] == 1.05 and RISER_GATES[-1] == 1.5
    assert 0.8 in SLIDE_LINES and 1.2 in RISER_GATES


def test_a_tie_goes_to_the_pair_closest_to_the_original_lines():
    scores = [
        {"slide_line": 0.6, "riser_gate": 1.05, "score": 0.8},
        {"slide_line": 0.85, "riser_gate": 1.25, "score": 0.9},
        {"slide_line": 0.75, "riser_gate": 1.2, "score": 0.9},
    ]
    # |0.85 - 0.8| + |1.25 - 1.2| = 0.10 and |0.75 - 0.8| + |1.2 - 1.2| = 0.05.
    assert pick(scores) == scores[2]


def month(pages: int, sliding: int, decliners_among_flat: int, client: str, early_days: int):
    # Every page has 10 impressions a day in the first half. A sliding page has 4 a
    # day in the second half (momentum 0.4). A flat page keeps 10 a day.
    late_days = early_days - 1 if early_days == 16 else early_days
    slide = np.arange(pages) < sliding
    return pd.DataFrame(
        {
            "client_hash_id": client,
            "imp_early": 10.0 * early_days,
            "imp_late": np.where(slide, 4.0, 10.0) * late_days,
            "imp": 1000.0,
            "log_impressions": np.log1p(1000.0),
            "active_days": 30,
            "has_first_half_traffic": 1,
            "position": 5.0,
            "ctr": 0.01,
            "spike_share": 0.1,
            "label_declining": (slide | (np.arange(pages) >= pages - decliners_among_flat)).astype(
                int
            ),
        }
    )


def test_pair_score_worked_by_hand():
    # 100 sliding pages, all declining and tied. 200 flat pages score zero, none declining.
    # Precision at K = 50, 100, 150, 200, 250 is 1, 1, 100/150, 100/200, 100/250.
    frame = per_day(month(300, 100, 0, "a", 16), "2026-03")
    expected = (1 + 1 + 100 / 150 + 100 / 200 + 100 / 250) / 5
    assert pair_score(frame, "2026-03", 0.8, 1.2) == pytest.approx(expected)


def test_each_month_pair_counts_equally_and_held_out_clients_stay_out():
    march = month(300, 100, 0, "a", 16)
    # In April every training page declines. A held-out client's pages would pull the
    # score down if the pick saw them.
    april = pd.concat(
        [month(600, 600, 0, "a", 15), month(300, 300, 0, "held_out", 15).assign(label_declining=0)]
    )
    frames = {("2026-03", "2026-04"): march, ("2026-04", "2026-05"): april}
    scores = score_grid(frames, {"a"})
    march_score = (1 + 1 + 100 / 150 + 100 / 200 + 100 / 250) / 5
    assert {row["score"] for row in scores} == {round((march_score + 1.0) / 2, 4)}
    # Every pair ties. The original lines win.
    assert (pick(scores)["slide_line"], pick(scores)["riser_gate"]) == (0.8, 1.2)


def shadow_pool() -> pd.DataFrame:
    # 35 pages the rule scores zero. Five risers at momentum 1.3 have the highest forest
    # scores and all declined. Of the 30 flat pages, the four with the highest forest
    # scores declined.
    risers = pd.DataFrame(
        {"momentum": 1.3, "forest_score": np.linspace(0.99, 0.95, 5), "label_declining": 1}
    )
    flat = pd.DataFrame(
        {
            "momentum": 1.0,
            "forest_score": np.linspace(0.9, 0.1, 30),
            "label_declining": (np.arange(30) < 4).astype(int),
        }
    )
    return pd.concat([risers, flat], ignore_index=True).assign(
        action_score=0.0,
        reason_code="holding_steady",
        imp=1000.0,
        has_first_half_traffic=1,
        active_days=31,
        spike_share=0.1,
    )


def test_shadow_precision_worked_by_hand():
    # At 1.20 the risers are gated out. The 10 shadow items are the top 10 flat pages.
    # 4 of them declined. At 1.35 the risers stay in. The 10 items are the 5 risers and
    # the top 5 flat pages. 9 of them declined.
    assert shadow_precision(shadow_pool(), "2026-03", 1.2) == pytest.approx(0.4)
    assert shadow_precision(shadow_pool(), "2026-03", 1.35) == pytest.approx(0.9)


def test_a_riser_gate_tie_goes_to_the_gate_closest_to_1_20():
    rows = [{"slide_line": 0.65, "riser_gate": g, "score": 0.7} for g in (1.05, 1.15, 1.3)]
    assert pick(rows)["riser_gate"] == 1.15


def test_step_2_scores_every_riser_gate():
    frames = {("2026-03", "2026-04"): month(1200, 400, 100, "a", 16)}
    rows = score_riser_gates(frames, {"a"}, slide_line=0.65)
    assert [row["riser_gate"] for row in rows] == RISER_GATES
    assert all(0 <= row["score"] <= 1 for row in rows)


def test_the_report_holds_counts_and_no_client_names():
    frames = {("2026-03", "2026-04"): month(300, 100, 0, "client_secret_name", 16)}
    slide_scores = score_grid(frames, {"client_secret_name"})
    riser_scores = [{"slide_line": 0.8, "riser_gate": g, "score": 0.5} for g in RISER_GATES]
    report = lines_report(slide_scores, riser_scores, frames, {"client_secret_name"})
    assert report["training_pages"] == {"2026-03 to 2026-04": 300}
    assert len(report["step_1_grid"]) == 80
    assert report["picked"] == {"slide_line": 0.8, "riser_gate": 1.2}
    assert "client_secret_name" not in json.dumps(report)


def test_the_pinned_lines_match_the_committed_pick():
    report = json.loads(
        (Path(__file__).parents[1] / "reports" / "v2_lines.json").read_text("utf-8")
    )
    assert report["picked"] == {"slide_line": V2_SLIDE_LINE, "riser_gate": V2_RISER_GATE}
