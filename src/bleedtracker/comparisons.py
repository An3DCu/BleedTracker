"""The rule and the forest against simpler rules, and what each feature is worth."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from bleedtracker.config import FEATURES
from bleedtracker.metrics import TOP_K, precision_at_k
from bleedtracker.model import SEED, make_forest

K_GRID = [10, 25, 50, 75, 100, 150, 200, 250]
RANDOM_DRAWS = 200
PER_CLIENT = 5


def scorer_columns(pool: pd.DataFrame) -> pd.DataFrame:
    """Add a score column for every simple rule the queue is compared with."""
    s = pool.copy()
    s["steepest_slide"] = -s.momentum
    s["lowest_ctr"] = -s.ctr
    s["biggest_pages"] = s.log_impressions
    is_ctr_fix = (s.imp >= 500) & s.position.between(1, 20) & (s.ctr < 0.005)
    s["ctr_fix_flag"] = is_ctr_fix.astype(int) * s.log_impressions
    return s


SCORERS = {
    "rule": "action_score",
    "forest": "forest_score",
    "steepest_slide": "steepest_slide",
    "lowest_ctr": "lowest_ctr",
    "biggest_pages": "biggest_pages",
    "ctr_fix_flag": "ctr_fix_flag",
}


def random_draws(pool: pd.DataFrame, seed: int = SEED) -> np.ndarray:
    """Precision of 200 random queues of 50 pages."""
    rng = np.random.default_rng(seed)
    labels = pool.label_declining.to_numpy()
    return np.array(
        [labels[rng.choice(len(pool), TOP_K, replace=False)].mean() for _ in range(RANDOM_DRAWS)]
    )


def sealed_comparison(pool: pd.DataFrame) -> list[dict]:
    """Every scorer's top 50 in one pool, then 200 random queues."""
    s = scorer_columns(pool)
    rows = []
    for name, column in SCORERS.items():
        top = s.nlargest(TOP_K, column)
        rows.append(
            {
                "scorer": name,
                "precision_at_50": round(float(top.label_declining.mean()), 4),
                "auc": round(float(roc_auc_score(s.label_declining, s[column])), 4),
                "impressions_lost_by_its_50": int((top.imp - top.imp_next).clip(lower=0).sum()),
                "median_page_size": int(top.imp.median()),
            }
        )
    draws = random_draws(s)
    rows.append(
        {
            "scorer": "random_50",
            "precision_at_50": round(float(draws.mean()), 4),
            "auc": 0.5,
            "impressions_lost_by_its_50": None,
            "median_page_size": None,
            "percentile_95": round(float(np.percentile(draws, 95)), 4),
        }
    )
    return rows


def precision_curve(pool: pd.DataFrame) -> list[dict]:
    """Tie-fair precision of the rule and the forest from K = 10 to 250.

    Rounded with NumPy, as the original table was.
    """
    base = pool.label_declining.mean()
    rows = []
    for k in K_GRID:
        rule = precision_at_k(pool.action_score, pool.label_declining, k)
        forest = precision_at_k(pool.forest_score, pool.label_declining, k)
        rows.append(
            {
                "K": k,
                "rule": float(np.round(rule, 4)),
                "forest": float(np.round(forest, 4)),
                "base_rate": float(np.round(base, 4)),
                "rule_margin_points": float(np.round((rule - base) * 100, 1)),
            }
        )
    return rows


def per_client_queue(pool: pd.DataFrame) -> dict:
    """Five pages per held-out client, averaged over clients with at least five pages."""
    rule, forest = [], []
    for _, group in pool.groupby("client_hash_id"):
        if len(group) < PER_CLIENT:
            continue
        rule.append(group.nlargest(PER_CLIENT, "action_score").label_declining.mean())
        forest.append(group.nlargest(PER_CLIENT, "forest_score").label_declining.mean())
    return {
        "rule": round(float(np.mean(rule)), 4),
        "forest": round(float(np.mean(forest)), 4),
        "clients": len(rule),
    }


def ablation(train: pd.DataFrame, test: pd.DataFrame, forest_scores) -> dict:
    """The forest's AUC with each feature left out, and with all five."""
    labels = test.label_declining
    result = {}
    for feature in FEATURES:
        keep = [f for f in FEATURES if f != feature]
        forest = make_forest().fit(train[keep], train.label_declining)
        scores = forest.predict_proba(test[keep])[:, 1]
        result[f"without_{feature}"] = round(float(roc_auc_score(labels, scores)), 4)
    result["all_five"] = round(float(roc_auc_score(labels, forest_scores)), 4)
    return result


def momentum_alone(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """The forest on momentum only."""
    forest = make_forest().fit(train[["momentum"]], train.label_declining)
    scores = forest.predict_proba(test[["momentum"]])[:, 1]
    return {
        "auc": round(float(roc_auc_score(test.label_declining, scores)), 4),
        "precision_at_50": round(precision_at_k(scores, test.label_declining), 4),
    }
