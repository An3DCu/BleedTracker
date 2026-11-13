"""The leakage audit.

An answer can sneak into a model three ways. A feature can be computed from the
label. It can come from after the decision moment. Or it can carry another system's
decision. The checks here look for the first two in code. Two planted leaks show
that the audit would notice one.
"""

import pandas as pd
from sklearn.metrics import roc_auc_score

from bleedtracker.config import FEATURES
from bleedtracker.label import OUTCOME_COLUMNS
from bleedtracker.metrics import precision_at_k
from bleedtracker.model import make_forest
from bleedtracker.warehouse import Source, query

SNAPSHOT_COLUMNS = ["content_updated_date", "last_optimized_date", "is_deleted"]
PLANTED_LEAKS = {
    "with_next_month_impressions": "imp_next",
    "with_the_label": "label_declining",
}


def no_outcome_or_snapshot_features() -> bool:
    """No outcome column and no column from the content snapshot is a feature."""
    return not set(FEATURES) & (set(OUTCOME_COLUMNS) | set(SNAPSHOT_COLUMNS))


def month_bounds(con, source: Source, month: str) -> tuple:
    """First and last day in a month's partition."""
    bounds = query(
        con, f"SELECT MIN(report_date) AS first, MAX(report_date) AS last FROM {source.fact(month)}"
    )
    return pd.Timestamp(bounds["first"].iloc[0]), pd.Timestamp(bounds["last"].iloc[0])


def windows_do_not_overlap(con, source: Source, feature_month: str, label_month: str) -> bool:
    """The label month opens after the feature month closes."""
    _, feature_last = month_bounds(con, source, feature_month)
    label_first, _ = month_bounds(con, source, label_month)
    return label_first > feature_last


def leak_probe(train: pd.DataFrame, test: pd.DataFrame, forest_scores) -> dict:
    """Precision@50 and AUC for the forest on its five features, then with each planted leak.

    `forest_scores` are the forest's scores on `test` from the five features.
    Each leak refits the same forest with one outcome column added.
    """
    labels = test.label_declining
    result = {
        "five_features": {
            "precision_at_50": round(precision_at_k(forest_scores, labels), 4),
            "auc": round(float(roc_auc_score(labels, forest_scores)), 4),
        }
    }
    for name, column in PLANTED_LEAKS.items():
        features = [*FEATURES, column]
        forest = make_forest().fit(train[features], train.label_declining)
        scores = forest.predict_proba(test[features])[:, 1]
        result[name] = {
            "precision_at_50": round(precision_at_k(scores, labels), 4),
            "auc": round(float(roc_auc_score(labels, scores)), 4),
        }
    return result
