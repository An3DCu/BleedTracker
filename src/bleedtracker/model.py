"""The client split, the forest, and one deployment of the rule and the forest side by side."""

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

from bleedtracker.config import FEATURES
from bleedtracker.rule import action_score, band_median_ctr, reason_code

SEED = 42
TEST_SHARE = 0.3
FOREST_PARAMS = {"n_estimators": 300, "min_samples_leaf": 400, "n_jobs": -1, "random_state": SEED}


def split_clients(frame: pd.DataFrame, seed: int = SEED) -> tuple[set, set]:
    """Split by client. Every client's pages land on one side only."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SHARE, random_state=seed)
    train_idx, test_idx = next(splitter.split(frame, groups=frame.client_hash_id))
    train_clients = set(frame.client_hash_id.iloc[train_idx])
    test_clients = set(frame.client_hash_id.iloc[test_idx])
    return train_clients, test_clients


def make_forest() -> RandomForestClassifier:
    """The forest picked by cross-validation on training clients in my original analysis."""
    return RandomForestClassifier(**FOREST_PARAMS)


@dataclass
class Deployment:
    """The rule and the forest, fitted on one frame and judged on another."""

    train: pd.DataFrame
    test: pd.DataFrame
    medians: pd.Series
    forest: RandomForestClassifier


def deploy(train: pd.DataFrame, test: pd.DataFrame) -> Deployment:
    """Fit the rule's band medians and the forest on `train`, then score `test`.

    Both frames need the features and the label. The test frame gets the rule's
    score and reason code and the forest's probability of a decline.
    """
    medians = band_median_ctr(train)
    forest = make_forest().fit(train[FEATURES], train.label_declining)
    scored = test.copy()
    scored["action_score"] = action_score(scored, medians)
    scored["reason_code"] = reason_code(scored, medians)
    scored["forest_score"] = forest.predict_proba(scored[FEATURES])[:, 1]
    return Deployment(train=train, test=scored, medians=medians, forest=forest)
