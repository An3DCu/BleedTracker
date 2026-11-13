import numpy as np
import pandas as pd
import pytest

from bleedtracker.config import FEATURES
from bleedtracker.model import deploy, split_clients


def synthetic_frame(clients: int = 10, pages_per_client: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = clients * pages_per_client
    momentum = rng.uniform(0, 2, n)
    return pd.DataFrame(
        {
            "client_hash_id": np.repeat([f"c{i}" for i in range(clients)], pages_per_client),
            "imp": rng.integers(100, 5000, n).astype(float),
            "log_imp_mar": 0.0,
            "active_days": rng.integers(1, 32, n),
            "pos_mar": rng.uniform(1, 40, n),
            "ctr_mar": rng.uniform(0, 0.05, n),
            "momentum_in_month": momentum,
            "label_declining": (momentum < 0.8).astype(int),
        }
    ).assign(log_imp_mar=lambda f: np.log1p(f.imp))


def test_every_client_lands_on_one_side():
    frame = synthetic_frame()
    train, test = split_clients(frame)
    assert not train & test
    assert train | test == set(frame.client_hash_id)
    # 30% of 10 clients, rounded up.
    assert len(test) == 3


def test_split_ignores_row_order():
    frame = synthetic_frame()
    assert split_clients(frame) == split_clients(frame.iloc[::-1])


def test_deploy_scores_the_test_frame():
    frame = synthetic_frame()
    train_clients, _ = split_clients(frame)
    train = frame[frame.client_hash_id.isin(train_clients)]
    test = frame[~frame.client_hash_id.isin(train_clients)]
    result = deploy(train, test)
    assert len(result.test) == len(test)
    assert {"action_score", "reason_code", "forest_score"} <= set(result.test.columns)
    assert result.test.forest_score.between(0, 1).all()
    assert "action_score" not in test.columns


def test_band_medians_come_from_training_rows_only():
    frame = synthetic_frame()
    train = frame.iloc[:200].assign(pos_mar=2.0, ctr_mar=0.01)
    test = frame.iloc[200:].assign(pos_mar=2.0, ctr_mar=0.9)
    result = deploy(train, test)
    assert result.medians.to_dict() == pytest.approx({"1-3": 0.01})


def test_forest_sees_only_the_five_features():
    frame = synthetic_frame()
    result = deploy(frame.iloc[:300], frame.iloc[300:])
    assert list(result.forest.feature_names_in_) == FEATURES
