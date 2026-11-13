"""The first deployment rebuilt from the warehouse and checked against my original results.

Runs only with a token: pytest -m warehouse
"""

import pytest
from sklearn.metrics import roc_auc_score

from bleedtracker.metrics import precision_at_k
from bleedtracker.pipeline import anchor_deployment, labelled_month
from bleedtracker.reference import load_reference
from bleedtracker.rule import reason_code
from bleedtracker.warehouse import MissingTokenError, Source, connect, read_token

pytestmark = pytest.mark.warehouse

REFERENCE = load_reference()["anchor"]


@pytest.fixture(scope="module")
def anchor():
    try:
        token = read_token()
    except MissingTokenError:
        pytest.skip("No Hugging Face token.")
    con = connect(token)
    frame = labelled_month(con, Source(), REFERENCE["feature_month"], REFERENCE["label_month"])
    deployment, train_clients, test_clients = anchor_deployment(frame)
    return frame, deployment, train_clients, test_clients


def rate(x: float) -> float:
    return round(float(x), 4)


def test_slice(anchor):
    frame, *_ = anchor
    assert len(frame) == REFERENCE["slice_pages"]
    assert frame.client_hash_id.nunique() == REFERENCE["slice_clients"]
    assert rate(frame.label_declining.mean()) == REFERENCE["slice_base_rate"]


def test_split(anchor):
    _, deployment, train_clients, test_clients = anchor
    assert len(train_clients) == REFERENCE["train_clients"]
    assert len(test_clients) == REFERENCE["test_clients"]
    assert len(deployment.train) == REFERENCE["train_pages"]
    assert len(deployment.test) == REFERENCE["test_pages"]
    assert rate(deployment.test.label_declining.mean()) == REFERENCE["test_base_rate"]


def test_rule_precision(anchor):
    test = anchor[1].test
    assert (
        rate(precision_at_k(test.action_score, test.label_declining))
        == REFERENCE["rule_precision_at_50"]
    )


def test_rule_precision_curve(anchor):
    test = anchor[1].test
    for k, expected in REFERENCE["rule_precision_at_k"].items():
        got = rate(precision_at_k(test.action_score, test.label_declining, int(k)))
        assert got == expected, f"K={k}"


def test_forest(anchor):
    test = anchor[1].test
    assert (
        rate(precision_at_k(test.forest_score, test.label_declining))
        == REFERENCE["forest_precision_at_50"]
    )
    assert rate(roc_auc_score(test.label_declining, test.forest_score)) == REFERENCE["forest_auc"]


def test_pages_the_rule_scores_zero(anchor):
    test = anchor[1].test
    zero = test[test.action_score == 0]
    assert len(zero) == REFERENCE["rule_zero_pages"]
    assert int(zero.label_declining.sum()) == REFERENCE["rule_zero_declines"]


def test_band_medians(anchor):
    medians = anchor[1].medians
    got = {band: rate(100 * value) for band, value in medians.items()}
    assert got == REFERENCE["band_median_ctr_percent"]


def test_reason_codes_in_the_slice(anchor):
    frame, deployment, *_ = anchor
    counts = reason_code(frame, deployment.medians).value_counts().to_dict()
    assert counts == REFERENCE["reason_codes_in_slice"]
