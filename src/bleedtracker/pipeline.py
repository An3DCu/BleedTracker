"""Steps that read the warehouse and put the pieces together."""

import pandas as pd

from bleedtracker.features import build_features, load_page_months
from bleedtracker.label import attach_label, load_next_month
from bleedtracker.model import Deployment, deploy, split_clients
from bleedtracker.warehouse import Source


def labeled_month(
    con, source: Source, feature_month: str, label_month: str, original_momentum: bool = True
) -> pd.DataFrame:
    """One feature month with its label joined from the next month."""
    frame = build_features(load_page_months(con, source, feature_month), original_momentum)
    return attach_label(frame, load_next_month(con, source, label_month))


def anchor_deployment(frame: pd.DataFrame) -> tuple[Deployment, set, set]:
    """The first deployment. It splits one month's clients and trains on 70% of them.

    The other 30% are judged.
    """
    train_clients, test_clients = split_clients(frame)
    train = frame[frame.client_hash_id.isin(train_clients)]
    test = frame[frame.client_hash_id.isin(test_clients)]
    return deploy(train, test), train_clients, test_clients
