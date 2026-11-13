"""The outcome: what each page did in the month after the feature month.

This is the only module that reads a month after the decision moment.
"""

import pandas as pd

from bleedtracker.config import DECLINE_DROP
from bleedtracker.warehouse import Source, query

OUTCOME_COLUMNS = ["imp_next", "seen_next_month", "label_declining"]


def next_month_sql(source: Source, month: str) -> str:
    """SQL for every page's impressions in the label month.

    No slice rules here. A page that stops appearing is an outcome. The lookup keeps
    every page with search data available.
    """
    return f"""
        SELECT f.content_hash_id,
               CAST(SUM(f.gsc_impressions) AS DOUBLE) AS imp_next
        FROM {source.fact(month)} f
        WHERE f.gsc_data_available IS TRUE
        GROUP BY f.content_hash_id
    """


def load_next_month(con, source: Source, month: str) -> pd.DataFrame:
    """Read every page's impressions in the label month."""
    return query(con, next_month_sql(source, month))


def attach_label(frame: pd.DataFrame, next_month: pd.DataFrame) -> pd.DataFrame:
    """Join the label month onto a feature frame.

    A page is declining when its label-month impressions are more than DECLINE_DROP
    below its feature-month impressions. A page missing from the label month counts
    as zero impressions.
    """
    labeled = frame.merge(next_month, on="content_hash_id", how="left")
    if len(labeled) != len(frame):
        raise ValueError("The label join changed the number of rows.")
    labeled["seen_next_month"] = labeled.imp_next.notna().astype(int)
    labeled["imp_next"] = labeled.imp_next.fillna(0.0)
    threshold = (1 - DECLINE_DROP) * labeled.imp
    labeled["label_declining"] = (labeled.imp_next < threshold).astype(int)
    return labeled


def without_outcome(frame: pd.DataFrame) -> pd.DataFrame:
    """The same frame with every outcome column taken out."""
    return frame.drop(columns=[c for c in OUTCOME_COLUMNS if c in frame.columns])
