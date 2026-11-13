"""One feature month: the slice of pages and their five features.

Everything here is measured inside the feature month. All of it is known on the month's last day.
"""

import numpy as np
import pandas as pd

from bleedtracker.config import FEATURES, MIN_HISTORY, MIN_IMPRESSIONS, second_half_start
from bleedtracker.warehouse import Source, query


def page_month_sql(source: Source, month: str) -> str:
    """SQL for one row per client and page in a month, inside the slice.

    The slice: rows with search data available, clients with history from before
    MIN_HISTORY, and pages with at least MIN_IMPRESSIONS impressions in the month.
    """
    mid = second_half_start(month).isoformat()
    return f"""
        SELECT f.client_hash_id,
               f.content_hash_id,
               CAST(SUM(f.gsc_impressions) AS DOUBLE)                                AS imp,
               CAST(SUM(f.gsc_clicks) AS DOUBLE)                                     AS clk,
               COUNT(*) FILTER (f.gsc_impressions > 0)                               AS active_days,
               CAST(SUM(f.gsc_sum_position) AS DOUBLE)                               AS sum_pos,
               CAST(MAX(f.gsc_impressions) AS DOUBLE)                                AS max_day_imp,
               CAST(SUM(f.gsc_impressions) FILTER (f.report_date >= DATE '{mid}') AS DOUBLE)
                                                                                     AS imp_late,
               CAST(SUM(f.gsc_impressions) FILTER (f.report_date <  DATE '{mid}') AS DOUBLE)
                                                                                     AS imp_early
        FROM {source.fact(month)} f
        JOIN {source.dim_clients()} c USING (client_hash_id)
        WHERE f.gsc_data_available IS TRUE
          AND c.gsc_data_start <= DATE '{MIN_HISTORY.isoformat()}'
        GROUP BY f.client_hash_id, f.content_hash_id
        HAVING SUM(f.gsc_impressions) >= {MIN_IMPRESSIONS}
        ORDER BY f.content_hash_id
    """


def load_page_months(con, source: Source, month: str) -> pd.DataFrame:
    """Read one month's slice from the warehouse."""
    pages = query(con, page_month_sql(source, month))
    if pages.content_hash_id.duplicated().any():
        raise ValueError(f"A page appears under two clients in {month}.")
    return pages


def build_features(pages: pd.DataFrame, original_momentum: bool = True) -> pd.DataFrame:
    """Add the five features and the two gate inputs to a month's slice.

    With `original_momentum`, a page with first-half impressions and no rows in the
    second half is scored as flat (momentum 1.0) and flagged as having no first-half
    traffic, as in my original analysis. Without it, that page gets momentum 0.0
    and keeps its first-half flag.
    """
    f = pages.copy()
    f["position"] = f.sum_pos / f.imp
    f["ctr"] = f.clk / f.imp
    f["log_impressions"] = np.log1p(f.imp)
    f["spike_share"] = f.max_day_imp / f.imp

    early = f.imp_early.where(f.imp_early > 0)
    if original_momentum:
        momentum = f.imp_late / early
        f["has_first_half_traffic"] = momentum.notna().astype(int)
    else:
        momentum = f.imp_late.fillna(0) / early
        f["has_first_half_traffic"] = early.notna().astype(int)
    f["momentum"] = momentum.fillna(1.0)

    if f[FEATURES].isna().any().any():
        raise ValueError("A feature has a missing value.")
    return f.reset_index(drop=True)
