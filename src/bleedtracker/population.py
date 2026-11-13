"""What the data holds, who the results describe and the numbers behind each limitation.

Everything read from the warehouse here is a count, a share or a date. The one
page-level table, the biggest bleeds, keeps only momentum and the two ranks.
"""

import numpy as np
import pandas as pd

from bleedtracker.config import MIN_HISTORY, RELEASE
from bleedtracker.metrics import TOP_K
from bleedtracker.warehouse import Source, query

SNAPSHOT_DATE = "2026-07-03"
DECISION_DATE = "2026-05-31"
STALE_DAYS = 180
STALE_MIN_IMPRESSIONS = 500
DATE_KEYS = {"first_day", "last_day"}

TABLES = {
    "dim_clients": "dim_clients.parquet",
    "dim_content": "dim_content.parquet",
    "fact_content_query_90d": "fact_content_query_90d.parquet",
    "fact_content_daily_performance": "fact_content_daily_performance/**/*.parquet",
    "fact_content_daily_performance_sample": "fact_content_daily_performance_sample.parquet",
}


def count_rows(con, source: Source, path: str) -> int:
    """Rows in a table. DuckDB reads the count from the file footers."""
    return int(query(con, f"SELECT COUNT(*) AS n FROM {source.table(path)}").n.iloc[0])


def data_facts(con, source: Source, month: str, sealed_features: pd.DataFrame) -> dict:
    """The release's size, the grain of the daily table and a profile of one month."""
    sizes = {name: count_rows(con, source, path) for name, path in TABLES.items()}
    grain = query(
        con,
        f"""SELECT report_date, client_hash_id, content_hash_id, COUNT(*) AS n
            FROM {source.fact(month)} GROUP BY 1, 2, 3 HAVING COUNT(*) > 1 LIMIT 5""",
    )
    profile = query(
        con,
        f"""SELECT COUNT(*) AS daily_rows,
                   MIN(report_date) AS first_day,
                   MAX(report_date) AS last_day,
                   COUNT(DISTINCT client_hash_id) AS clients,
                   COUNT(DISTINCT content_hash_id) AS pages,
                   AVG(CASE WHEN gsc_data_available IS TRUE THEN 1 ELSE 0 END) AS gsc_true_share,
                   AVG(CASE WHEN ga4_data_available IS TRUE THEN 1 ELSE 0 END) AS ga4_true_share,
                   AVG(CASE WHEN ga4_data_available IS NULL THEN 1 ELSE 0 END) AS ga4_null_share
            FROM {source.fact(month)}""",
    ).iloc[0]
    clients = query(
        con,
        f"""SELECT COUNT(*) AS total,
                   COUNT(*) FILTER (gsc_data_start <= DATE '{MIN_HISTORY.isoformat()}') AS eligible
            FROM {source.dim_clients()}""",
    ).iloc[0]
    page_rows = query(
        con,
        f"""SELECT COUNT(*) AS n FROM (
                SELECT client_hash_id, content_hash_id FROM {source.fact(month)}
                WHERE gsc_data_available IS TRUE GROUP BY 1, 2)""",
    ).n.iloc[0]
    return {
        "release": f"{RELEASE} (build v20260703)",
        "grain_probe_duplicate_rows": len(grain),
        "feature_month_profile": {
            key: round(float(value), 4) if key not in DATE_KEYS else str(pd.Timestamp(value))
            for key, value in profile.items()
        },
        "clients_total": int(clients.total),
        "clients_eligible": int(clients.eligible),
        "clients_in_sealed_slice": int(sealed_features.client_hash_id.nunique()),
        "page_rows_in_feature_month": int(page_rows),
        "pages_scored_sealed": len(sealed_features),
        "release_row_counts": sizes,
        "release_star_schema_rows": sum(v for k, v in sizes.items() if not k.endswith("_sample")),
    }


def client_shares(frame: pd.DataFrame) -> pd.Series:
    """Each client's share of the pages, largest first."""
    return frame.client_hash_id.value_counts(normalize=True)


def snapshot_audit(con, source: Source, features: pd.DataFrame) -> dict:
    """How many pages carry an update date from after the decision moment.

    The content table is a snapshot taken after the panel ends. It's read here for this audit
    only. It's never a feature.
    """
    con.register("audit_pages", features[["content_hash_id", "imp"]])
    j = query(
        con,
        f"""SELECT p.imp, c.content_updated_date, c.last_optimized_date
            FROM audit_pages p
            LEFT JOIN {source.table("dim_content.parquet")} c USING (content_hash_id)""",
    )
    con.unregister("audit_pages")
    decision = pd.Timestamp(DECISION_DATE)
    days_since = (decision - pd.to_datetime(j.content_updated_date)).dt.days
    after = days_since < 0
    optimized_after = pd.to_datetime(j.last_optimized_date) > decision
    stale = (days_since >= STALE_DAYS) & (j.imp >= STALE_MIN_IMPRESSIONS)
    return {
        "update_date_after_decision_share": round(float(after.mean()), 4),
        "optimized_date_after_decision_share": round(float(optimized_after.mean()), 4),
        "stale_flag_fires_on_pages": int(stale.sum()),
    }


def population_audit(sealed_features: pd.DataFrame, test_clients: set) -> dict:
    """How concentrated the sealed month is, overall and in the held-out pool."""
    share = client_shares(sealed_features)
    present = set(sealed_features.client_hash_id) & test_clients
    held_out = client_shares(sealed_features[sealed_features.client_hash_id.isin(test_clients)])
    return {
        "largest_client_share_of_slice": round(float(share.iloc[0]), 4),
        "top3_client_share_of_slice": round(float(share.head(3).sum()), 4),
        "held_out_clients_present_in_sealed_month": len(present),
        "held_out_clients_expected": len(test_clients),
        "largest_client_share_of_held_out_pool": round(float(held_out.iloc[0]), 4),
    }


def decision_envelope(sealed_features: pd.DataFrame, open_frames: dict) -> dict:
    """The editor's capacity against the size of the portfolio."""
    return {
        "decision_date": DECISION_DATE,
        "scorable_pages": len(sealed_features),
        "clients": int(sealed_features.client_hash_id.nunique()),
        "impressions_in_feature_month": int(sealed_features.imp.sum()),
        "editor_capacity": TOP_K,
        "capacity_share_of_portfolio": round(TOP_K / len(sealed_features), 6),
        "base_rate_by_open_pair": {
            f"{a} to {b}": round(float(frame.label_declining.mean()), 4)
            for (a, b), frame in open_frames.items()
        },
    }


def month_table(frames: dict, test_clients: set) -> list[dict]:
    """How each month behaved, whatever the model did."""
    rows = []
    for (a, b), frame in frames.items():
        held_out = frame[frame.client_hash_id.isin(test_clients)]
        rows.append(
            {
                "pair": f"{a} to {b}",
                "pages": len(frame),
                "slice_base_rate": round(float(frame.label_declining.mean()), 4),
                "held_out_base_rate": round(float(held_out.label_declining.mean()), 4),
                "median_next_over_now": round(float((frame.imp_next / frame.imp).median()), 4),
                "share_of_pages_that_grew": round(float((frame.imp_next > frame.imp).mean()), 4),
            }
        )
    return rows


def triage_numbers(queue: pd.DataFrame, pool: pd.DataFrame) -> dict:
    """How much of the sealed month's decline the editor's 50 pages cover."""
    editor = queue[queue.slot_type == "editor_slot"]
    pool_lost = (pool.imp - pool.imp_next).clip(lower=0).sum()
    queue_lost = (editor.imp - editor.imp_next).clip(lower=0).sum()
    return {
        "sealed_declines": int(pool.label_declining.sum()),
        "queue_recall": round(float(editor.label_declining.sum() / pool.label_declining.sum()), 5),
        "impressions_lost_by_pool": int(pool_lost),
        "impressions_in_front_of_editor": int(queue_lost),
        "impressions_share_covered": round(float(queue_lost / pool_lost), 4),
    }


def biggest_bleeds(pool: pd.DataFrame, count: int = 5) -> list[dict]:
    """Where the rule and the forest ranked the pages that lost the most impressions.

    Only momentum and the two ranks are kept. No single page's impressions are.
    """
    rule_rank = pool.action_score.rank(ascending=False, method="first")
    forest_rank = pool.forest_score.rank(ascending=False, method="first")
    big = pool.assign(lost=pool.imp - pool.imp_next).nlargest(count, "lost")
    return [
        {
            "momentum": float(np.round(row.momentum, 3)),
            "rule_rank": int(rule_rank[i]),
            "forest_rank": int(forest_rank[i]),
        }
        for i, row in big.iterrows()
    ]
