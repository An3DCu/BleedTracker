"""A tiny warehouse on disk, laid out like the release, with values worked out by hand."""

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from bleedtracker.warehouse import Source


def _write(con: duckdb.DuckDBPyConnection, rows: list[tuple], path: Path) -> None:
    df = pd.DataFrame(
        rows,
        columns=[
            "report_date",
            "client_hash_id",
            "content_hash_id",
            "gsc_data_available",
            "gsc_impressions",
            "gsc_clicks",
            "gsc_sum_position",
        ],
    )
    path.parent.mkdir(parents=True)
    con.register("rows_df", df)
    con.execute(
        f"""COPY (
              SELECT CAST(report_date AS DATE) AS report_date,
                     client_hash_id, content_hash_id,
                     CAST(gsc_data_available AS BOOLEAN) AS gsc_data_available,
                     CAST(gsc_impressions AS BIGINT) AS gsc_impressions,
                     CAST(gsc_clicks AS BIGINT) AS gsc_clicks,
                     CAST(gsc_sum_position AS BIGINT) AS gsc_sum_position
              FROM rows_df
            ) TO '{path.as_posix()}' (FORMAT parquet)"""
    )
    con.unregister("rows_df")


def march_rows() -> list[tuple]:
    rows = []
    # p_steady: 10 impressions, 1 click and position 2 on every day of March.
    for day in range(1, 32):
        rows.append((f"2026-03-{day:02d}", "c1", "p_steady", True, 10, 1, 20))
    # p_vanish: rows on the 5th and 6th only, then nothing in the second half.
    rows.append(("2026-03-05", "c1", "p_vanish", True, 50, 0, 150))
    rows.append(("2026-03-06", "c1", "p_vanish", True, 60, 0, 180))
    # p_late_only: client c3 starts exactly on 2026-01-01. Exactly 100 counted impressions.
    rows.append(("2026-03-20", "c3", "p_late_only", True, 100, 5, 400))
    rows.append(("2026-03-21", "c3", "p_late_only", None, 1000, 50, 4000))
    rows.append(("2026-03-22", "c3", "p_late_only", False, 500, 25, 2000))
    rows.append(("2026-03-25", "c3", "p_late_only", True, 0, 0, 0))
    # p_small: one impression under the floor.
    rows.append(("2026-03-02", "c1", "p_small", True, 99, 3, 99))
    # p_other: client c2 starts after 2026-01-01.
    rows.append(("2026-03-02", "c2", "p_other", True, 500, 10, 500))
    return rows


def april_rows() -> list[tuple]:
    return [
        # 248 is exactly 80% of 310: not a decline.
        ("2026-04-03", "c1", "p_steady", True, 248, 9, 496),
        ("2026-04-04", "c1", "p_steady", False, 1000, 0, 0),
        # 87 is under 80% of 110 (88): a decline.
        ("2026-04-03", "c1", "p_vanish", True, 87, 0, 261),
        # p_late_only has no April row: counted as zero.
        ("2026-04-03", "c2", "p_other", True, 5, 0, 5),
    ]


@pytest.fixture
def tiny_source(tmp_path: Path) -> Source:
    con = duckdb.connect()
    clients = pd.DataFrame(
        {
            "client_hash_id": ["c1", "c2", "c3"],
            "gsc_data_start": pd.to_datetime(["2025-06-01", "2026-02-01", "2026-01-01"]).date,
        }
    )
    con.register("clients_df", clients)
    con.execute(
        "COPY (SELECT client_hash_id, CAST(gsc_data_start AS DATE) AS gsc_data_start "
        f"FROM clients_df) TO '{(tmp_path / 'dim_clients.parquet').as_posix()}' (FORMAT parquet)"
    )
    fact = tmp_path / "fact_content_daily_performance"
    _write(con, march_rows(), fact / "month=2026-03" / "part-0.parquet")
    _write(con, april_rows(), fact / "month=2026-04" / "part-0.parquet")
    con.close()
    return Source(root=tmp_path.as_posix())


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    yield connection
    connection.close()
