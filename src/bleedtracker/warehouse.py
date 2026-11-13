"""Access to the warehouse. It covers the token, the DuckDB connection and retried queries."""

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from bleedtracker.config import RELEASE


class MissingTokenError(RuntimeError):
    """Raised when no Hugging Face token can be found."""


def read_token(start: Path | None = None) -> str:
    """Return the Hugging Face read token.

    The HF_TOKEN environment variable comes first. Otherwise the first `.env` file
    found in `start` or one of its parents is read.
    """
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    start = (start or Path.cwd()).resolve()
    for folder in [start, *start.parents]:
        env_file = folder / ".env"
        if not env_file.is_file():
            continue
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "HF_TOKEN":
                value = value.strip().strip("\"'")
                if value:
                    return value
        break
    raise MissingTokenError(
        "No Hugging Face token found. Set HF_TOKEN, or copy .env.example to .env "
        "and fill in a read token."
    )


def _quote(text: str) -> str:
    """Quote a string for use inside SQL."""
    return "'" + text.replace("'", "''") + "'"


@dataclass(frozen=True)
class Source:
    """Where the warehouse tables live. That is the Hugging Face release or a local folder."""

    root: str = RELEASE

    def table(self, path: str) -> str:
        """SQL that reads the parquet files at `path` under the root."""
        return f"read_parquet({_quote(self.root + '/' + path)})"

    def fact(self, month: str) -> str:
        """SQL that reads one month of the daily fact table."""
        return self.table(f"fact_content_daily_performance/month={month}/*.parquet")

    def dim_clients(self) -> str:
        """SQL that reads the client table."""
        return self.table("dim_clients.parquet")


def connect(token: str | None = None) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection that can read the gated release."""
    token = token or read_token()
    con = duckdb.connect()
    con.execute(f"CREATE SECRET hf (TYPE huggingface, TOKEN {_quote(token)})")
    return con


def query(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    tries: int = 6,
    base_wait: float = 20,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Run a query and return a DataFrame.

    When the hub answers HTTP 429, the query is tried again, waiting 20 s longer
    each time (20, 40, 60, 80, 100 s with the defaults).
    """
    for attempt in range(tries):
        try:
            return con.execute(sql).df()
        except Exception as exc:
            if "429" not in str(exc) or attempt == tries - 1:
                raise
            wait = base_wait * (attempt + 1)
            retry = f"retry {attempt + 1} of {tries - 1}"
            print(f"The hub answered HTTP 429. Waiting {wait:.0f} s, {retry}.")
            sleep(wait)
    raise AssertionError("unreachable")
