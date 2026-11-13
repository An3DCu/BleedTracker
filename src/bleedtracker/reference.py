"""My original results, read with today's names.

`reference/original.json` keeps the names the original analysis used. This module
is the one place where they are mapped to the names the package uses now.
"""

import json
from pathlib import Path

REFERENCE_FILE = Path(__file__).resolve().parents[2] / "reference" / "original.json"

OLD_NAMES = {
    "log_imp_mar": "log_impressions",
    "pos_mar": "position",
    "ctr_mar": "ctr",
    "momentum_in_month": "momentum",
    "1-3": "up to 3",
    "4-10": "3 to 10",
    "11-20": "10 to 20",
    "21+": "over 20",
}


def new_name(key: str) -> str:
    """Today's name for a key. A key like "without_pos_mar" keeps its prefix."""
    if key.startswith("without_"):
        feature = key.removeprefix("without_")
        return "without_" + OLD_NAMES.get(feature, feature)
    return OLD_NAMES.get(key, key)


def rename(value):
    """Rename every dictionary key that has a new name, at any depth."""
    if isinstance(value, dict):
        return {new_name(key): rename(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rename(item) for item in value]
    return value


def load_reference(path: Path = REFERENCE_FILE) -> dict:
    """Read my original results with today's names."""
    return rename(json.loads(path.read_text(encoding="utf-8")))
