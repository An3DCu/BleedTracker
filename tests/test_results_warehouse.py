"""Every rebuilt result against every value in my original results.

Runs only with a token: pytest -m warehouse
"""

import pytest

from bleedtracker.check import compare_with_reference
from bleedtracker.reference import load_reference
from bleedtracker.results import build_results

pytestmark = pytest.mark.warehouse


def test_every_original_value_is_rebuilt(warehouse_run):
    results = build_results(
        warehouse_run.con, warehouse_run.source, warehouse_run, progress=lambda _: None
    )
    comparison = compare_with_reference(results, load_reference())
    assert comparison.differences == []
    assert comparison.numbers == comparison.matching
