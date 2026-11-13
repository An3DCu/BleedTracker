"""Every notebook runs top to bottom from the committed results, and no output carries an id."""

import re
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

NOTEBOOKS = sorted((Path(__file__).parents[1] / "notebooks").glob("*.ipynb"))
ID_LIKE = re.compile(r"(client|content)_[0-9a-f]{6,}|[0-9a-f]{32,}")


def test_there_are_five_notebooks():
    assert [p.name for p in NOTEBOOKS] == [
        "01_question_and_data.ipynb",
        "02_rule_and_forest.ipynb",
        "03_validation_and_sealed_month.ipynb",
        "04_queue_and_recommendations.ipynb",
        "05_upgrades.ipynb",
    ]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_runs_without_errors(path):
    nb = nbformat.read(path, as_version=4)
    NotebookClient(
        nb, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}
    ).execute()
    for cell in nb.cells:
        for output in cell.get("outputs", []):
            assert output.output_type != "error"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_committed_outputs_carry_no_ids(path):
    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        for output in cell.get("outputs", []):
            text = str(output.get("text", "")) + str(output.get("data", {}).get("text/plain", ""))
            assert not ID_LIKE.search(text)
