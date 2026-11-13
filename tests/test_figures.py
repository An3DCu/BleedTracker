import json
import shutil
from pathlib import Path

from bleedtracker.__main__ import main
from bleedtracker.figures import draw_all, draw_queue_length, month_label, months, signed

RESULTS_FILE = Path(__file__).parents[1] / "reports" / "results.json"


def results() -> dict:
    return json.loads(RESULTS_FILE.read_text(encoding="utf-8"))


def test_signed_numbers_use_a_minus_sign():
    assert signed(65.6) == "+65.6"
    assert signed(-4.1) == "−4.1"


def test_month_label():
    assert month_label("sealed 2026-05 to 2026-06") == "2026-05 to 06"


def test_months():
    assert months(1) == "one month"
    assert months(2) == "two months"


def test_counts_in_the_text_come_from_the_results():
    title = draw_queue_length(results()).axes[1].get_title(loc="left")
    assert "Above its pool in two months. Below it in two, including the sealed month." in title


def test_every_figure_is_drawn_as_png_and_svg(tmp_path):
    written = draw_all(results(), tmp_path)
    assert sorted(p.name for p in written) == [
        "figure_1_deployments.png",
        "figure_1_deployments.svg",
        "figure_2_queue_length.png",
        "figure_2_queue_length.svg",
        "figure_3_queue.png",
        "figure_3_queue.svg",
    ]


def test_a_redraw_gives_the_same_files(tmp_path):
    first = draw_all(results(), tmp_path / "first")
    second = draw_all(results(), tmp_path / "second")
    for a, b in zip(first, second, strict=True):
        assert a.read_bytes() == b.read_bytes(), a.name


def test_figures_command_needs_a_results_file(tmp_path):
    assert main(["figures", "--reports", str(tmp_path)]) == 2


def test_figures_command_redraws_from_the_results_file(tmp_path):
    shutil.copy(RESULTS_FILE, tmp_path / "results.json")
    assert main(["figures", "--reports", str(tmp_path)]) == 0
    assert len(list((tmp_path / "figures").iterdir())) == 6
