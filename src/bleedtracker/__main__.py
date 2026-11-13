"""The command line. Run it as python -m bleedtracker reproduce, figures or pick-lines."""

import argparse
import json
import sys
from pathlib import Path

from bleedtracker.check import compare_with_reference
from bleedtracker.deployments import ANCHOR_PAIR, plans
from bleedtracker.export import write_queue_csv
from bleedtracker.figures import draw_all
from bleedtracker.lines import lines_report, pick, score_grid, score_riser_gates
from bleedtracker.model import split_clients
from bleedtracker.reference import REFERENCE_FILE, load_reference
from bleedtracker.release import ReleaseHeld, load_open_frames
from bleedtracker.results import build_results, prepare_run
from bleedtracker.warehouse import MissingTokenError, Source, connect

REPORTS = Path("reports")


def write_json(data: dict, path: Path) -> None:
    """Write results as JSON with LF line endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
    path.write_bytes((text + "\n").encode("utf-8"))


def draw_figures(results: dict, reports: Path) -> None:
    """Draw the three figures into the reports folder."""
    folder = reports / "figures"
    written = draw_all(results, folder)
    print(f"Drawing {len(written) // 2} figures in {folder.as_posix()} as PNG and SVG")


def figures(reports: Path) -> int:
    """Redraw the figures from the results file. No token is needed."""
    results_path = reports / "results.json"
    if not results_path.is_file():
        print(f"{results_path.as_posix()} not found. Run python -m bleedtracker reproduce first.")
        return 2
    draw_figures(json.loads(results_path.read_text(encoding="utf-8")), reports)
    return 0


def reproduce(reports: Path) -> int:
    """Rebuild every result, write it and check it. Returns the exit status."""
    try:
        con = connect()
    except MissingTokenError as error:
        print(error)
        return 2
    source = Source()
    try:
        run = prepare_run(con, source)
    except ReleaseHeld as error:
        print(error)
        return 1
    results = build_results(con, source, run)

    results_path = reports / "results.json"
    print(f"Writing {results_path.as_posix()}")
    write_json(results, results_path)
    queue, pool = run.ladder.queues[plans()[-1].name]
    csv_path = reports / "sealed_queue.csv"
    rows = write_queue_csv(queue, pool, csv_path)
    print(f"Writing {csv_path.as_posix()} ({rows:,} pages, kept out of git)")

    draw_figures(results, reports)

    print(f"Checking against {REFERENCE_FILE.parent.name}/{REFERENCE_FILE.name}")
    comparison = compare_with_reference(results, load_reference())
    print(comparison.summary())
    for difference in comparison.differences:
        print(f"  {difference}")
    return 0 if comparison.all_match else 1


def pick_lines(reports: Path) -> int:
    """Pick rule v2's slide line and riser gate on the training clients of the open months."""
    try:
        con = connect()
    except MissingTokenError as error:
        print(error)
        return 2
    print("Reading the four open months")
    frames = load_open_frames(con, Source(), original_momentum=False)
    train_clients, _ = split_clients(frames[ANCHOR_PAIR])
    print(f"Step 1 scores 80 pairs of lines on the {len(train_clients)} training clients")
    slide_scores = score_grid(frames, train_clients)
    slide_line = pick(slide_scores)["slide_line"]
    print(f"Step 2 scores 10 riser gates by the shadow queue at slide line {slide_line}")
    riser_scores = score_riser_gates(frames, train_clients, slide_line)
    report = lines_report(slide_scores, riser_scores, frames, train_clients)
    path = reports / "v2_lines.json"
    print(f"Writing {path.as_posix()}")
    write_json(report, path)
    print(
        f"Slide line {slide_line} scores {report['step_1_picked']['score']} "
        f"against {report['step_1_original_lines']['score']} for 0.8."
    )
    print(
        f"Riser gate {report['picked']['riser_gate']} gives the shadow queue "
        f"{report['step_2_picked']['score']} against {report['step_2_original_gate']['score']} "
        "for 1.20."
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bleedtracker")
    commands = parser.add_subparsers(dest="command", required=True)
    rebuild = commands.add_parser(
        "reproduce", help="rebuild every result from the warehouse and check it"
    )
    rebuild.add_argument("--reports", type=Path, default=REPORTS, help="where to write the results")
    redraw = commands.add_parser("figures", help="redraw the figures from reports/results.json")
    redraw.add_argument("--reports", type=Path, default=REPORTS, help="where the results are")
    lines = commands.add_parser(
        "pick-lines", help="pick rule v2's lines on the training clients of the open months"
    )
    lines.add_argument("--reports", type=Path, default=REPORTS, help="where to write the pick")
    args = parser.parse_args(argv)
    if args.command == "figures":
        return figures(args.reports)
    if args.command == "pick-lines":
        return pick_lines(args.reports)
    return reproduce(args.reports)


if __name__ == "__main__":
    sys.exit(main())
