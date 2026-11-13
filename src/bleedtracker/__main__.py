"""The command line. Run it as python -m bleedtracker reproduce."""

import argparse
import json
import sys
from pathlib import Path

from bleedtracker.check import compare_with_reference
from bleedtracker.deployments import plans
from bleedtracker.export import write_queue_csv
from bleedtracker.reference import REFERENCE_FILE, load_reference
from bleedtracker.release import ReleaseHeld
from bleedtracker.results import build_results, prepare_run
from bleedtracker.warehouse import MissingTokenError, Source, connect

REPORTS = Path("reports")


def write_json(data: dict, path: Path) -> None:
    """Write results as JSON with LF line endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
    path.write_bytes((text + "\n").encode("utf-8"))


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

    print(f"Checking against {REFERENCE_FILE.parent.name}/{REFERENCE_FILE.name}")
    comparison = compare_with_reference(results, load_reference())
    print(comparison.summary())
    for difference in comparison.differences:
        print(f"  {difference}")
    return 0 if comparison.all_match else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bleedtracker")
    commands = parser.add_subparsers(dest="command", required=True)
    rebuild = commands.add_parser(
        "reproduce", help="rebuild every result from the warehouse and check it"
    )
    rebuild.add_argument("--reports", type=Path, default=REPORTS, help="where to write the results")
    args = parser.parse_args(argv)
    return reproduce(args.reports)


if __name__ == "__main__":
    sys.exit(main())
