"""Check the statement-coverage threshold using coverage.py's report precision."""

import argparse
import json
from pathlib import Path
from typing import cast

from coverage import Coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("minimum", type=int)
    args = parser.parse_args()

    totals = json.loads(args.report.read_text())["totals"]
    if totals["num_statements"] == 0:
        raise SystemExit("Coverage report contains no statements")
    missing = totals["missing_lines"]
    percent = 100 * totals["covered_lines"] / totals["num_statements"]
    print(f"Statement coverage {percent:.2f}% with {missing} missing lines")
    if args.minimum == 100:
        raise SystemExit(int(missing != 0))
    precision = cast(int, Coverage().get_option("report:precision"))
    raise SystemExit(int(round(percent, precision) < args.minimum))


if __name__ == "__main__":
    main()
