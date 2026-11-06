"""Run an offline test group in isolated pytest workers and combine their coverage."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple


class UnitGroup(NamedTuple):
    paths: tuple[str, ...]
    workers: int


# More specific paths come first so each test belongs to exactly one group.
UNIT_GROUPS = {
    "focused": UnitGroup(
        (
            "tests/coverage",
            "tests/test_batch_processor_coverage.py",
            "tests/v2/decisions",
        ),
        2,
    ),
    "v2": UnitGroup(("tests/v2",), 4),
    "core": UnitGroup(("tests",), 2),
}


def group_for_path(path: str) -> str:
    for name, group in UNIT_GROUPS.items():
        if any(path == root or path.startswith(root + "/") for root in group.paths):
            return name
    raise ValueError(f"Path is outside the unit test groups: {path}")


def pytest_arguments(name: str) -> list[str]:
    roots = UNIT_GROUPS[name].paths
    separate = ["tests/docs", "tests/llm"]
    for other, group in UNIT_GROUPS.items():
        if other != name:
            separate.extend(group.paths)
    args = [*roots]
    args.extend(
        "--ignore=" + path
        for path in separate
        if any(path.startswith(root + "/") for root in roots)
    )
    args.extend(
        ["--asyncio-mode=auto", "--strict-config", "--strict-markers", "-m", "not llm"]
    )
    if name == "focused":
        args.extend(
            [
                "-W",
                "error::pytest.PytestUnraisableExceptionWarning",
                "-W",
                "error::pytest.PytestUnhandledThreadExceptionWarning",
            ]
        )
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=UNIT_GROUPS)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    workers = args.workers
    if workers is None:
        workers = min(UNIT_GROUPS[args.group].workers, os.cpu_count() or 1)
    if workers < 0:
        parser.error("--workers must be nonnegative")

    destination = Path(os.environ.get("COVERAGE_FILE", ".coverage")).resolve()
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        directory = Path(temporary)
        data = directory / ".coverage"
        env = {
            **os.environ,
            "COVERAGE_FILE": str(data),
            "COVERAGE_RCFILE": str(Path(".coveragerc").resolve()),
            "INSTRUCTOR_CI_BRANCH_COVERAGE": str(
                int(args.group != "focused" or sys.version_info[:2] == (3, 11))
            ),
        }
        coverage = [sys.executable, "-m", "coverage"]
        pytest = [
            "-m",
            "pytest",
            "-p",
            "scripts.ci_worker_coverage",
            *pytest_arguments(args.group),
            "-n",
            str(workers),
            "--dist=worksteal",
        ]
        subprocess.run([sys.executable, *pytest], env=env, check=True)
        subprocess.run([*coverage, "combine", str(directory)], env=env, check=True)
        if args.group == "focused":
            report = directory / "coverage.json"
            subprocess.run([*coverage, "json", "-o", str(report)], env=env, check=True)
            subprocess.run(
                [
                    sys.executable,
                    "scripts/check_statement_coverage.py",
                    str(report),
                    "90",
                ],
                env=env,
                check=True,
            )
        data.replace(destination)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error
