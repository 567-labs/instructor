#!/usr/bin/env python3
"""Validate release metadata and extract notes for the declared version."""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]


REPOSITORY = "567-labs/instructor"


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        return tomllib.load(file)


def _project_version(root: Path) -> str:
    data = _load_toml(root / "pyproject.toml")
    return str(data["project"]["version"])


def _repository_url(root: Path) -> str:
    data = _load_toml(root / "pyproject.toml")
    return str(data["project"]["urls"]["repository"])


def _runtime_version(root: Path) -> str:
    init_path = root / "instructor" / "__init__.py"
    tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    versions = [
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]
    if len(versions) != 1:
        raise ValueError(
            "instructor/__init__.py must contain exactly one string __version__"
        )
    return versions[0]


def _lock_version(root: Path) -> str:
    data = _load_toml(root / "uv.lock")
    matches = [
        package
        for package in data.get("package", [])
        if package.get("name") == "instructor"
    ]
    if len(matches) != 1:
        raise ValueError("uv.lock must contain exactly one instructor package")
    return str(matches[0]["version"])


def _release_notes(changelog: str, version: str) -> str:
    unreleased = list(re.finditer(r"^## \[Unreleased\]$", changelog, re.MULTILINE))
    if len(unreleased) != 1:
        raise ValueError("CHANGELOG.md must contain exactly one [Unreleased] section")

    heading_pattern = re.compile(
        rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}$",
        re.MULTILINE,
    )
    headings = list(heading_pattern.finditer(changelog))
    if len(headings) != 1:
        raise ValueError(
            f"CHANGELOG.md must contain exactly one dated [{version}] section"
        )

    heading = headings[0]
    if unreleased[0].start() > heading.start():
        raise ValueError("[Unreleased] must appear before the current release section")

    next_heading = re.search(r"^## \[", changelog[heading.end() :], re.MULTILINE)
    end = heading.end() + next_heading.start() if next_heading else len(changelog)
    notes = changelog[heading.end() : end].strip()
    notes = re.sub(r"\n---\s*$", "", notes).strip()
    if not notes or "### " not in notes:
        raise ValueError(f"CHANGELOG.md [{version}] release notes are empty")

    comparison = re.compile(
        rf"^\[{re.escape(version)}\]: "
        rf"https://github\.com/{re.escape(REPOSITORY)}/compare/"
        rf"v\d+\.\d+\.\d+\.\.\.v{re.escape(version)}$",
        re.MULTILINE,
    )
    if not comparison.search(changelog):
        raise ValueError(f"CHANGELOG.md is missing the [{version}] comparison link")

    return notes + "\n"


def prepare_release(
    root: Path,
    output: Path,
    expected_version: str | None = None,
) -> str:
    """Validate version metadata and write the matching changelog section."""
    version = _project_version(root)
    runtime_version = _runtime_version(root)
    lock_version = _lock_version(root)
    repository_url = _repository_url(root)
    if runtime_version != version:
        raise ValueError(
            "version mismatch: "
            f"pyproject.toml={version}, instructor.__version__={runtime_version}"
        )
    if lock_version != version:
        raise ValueError(
            f"version mismatch: pyproject.toml={version}, uv.lock={lock_version}"
        )
    expected_repository_url = f"https://github.com/{REPOSITORY}"
    if repository_url != expected_repository_url:
        raise ValueError(
            "repository URL mismatch: "
            f"pyproject.toml={repository_url}, expected={expected_repository_url}"
        )
    if expected_version and expected_version != version:
        raise ValueError(
            f"expected version {expected_version}, but source declares {version}"
        )

    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    notes = _release_notes(changelog, version)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(notes, encoding="utf-8")
    return version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("dist/release-notes.md"))
    parser.add_argument("--expected-version")
    args = parser.parse_args()

    try:
        version = prepare_release(
            args.project_root.resolve(),
            args.output,
            expected_version=args.expected_version,
        )
    except (KeyError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
