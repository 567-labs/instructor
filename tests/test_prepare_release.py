from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_release.py"


def _write_release_files(root: Path, changelog: str) -> None:
    (root / "pyproject.toml").write_text(
        '[project]\nname = "instructor"\nversion = "1.2.3"\n'
        '[project.urls]\nrepository = "https://github.com/567-labs/instructor"\n',
        encoding="utf-8",
    )
    (root / "uv.lock").write_text(
        'version = 1\n[[package]]\nname = "instructor"\nversion = "1.2.3"\n',
        encoding="utf-8",
    )
    package = root / "instructor"
    package.mkdir()
    (package / "__init__.py").write_text('__version__ = "1.2.3"\n', encoding="utf-8")
    (root / "CHANGELOG.md").write_text(changelog, encoding="utf-8")


def _run(
    root: Path, expected_version: str = "1.2.3"
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--project-root",
            str(root),
            "--output",
            str(root / "release-notes.md"),
            "--expected-version",
            expected_version,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _changelog() -> str:
    return """# Changelog

## [Unreleased]

## [1.2.3] - 2026-08-02

### Fixed
- Correct retry accounting.

---

## [1.2.2] - 2026-07-01

### Fixed
- Previous fix.

[Unreleased]: https://github.com/567-labs/instructor/compare/v1.2.3...HEAD
[1.2.3]: https://github.com/567-labs/instructor/compare/v1.2.2...v1.2.3
"""


def test_prepare_release_validates_and_extracts_notes(tmp_path: Path) -> None:
    _write_release_files(tmp_path, _changelog())

    result = _run(tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "1.2.3"
    assert (tmp_path / "release-notes.md").read_text(encoding="utf-8") == (
        "### Fixed\n- Correct retry accounting.\n"
    )


def test_prepare_release_rejects_expected_version_mismatch(tmp_path: Path) -> None:
    _write_release_files(tmp_path, _changelog())

    result = _run(tmp_path, expected_version="1.2.4")

    assert result.returncode == 2
    assert "expected version 1.2.4, but source declares 1.2.3" in result.stderr


def test_prepare_release_rejects_duplicate_release_sections(tmp_path: Path) -> None:
    changelog = _changelog().replace(
        "## [1.2.2] - 2026-07-01", "## [1.2.3] - 2026-07-01"
    )
    _write_release_files(tmp_path, changelog)

    result = _run(tmp_path)

    assert result.returncode == 2
    assert "exactly one dated [1.2.3] section" in result.stderr


def test_prepare_release_rejects_lockfile_version_drift(tmp_path: Path) -> None:
    _write_release_files(tmp_path, _changelog())
    (tmp_path / "uv.lock").write_text(
        'version = 1\n[[package]]\nname = "instructor"\nversion = "1.2.2"\n',
        encoding="utf-8",
    )

    result = _run(tmp_path)

    assert result.returncode == 2
    assert "pyproject.toml=1.2.3, uv.lock=1.2.2" in result.stderr


def test_prepare_release_rejects_runtime_version_drift(tmp_path: Path) -> None:
    _write_release_files(tmp_path, _changelog())
    (tmp_path / "instructor" / "__init__.py").write_text(
        '__version__ = "1.2.2"\n', encoding="utf-8"
    )

    result = _run(tmp_path)

    assert result.returncode == 2
    assert "pyproject.toml=1.2.3, instructor.__version__=1.2.2" in result.stderr


def test_prepare_release_requires_single_runtime_version(tmp_path: Path) -> None:
    _write_release_files(tmp_path, _changelog())
    (tmp_path / "instructor" / "__init__.py").write_text("", encoding="utf-8")

    result = _run(tmp_path)

    assert result.returncode == 2
    assert "exactly one string __version__" in result.stderr


def test_prepare_release_requires_comparison_link(tmp_path: Path) -> None:
    changelog = _changelog().replace(
        "[1.2.3]: https://github.com/567-labs/instructor/compare/v1.2.2...v1.2.3",
        "",
    )
    _write_release_files(tmp_path, changelog)

    result = _run(tmp_path)

    assert result.returncode == 2
    assert "missing the [1.2.3] comparison link" in result.stderr


def test_prepare_release_rejects_stale_repository_url(tmp_path: Path) -> None:
    _write_release_files(tmp_path, _changelog())
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(
            "https://github.com/567-labs/instructor",
            "https://github.com/instructor-ai/instructor",
        ),
        encoding="utf-8",
    )

    result = _run(tmp_path)

    assert result.returncode == 2
    assert "repository URL mismatch" in result.stderr
    assert "expected=https://github.com/567-labs/instructor" in result.stderr
