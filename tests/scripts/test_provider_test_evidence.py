"""Exercise reporting against real pytest output, including xdist and collection."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.provider_test_evidence import summarize

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/provider_test_evidence.py"


@pytest.mark.parametrize(
    ("source", "expected", "exit_code", "extra"),
    [
        ("def test_ok(): pass", "| 1 | 0 | 0 | 0 |", 0, []),
        (
            "import pytest\ndef test_skip(): pytest.skip('OPENAI_API_KEY not set')",
            "credentials unavailable: 1",
            0,
            [],
        ),
        (
            "import pytest\npytest.skip('SDK package not installed', allow_module_level=True)",
            "SDK/package unavailable: 1",
            5,
            [],
        ),
        ("def test_bad(): assert False", "| 0 | 1 | 0 | 0 |", 1, []),
        ("raise RuntimeError('collection broke')", "| 0 | 0 | 1 | 0 |", 2, []),
        ("# empty module", "No test outcomes recorded", 5, []),
        (
            "import pytest\n@pytest.mark.xfail\ndef test_known(): assert False",
            "expected failure: 1",
            0,
            [],
        ),
        (
            "import pytest\ndef test_ok(): pass\ndef test_skip(): pytest.skip('unsupported streaming')",
            "| 1 | 0 | 0 | 1 |",
            0,
            ["-p", "xdist.plugin", "-n", "2"],
        ),
    ],
)
def test_real_pytest_reports(tmp_path: Path, source, expected, exit_code, extra):
    (tmp_path / "test_example.py").write_text(source)
    report = tmp_path / "provider-evidence-initial.xml"
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", f"--junitxml={report}", *extra],
        cwd=tmp_path,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTEST_ADDOPTS": ""},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert run.returncode == exit_code, run.stdout + run.stderr
    assert expected in summarize(report)


def test_absent_and_corrupt_reports_are_unknown(tmp_path: Path):
    report = tmp_path / "missing.xml"
    assert "counts are unknown" in summarize(report)
    report.write_text("<interrupted")
    assert "counts are unknown" in summarize(report)
    assert "| 0 |" not in summarize(report)
    report.write_text("<html/>")
    assert "counts are unknown" in summarize(report)


def test_cli_keeps_retry_evidence_and_hides_secret_values(tmp_path: Path):
    summary = tmp_path / "summary.md"
    (tmp_path / "provider-evidence-initial.xml").write_text(
        '<testsuites><testsuite><testcase><failure message="private payload"/></testcase></testsuite></testsuites>'
    )
    (tmp_path / "provider-evidence-retry.xml").write_text(
        "<testsuites><testsuite><testcase/></testsuite></testsuites>"
    )
    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--reports",
            str(tmp_path),
        ],
        env={
            **os.environ,
            "GITHUB_STEP_SUMMARY": str(summary),
            "CONTRACT_TEST_KEY": "private-key-value",
            "PROVIDER_TEST_REQUIREMENTS": "CONTRACT_TEST_KEY",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    content = summary.read_text()
    assert "| 0 | 1 | 0 | 0 |" in content
    assert "| 1 | 0 | 0 | 0 |" in content
    assert "`CONTRACT_TEST_KEY`: present" in content
    assert "private" not in content
    assert run.stdout.strip() == content.strip()


@pytest.mark.parametrize("documented_only", [False, True])
def test_cli_missing_primary_report_is_unknown(tmp_path: Path, documented_only: bool):
    if documented_only:
        (tmp_path / "provider-evidence-documented-models.xml").write_text(
            "<testsuites><testsuite><testcase/></testsuite></testsuites>"
        )
    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--reports",
            str(tmp_path),
            "--requirements",
            "CONTRACT_TEST_KEY",
        ],
        env={**os.environ, "GITHUB_STEP_SUMMARY": "", "CONTRACT_TEST_KEY": ""},
        capture_output=True,
        text=True,
        check=True,
    )
    assert "counts are unknown" in run.stdout
    assert "`CONTRACT_TEST_KEY`: missing" in run.stdout
    if documented_only:
        assert "provider-evidence-initial" in run.stdout
        assert "provider-evidence-documented-models" in run.stdout
        assert "| 1 | 0 | 0 | 0 |" in run.stdout
    else:
        assert "| 0 |" not in run.stdout
