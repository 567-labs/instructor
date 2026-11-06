"""Regression tests for avoiding eager OpenAI SDK imports."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


def _check_openai_not_loaded(access_code: str) -> None:
    script = textwrap.dedent(
        f"""
        import sys
        import instructor

        {access_code}

        loaded = [module for module in sys.modules if module.startswith("openai")]
        if loaded:
            print(f"FAIL: openai loaded ({{len(loaded)}} modules)")
            sys.exit(1)
        print("PASS: openai not loaded")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, (
        f"openai was eagerly loaded after: {access_code}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_bare_import_does_not_load_openai():
    _check_openai_not_loaded("pass")


def test_from_provider_access_does_not_load_openai():
    _check_openai_not_loaded("_ = instructor.from_provider")


def test_instructor_class_access_does_not_load_openai():
    _check_openai_not_loaded("_ = instructor.Instructor")


def test_mode_access_does_not_load_openai():
    _check_openai_not_loaded("_ = instructor.Mode")


def test_partial_access_does_not_load_openai():
    _check_openai_not_loaded("_ = instructor.Partial")


@pytest.mark.parametrize(
    "symbol",
    [
        "from_provider",
        "Instructor",
        "AsyncInstructor",
        "Mode",
        "Partial",
        "Maybe",
    ],
)
def test_core_symbols_accessible_without_openai(symbol: str):
    script = textwrap.dedent(
        f"""
        import instructor

        obj = getattr(instructor, "{symbol}")
        assert obj is not None
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, (
        f"Failed to access instructor.{symbol}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
