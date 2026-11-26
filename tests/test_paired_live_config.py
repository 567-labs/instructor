"""Collection contracts for the paired live cases, checked without a network call."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.llm.shared_config import PROVIDER_CONFIGS, paired_parameters


ROUTES = {
    "openai",
    "openai-responses",
    "anthropic",
    "google",
    "cohere",
    "mistral",
    "openrouter",
}
CASES = {
    "test_int",
    "test_simple_extraction",
    "test_list_extraction",
    "test_nested_model_extraction",
    "test_enum_extraction",
    "test_nested_enum_collection",
    "test_validation_with_retries",
    "test_partial_streaming",
    "test_iterable_streaming",
}


def test_paired_collection_has_every_case_for_every_route() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    for key in [*(config[2] for config in PROVIDER_CONFIGS), "OPENROUTER_API_KEY"]:
        env[key] = ""
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/llm/test_core_providers",
            "--collect-only",
            "-m",
            "paired",
            "-q",
        ],
        cwd=root,
        env={**env, "PYTEST_ADDOPTS": ""},
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    ids = [
        line.rsplit("::", 1)[-1]
        for line in run.stdout.splitlines()
        if "::test_" in line
    ]
    assert sorted(ids) == sorted(
        f"{case}[{route}]" for case in CASES for route in ROUTES
    )


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("", "google/gemini-3.8-flash"),
        ("custom", "google/custom"),
        ("google/custom", "google/custom"),
        ("gemini/custom", "google/custom"),
    ],
)
def test_paired_google_id_is_independent_of_override(
    monkeypatch: pytest.MonkeyPatch, configured: str, expected: str
) -> None:
    monkeypatch.setenv("GOOGLE_GENAI_MODEL", configured)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    params = {param.id: param for param in paired_parameters(only=True)}
    assert params.keys() == ROUTES
    google = params["google"]
    config = google.values[0]
    assert isinstance(config, tuple)
    assert config[0] == expected
    assert any(
        mark.name == "skip"
        and mark.kwargs.get("reason") == "GOOGLE_API_KEY is not configured"
        for mark in google.marks
    )
