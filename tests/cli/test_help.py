"""CLI help must be available before provider credentials are configured."""

import os
import subprocess
import sys

import pytest


def run_without_credentials(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("OPENAI_", "ANTHROPIC_", "AZURE_OPENAI_"))
    }
    return subprocess.run(
        [sys.executable, "-m", "instructor.cli.cli", *arguments],
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )


@pytest.mark.parametrize(
    "arguments",
    [[], ["files"], ["jobs"], ["files", "upload"], ["jobs", "create-from-file"]],
)
def test_help_without_credentials(arguments: list[str]) -> None:
    result = run_without_credentials([*arguments, "--help"])
    assert result.returncode == 0, result.stderr
    assert "Usage:" in result.stdout


@pytest.mark.parametrize("module_name", ["files", "jobs"])
def test_client_creation_is_lazy_and_preserves_injection(
    module_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import openai
    from instructor.cli import files, jobs

    module = files if module_name == "files" else jobs
    original = module.client
    for key in os.environ:
        if key.startswith("OPENAI_"):
            monkeypatch.delenv(key)
    module.client = None
    try:
        operation = files.get_files if module_name == "files" else jobs.get_jobs
        with pytest.raises(openai.OpenAIError):
            operation()
        assert module.client is None

        monkeypatch.setenv("OPENAI_API_KEY", "cli-contract")
        created = module._get_client()
        try:
            assert isinstance(created, openai.OpenAI)
            assert module._get_client() is created
            with openai.OpenAI(api_key="injected-cli-contract") as injected:
                module.client = injected
                assert module._get_client() is injected
        finally:
            created.close()
    finally:
        module.client = original


@pytest.mark.parametrize(
    "arguments", [["files", "delete", "file-test"], ["jobs", "cancel", "job-test"]]
)
def test_mutating_commands_fail_without_credentials(arguments: list[str]) -> None:
    result = run_without_credentials(arguments)
    assert result.returncode != 0, result.stdout
    assert "OpenAIError" in result.stderr
