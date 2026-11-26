"""Batch commands that refresh the job table must keep the chosen provider."""

from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner


@pytest.mark.parametrize(
    "arguments",
    [
        ["cancel", "--batch-id", "batch_1", "--provider", "openai"],
        ["delete", "--batch-id", "batch_1", "--provider", "openai"],
        ["create-from-file", "--file-path", "requests.jsonl"],
    ],
)
def test_job_table_refresh_keeps_the_provider(monkeypatch, arguments):
    from instructor.cli import batch as batch_cli

    listed = []
    monkeypatch.setattr(batch_cli, "BatchProcessor", MagicMock())
    monkeypatch.setattr(
        batch_cli, "get_jobs", lambda _limit, provider: listed.append(provider) or []
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    result = CliRunner().invoke(batch_cli.app, arguments)

    assert result.exit_code == 0, result.output
    assert listed == ["openai"]
