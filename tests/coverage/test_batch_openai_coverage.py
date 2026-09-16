"""Behavior tests for the OpenAI batch provider and the provider contract."""

from __future__ import annotations

import io
import importlib.util
import runpy
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import openai
import pytest

import instructor.batch.providers as providers
from instructor.batch.models import BatchJobInfo, BatchStatus
from instructor.batch.providers.base import BatchProvider
from instructor.batch.providers.openai import OpenAIProvider

pytestmark = pytest.mark.unit


class BatchResponse:
    """Small SDK-shaped batch response used by the provider tests."""

    def __init__(
        self,
        batch_id: str = "batch_123",
        status: str = "completed",
        output_file_id: str | None = "file_output",
        request_counts: Any = None,
        error_file_id: str | None = None,
    ) -> None:
        self.id = batch_id
        self.status = status
        self.output_file_id = output_file_id
        self.error_file_id = error_file_id
        self.request_counts = request_counts
        self.created_at = 1_700_000_000

    def model_dump(self) -> dict[str, Any]:
        counts = self.request_counts
        return {
            "id": self.id,
            "status": self.status,
            "created_at": self.created_at,
            "request_counts": {
                "total": getattr(counts, "total", None),
                "completed": getattr(counts, "completed", None),
                "failed": getattr(counts, "failed", None),
            },
            "input_file_id": "file_input",
            "output_file_id": self.output_file_id,
            "metadata": {"source": "unit-test"},
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        }


class FilesAPI:
    def __init__(self, text: str = '{"ok": true}\n') -> None:
        self.text = text
        self.uploads: list[tuple[bytes, str]] = []
        self.content_ids: list[str] = []
        self.create_error: Exception | None = None
        self.content_error: Exception | None = None

    def create(self, *, file: Any, purpose: str) -> SimpleNamespace:
        if self.create_error:
            raise self.create_error
        self.uploads.append((file.read(), purpose))
        return SimpleNamespace(id="file_input")

    def content(self, file_id: str) -> SimpleNamespace:
        if self.content_error:
            raise self.content_error
        self.content_ids.append(file_id)
        return SimpleNamespace(text=self.text)


class BatchesAPI:
    def __init__(self, responses: Sequence[BatchResponse] | None = None) -> None:
        self.responses = responses or [BatchResponse()]
        self.retrieve_ids: list[str] = []
        self.created: list[dict[str, Any]] = []
        self.cancelled: list[str] = []
        self.limits: list[int] = []
        self.create_error: Exception | None = None
        self.retrieve_error: Exception | None = None
        self.cancel_error: Exception | None = None
        self.list_error: Exception | None = None

    def create(self, **kwargs: Any) -> SimpleNamespace:
        if self.create_error:
            raise self.create_error
        self.created.append(kwargs)
        return SimpleNamespace(id="batch_created")

    def retrieve(self, batch_id: str) -> BatchResponse:
        if self.retrieve_error:
            raise self.retrieve_error
        self.retrieve_ids.append(batch_id)
        index = min(len(self.retrieve_ids) - 1, len(self.responses) - 1)
        return self.responses[index]

    def cancel(self, batch_id: str) -> BatchResponse:
        if self.cancel_error:
            raise self.cancel_error
        self.cancelled.append(batch_id)
        return BatchResponse(batch_id=batch_id, status="cancelled")

    def list(self, *, limit: int) -> SimpleNamespace:
        if self.list_error:
            raise self.list_error
        self.limits.append(limit)
        return SimpleNamespace(data=self.responses)


class OpenAIClient:
    def __init__(self, responses: list[BatchResponse] | None = None) -> None:
        self.files = FilesAPI()
        self.batches = BatchesAPI(responses)


def install_client(
    monkeypatch: pytest.MonkeyPatch, client: OpenAIClient
) -> OpenAIProvider:
    monkeypatch.setattr(openai, "OpenAI", lambda: client)
    return OpenAIProvider()


def test_submit_batch_uploads_file_and_uses_default_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    batch_file = tmp_path / "requests.jsonl"
    batch_file.write_bytes(b'{"custom_id": "one"}\n')
    client = OpenAIClient()
    provider = install_client(monkeypatch, client)

    batch_id = provider.submit_batch(str(batch_file))

    assert batch_id == "batch_created"
    assert client.files.uploads == [(b'{"custom_id": "one"}\n', "batch")]
    assert client.batches.created == [
        {
            "input_file_id": "file_input",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "metadata": {"description": "Instructor batch job"},
        }
    ]


def test_submit_batch_rewinds_buffer_and_passes_custom_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = io.BytesIO(b'{"custom_id": "two"}\n')
    buffer.seek(5)
    client = OpenAIClient()
    provider = install_client(monkeypatch, client)

    batch_id = provider.submit_batch(
        buffer, metadata={"source": "daily-import"}, completion_window="48h"
    )

    assert batch_id == "batch_created"
    assert client.files.uploads == [(b'{"custom_id": "two"}\n', "batch")]
    assert client.batches.created == [
        {
            "input_file_id": "file_input",
            "endpoint": "/v1/chat/completions",
            "completion_window": "48h",
            "metadata": {"source": "daily-import"},
        }
    ]


def test_submit_batch_rejects_invalid_input_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    invalid_input: Any = 123

    with pytest.raises(ValueError, match="Unsupported file_path_or_buffer type"):
        OpenAIProvider().submit_batch(invalid_input)


@pytest.mark.parametrize("error", [ValueError("bad input"), TypeError("bad file")])
def test_submit_batch_preserves_validation_errors(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    client = OpenAIClient()
    client.files.create_error = error
    provider = install_client(monkeypatch, client)

    with pytest.raises(type(error), match=str(error)) as caught:
        provider.submit_batch(io.BytesIO(b"{}\n"))

    assert caught.value is error
    assert client.batches.created == []


def test_submit_batch_wraps_service_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OpenAIClient()
    client.batches.create_error = ConnectionError("service unavailable")
    provider = install_client(monkeypatch, client)

    with pytest.raises(RuntimeError, match="Failed to submit OpenAI batch") as caught:
        provider.submit_batch(io.BytesIO(b"{}\n"))

    assert isinstance(caught.value.__cause__, ConnectionError)
    assert "service unavailable" in str(caught.value)


def test_get_status_maps_sdk_response_and_missing_count_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = BatchResponse(status="in_progress", request_counts=SimpleNamespace(total=4))
    client = OpenAIClient([batch])
    provider = install_client(monkeypatch, client)

    status = provider.get_status("batch_123")

    assert status == {
        "id": "batch_123",
        "status": "in_progress",
        "created_at": 1_700_000_000,
        "request_counts": {"total": 4, "completed": 0, "failed": 0},
    }
    assert client.batches.retrieve_ids == ["batch_123"]


def test_get_status_wraps_sdk_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OpenAIClient()
    client.batches.retrieve_error = ConnectionError("status service unavailable")
    provider = install_client(monkeypatch, client)

    with pytest.raises(Exception, match="Failed to get OpenAI batch status") as caught:
        provider.get_status("batch_missing")

    assert isinstance(caught.value.__cause__, ConnectionError)


@pytest.mark.parametrize("operation", ["retrieve", "download"])
def test_completed_batch_reads_or_writes_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, operation: str
) -> None:
    counts = SimpleNamespace(total=1, completed=1, failed=0)
    batch = BatchResponse(request_counts=counts)
    client = OpenAIClient([batch])
    client.files.text = '{"custom_id": "one", "response": {}}\n'
    provider = install_client(monkeypatch, client)
    destination = tmp_path / "results.jsonl"

    if operation == "retrieve":
        result = provider.retrieve_results("batch_123")
        assert result == client.files.text
    else:
        result = provider.download_results("batch_123", str(destination))
        assert result is None
        assert destination.read_text() == client.files.text

    assert client.batches.retrieve_ids == ["batch_123"]
    assert client.files.content_ids == ["file_output"]


@pytest.mark.parametrize("operation", ["retrieve", "download"])
def test_results_wait_for_output_file_then_succeed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    operation: str,
) -> None:
    waiting = BatchResponse(output_file_id=None)
    ready = BatchResponse(output_file_id="file_ready")
    client = OpenAIClient([waiting, ready])
    provider = install_client(monkeypatch, client)
    waits: list[int] = []
    monkeypatch.setattr("time.sleep", waits.append)
    destination = tmp_path / "results.jsonl"

    if operation == "retrieve":
        assert provider.retrieve_results("batch_123") == client.files.text
    else:
        provider.download_results("batch_123", str(destination))
        assert destination.read_text() == client.files.text

    output = capsys.readouterr().out
    assert waits == [5]
    assert "waiting 5s (attempt 1/10)" in output
    assert "Output file now available: file_ready" in output
    assert client.batches.retrieve_ids == ["batch_123", "batch_123"]
    assert client.files.content_ids == ["file_ready"]


@pytest.mark.parametrize("operation", ["retrieve", "download"])
def test_results_reject_noncompleted_batches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, operation: str
) -> None:
    pending_client = OpenAIClient([BatchResponse(status="in_progress")])
    provider = install_client(monkeypatch, pending_client)
    destination = tmp_path / "results.jsonl"

    with pytest.raises(Exception, match="Batch not completed, status: in_progress"):
        if operation == "retrieve":
            provider.retrieve_results("batch_pending")
        else:
            provider.download_results("batch_pending", str(destination))

    assert pending_client.files.content_ids == []
    assert not destination.exists()


@pytest.mark.parametrize("operation", ["retrieve", "download"])
def test_results_stop_when_status_changes_while_waiting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, operation: str
) -> None:
    responses = [
        BatchResponse(output_file_id=None),
        BatchResponse(status="failed", output_file_id=None),
    ]
    client = OpenAIClient(responses)
    provider = install_client(monkeypatch, client)
    waits: list[int] = []
    monkeypatch.setattr("time.sleep", waits.append)
    destination = tmp_path / "results.jsonl"

    with pytest.raises(Exception, match="Batch status changed to failed"):
        if operation == "retrieve":
            provider.retrieve_results("batch_123")
        else:
            provider.download_results("batch_123", str(destination))

    assert waits == [5]
    assert len(client.batches.retrieve_ids) == 2
    assert client.files.content_ids == []
    assert not destination.exists()


@pytest.mark.parametrize("operation", ["retrieve", "download"])
@pytest.mark.parametrize("available_file", [None, "file_output", "file_errors"])
def test_results_report_exhausted_output_retries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    operation: str,
    available_file: str | None,
) -> None:
    client = OpenAIClient(
        [
            BatchResponse(
                output_file_id="file_output"
                if available_file == "file_output"
                else None,
                error_file_id="file_errors"
                if available_file == "file_errors"
                else None,
                request_counts=SimpleNamespace(total=2, completed=1, failed=1),
            )
        ]
    )
    provider = install_client(monkeypatch, client)
    waits: list[int] = []
    monkeypatch.setattr("time.sleep", waits.append)
    destination = tmp_path / "results.jsonl"

    with pytest.raises(
        Exception, match="Expected result files not available after 10 retries"
    ):
        if operation == "retrieve":
            provider.retrieve_results("batch_123")
        else:
            provider.download_results("batch_123", str(destination))

    assert waits == list(range(5, 15))
    assert len(client.batches.retrieve_ids) == 11
    assert client.files.content_ids == []
    assert not destination.exists()


@pytest.mark.parametrize("operation", ["retrieve", "download"])
def test_results_wrap_file_read_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, operation: str
) -> None:
    client = OpenAIClient()
    client.files.content_error = ConnectionError("file service unavailable")
    provider = install_client(monkeypatch, client)
    destination = tmp_path / "results.jsonl"

    expected = (
        "Failed to retrieve OpenAI results"
        if operation == "retrieve"
        else "Failed to download OpenAI results"
    )
    with pytest.raises(Exception, match=expected) as caught:
        if operation == "retrieve":
            provider.retrieve_results("batch_123")
        else:
            provider.download_results("batch_123", str(destination))

    assert isinstance(caught.value.__cause__, ConnectionError)
    assert not destination.exists()


def test_cancel_delete_and_list_batches_map_sdk_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = SimpleNamespace(total=4, completed=4, failed=0)
    batch = BatchResponse(request_counts=counts)
    client = OpenAIClient([batch])
    provider = install_client(monkeypatch, client)

    cancelled = provider.cancel_batch("batch_cancel")
    deleted = provider.delete_batch("batch_123")
    listed = provider.list_batches(limit=3)

    assert cancelled["id"] == "batch_cancel"
    assert cancelled["status"] == "cancelled"
    assert client.batches.cancelled == ["batch_cancel"]
    assert deleted == {
        "id": "batch_123",
        "status": "completed",
        "message": "OpenAI does not support batch deletion",
    }
    assert client.batches.limits == [3]
    assert len(listed) == 1
    assert isinstance(listed[0], BatchJobInfo)
    assert listed[0].provider == "openai"
    assert listed[0].status == BatchStatus.COMPLETED
    assert listed[0].request_counts.total == 4
    assert listed[0].files.output_file_id == "file_output"


@pytest.mark.parametrize(
    ("operation", "error_attribute", "expected"),
    [
        ("cancel", "cancel_error", "Failed to cancel OpenAI batch"),
        ("delete", "retrieve_error", "Failed to delete OpenAI batch"),
        ("list", "list_error", "Failed to list OpenAI batches"),
    ],
)
def test_batch_actions_wrap_sdk_errors(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    error_attribute: str,
    expected: str,
) -> None:
    client = OpenAIClient()
    setattr(
        client.batches, error_attribute, ConnectionError("batch service unavailable")
    )
    provider = install_client(monkeypatch, client)

    with pytest.raises(Exception, match=expected) as caught:
        if operation == "cancel":
            provider.cancel_batch("batch_123")
        elif operation == "delete":
            provider.delete_batch("batch_123")
        else:
            provider.list_batches()

    assert isinstance(caught.value.__cause__, ConnectionError)


def test_provider_factory_selects_available_providers() -> None:
    openai_provider = providers.get_provider("openai")
    anthropic_provider = providers.get_provider("anthropic")

    openai_provider_type = providers.OpenAIProvider
    anthropic_provider_type = providers.AnthropicProvider
    assert openai_provider_type is not None
    assert anthropic_provider_type is not None
    assert isinstance(openai_provider, openai_provider_type)
    assert isinstance(anthropic_provider, anthropic_provider_type)
    assert isinstance(openai_provider, BatchProvider)
    assert isinstance(anthropic_provider, BatchProvider)


@pytest.mark.parametrize(
    ("provider_name", "attribute", "expected"),
    [
        ("openai", "OpenAIProvider", "OpenAI is not installed"),
        ("anthropic", "AnthropicProvider", "Anthropic is not installed"),
    ],
)
def test_provider_factory_reports_missing_optional_provider(
    monkeypatch: pytest.MonkeyPatch,
    provider_name: str,
    attribute: str,
    expected: str,
) -> None:
    monkeypatch.setattr(providers, attribute, None)

    with pytest.raises(ValueError, match=expected):
        providers.get_provider(provider_name)


def test_provider_factory_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported provider: unknown"):
        providers.get_provider("unknown")


def test_provider_module_handles_missing_optional_sdks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: None
        if name in {"openai", "anthropic"}
        else original_find_spec(name),
    )

    namespace = runpy.run_path(
        providers.__file__, run_name="instructor.batch.providers.without_sdks"
    )

    assert namespace["OpenAIProvider"] is None
    assert namespace["AnthropicProvider"] is None
    with pytest.raises(ValueError, match="OpenAI is not installed"):
        namespace["get_provider"]("openai")
    with pytest.raises(ValueError, match="Anthropic is not installed"):
        namespace["get_provider"]("anthropic")


def test_batch_provider_contract_requires_and_defines_all_operations() -> None:
    expected = {
        "submit_batch",
        "get_status",
        "retrieve_results",
        "download_results",
        "cancel_batch",
        "delete_batch",
        "list_batches",
    }
    assert BatchProvider.__abstractmethods__ == expected
    with pytest.raises(TypeError, match="abstract methods"):
        BatchProvider()
