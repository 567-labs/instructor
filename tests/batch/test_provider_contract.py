"""Batch provider contract: every provider runs the same operations the same way.

Each provider gets one SDK-typed fake behind ``Harness``. Shared behaviors are
parametrized over operation x provider; provider-specific behaviors follow as
explicit cases.
"""

from __future__ import annotations

import gc
import importlib
import importlib.util
import io
import json
import logging
import weakref
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import anthropic
import openai
import pytest
from anthropic.types.messages import (
    MessageBatch,
    MessageBatchIndividualResponse,
)
from mistralai.client.models import (
    BatchJob,
    DeleteBatchJobResponse,
    ListBatchJobsResponse,
)
from openai.types import Batch, FileObject
from pydantic import BaseModel

import instructor.batch.providers as batch_providers
from instructor.batch import BatchProcessor
from instructor.batch.models import BatchError, BatchStatus, BatchSuccess
from instructor.batch.providers import mistral as mistral_module
from instructor.batch.providers.anthropic import AnthropicProvider
from instructor.batch.providers.base import BatchProvider
from instructor.batch.providers.mistral import MistralProvider
from instructor.batch.providers.openai import OpenAIProvider

pytestmark = pytest.mark.unit

CREATED = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
EPOCH = int(CREATED.timestamp())
PAYLOAD = b'{"custom_id": "request-0"}\n'


# --- SDK-typed builders ------------------------------------------------------


def openai_batch(
    status: str = "completed",
    output: str | None = "file_output",
    error: str | None = None,
    counts: tuple[int, int, int] | None = (1, 1, 0),
    batch_id: str = "batch_123",
) -> Batch:
    total, completed, failed = counts or (0, 0, 0)
    return Batch.model_validate(
        {
            "id": batch_id,
            "object": "batch",
            "endpoint": "/v1/chat/completions",
            "input_file_id": "file_input",
            "completion_window": "24h",
            "created_at": EPOCH,
            "status": status,
            "output_file_id": output,
            "error_file_id": error,
            "metadata": {"source": "unit-test"},
            "request_counts": None
            if counts is None
            else {"total": total, "completed": completed, "failed": failed},
        }
    )


def anthropic_batch(
    status: str = "ended",
    batch_id: str = "batch_123",
    succeeded: int = 2,
    errored: int = 0,
    processing: int = 0,
) -> MessageBatch:
    return MessageBatch.model_validate(
        {
            "id": batch_id,
            "type": "message_batch",
            "processing_status": status,
            "created_at": CREATED,
            "ended_at": CREATED + timedelta(minutes=1) if status == "ended" else None,
            "expires_at": CREATED + timedelta(days=1),
            "archived_at": None,
            "cancel_initiated_at": None,
            "results_url": f"https://api.anthropic.com/v1/messages/batches/{batch_id}/results",
            "request_counts": {
                "canceled": 0,
                "errored": errored,
                "expired": 0,
                "processing": processing,
                "succeeded": succeeded,
            },
        }
    )


def anthropic_result(custom_id: str, text: str) -> MessageBatchIndividualResponse:
    return MessageBatchIndividualResponse.model_validate(
        {
            "custom_id": custom_id,
            "result": {
                "type": "succeeded",
                "message": {
                    "id": f"msg_{custom_id}",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5",
                    "content": [{"type": "text", "text": text}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 12, "output_tokens": 3},
                },
            },
        }
    )


def anthropic_errored(custom_id: str) -> MessageBatchIndividualResponse:
    return MessageBatchIndividualResponse.model_validate(
        {
            "custom_id": custom_id,
            "result": {
                "type": "errored",
                "error": {
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": "Invalid"},
                },
            },
        }
    )


def mistral_job(
    status: str = "SUCCESS",
    job_id: str = "batch_123",
    succeeded: int = 2,
    failed: int = 0,
    output: str | None = "file_out",
    error: str | None = None,
    completed: int | None = None,
) -> BatchJob:
    return BatchJob.model_validate(
        {
            "id": job_id,
            "status": status,
            "created_at": EPOCH,
            "input_files": ["file_in"],
            "output_file": output,
            "error_file": error,
            "errors": [],
            "endpoint": "/v1/chat/completions",
            "model": "mistral-small-latest",
            "total_requests": succeeded + failed,
            "completed_requests": succeeded + failed
            if completed is None
            else completed,
            "succeeded_requests": succeeded,
            "failed_requests": failed,
        }
    )


# --- one fake per provider behind a shared protocol --------------------------


class Harness:
    """Installs a fake SDK client; records SDK calls and raises scripted failures."""

    name: ClassVar[str]
    label: ClassVar[str]
    running: ClassVar[str]
    # Contract operation -> the SDK call it depends on.
    sdk_calls: ClassVar[dict[str, str]]
    upload_call: ClassVar[str]
    submit_kwargs: ClassVar[dict[str, Any]] = {}
    messages: ClassVar[dict[str, str]] = {}

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.monkeypatch = monkeypatch
        self.calls: list[tuple[str, Any]] = []
        self.failures: dict[str, BaseException] = {}
        self.uploaded: list[dict[str, Any]] = []

    provider: BatchProvider

    def record(self, method: str, argument: Any = None) -> None:
        if method in self.failures:
            raise self.failures[method]
        self.calls.append((method, argument))

    def called(self, method: str) -> list[Any]:
        return [argument for name, argument in self.calls if name == method]

    def upload(self, content: bytes) -> None:
        self.uploaded.extend(json.loads(line) for line in content.splitlines())

    def set_running(self) -> None:
        raise NotImplementedError

    def serve(self, texts: list[str]) -> tuple[str, str]:
        """Serve result records; return the expected retrieved and downloaded text."""
        raise NotImplementedError

    def interrupt_results(self) -> None:
        """Make the result stream fail after it has started."""
        raise NotImplementedError

    def message(self, operation: str) -> str:
        default = {
            "submit": "Failed to submit {} batch",
            "status": "Failed to get {} batch status",
            "retrieve": "Failed to retrieve {} results",
            "download": "Failed to download {} results",
            "cancel": "Failed to cancel {} batch",
            "delete": "Failed to delete {} batch",
            "list": "Failed to list {} batches",
        }
        return self.messages.get(operation, default[operation]).format(self.label)


class OpenAIHarness(Harness):
    name = "openai"
    label = "OpenAI"
    running = "in_progress"
    upload_call = "files.create"
    sdk_calls = {
        "submit": "batches.create",
        "status": "batches.retrieve",
        "retrieve": "files.content",
        "download": "files.content",
        "cancel": "batches.cancel",
        "delete": "batches.retrieve",
        "list": "batches.list",
    }

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        super().__init__(monkeypatch)
        self.batches = [openai_batch()]
        self.text = '{"ok": true}\n'
        self.upload_names: list[str | None] = []
        self.created: list[dict[str, Any]] = []
        self.waits: list[float] = []
        client = SimpleNamespace(
            files=SimpleNamespace(create=self.files_create, content=self.content),
            batches=SimpleNamespace(
                create=self.batches_create,
                retrieve=self.retrieve,
                cancel=self.cancel,
                list=self.list_batches,
            ),
        )
        monkeypatch.setattr(openai, "OpenAI", lambda: client)
        monkeypatch.setattr("time.sleep", self.waits.append)
        self.provider = OpenAIProvider()

    def files_create(self, *, file: Any, purpose: str) -> FileObject:
        self.record("files.create", purpose)
        name, handle = file if isinstance(file, tuple) else (None, file)
        self.upload_names.append(name)
        self.upload(handle.read())
        return FileObject(
            id="file_input",
            bytes=1,
            created_at=EPOCH,
            filename="batch.jsonl",
            object="file",
            purpose="batch",
            status="processed",
        )

    def content(self, file_id: str) -> SimpleNamespace:
        self.record("files.content", file_id)
        return SimpleNamespace(text=self.text)

    def batches_create(self, **kwargs: Any) -> Batch:
        self.record("batches.create", kwargs)
        self.created.append(kwargs)
        return openai_batch(batch_id="batch_created", status="validating")

    def retrieve(self, batch_id: str) -> Batch:
        self.record("batches.retrieve", batch_id)
        index = min(len(self.called("batches.retrieve")), len(self.batches)) - 1
        return self.batches[index]

    def cancel(self, batch_id: str) -> Batch:
        self.record("batches.cancel", batch_id)
        return openai_batch(batch_id=batch_id, status="cancelled")

    def list_batches(self, *, limit: int) -> SimpleNamespace:
        self.record("batches.list", limit)
        return SimpleNamespace(data=self.batches)

    def set_running(self) -> None:
        self.batches = [openai_batch(status=self.running)]

    def serve(self, texts: list[str]) -> tuple[str, str]:
        self.text = "".join(f"{text}\n" for text in texts)
        return self.text, self.text

    def interrupt_results(self) -> None:
        # A lone surrogate cannot be encoded, so the write fails mid-download.
        self.text = "new\ud800"


class AnthropicHarness(Harness):
    name = "anthropic"
    label = "Anthropic"
    running = "in_progress"
    upload_call = "batches.create"
    sdk_calls = {
        "submit": "batches.create",
        "status": "batches.retrieve",
        "retrieve": "batches.results",
        "download": "batches.results",
        "cancel": "batches.cancel",
        "delete": "batches.retrieve",
        "list": "batches.list",
    }

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, beta: bool = False) -> None:
        super().__init__(monkeypatch)
        self.batch: Any = anthropic_batch()
        self.listed: list[MessageBatch] = []
        self.results: Callable[[], Iterator[MessageBatchIndividualResponse]] = (
            lambda: iter([anthropic_result("one", "answer")])
        )
        self.closed: list[bool] = []
        batches = SimpleNamespace(
            create=self.create,
            retrieve=self.retrieve,
            results=self.stream,
            cancel=self.cancel,
            list=self.list_batches,
        )
        if beta:
            client = SimpleNamespace(
                messages=SimpleNamespace(),
                beta=SimpleNamespace(messages=SimpleNamespace(batches=batches)),
            )
        else:
            client = SimpleNamespace(messages=SimpleNamespace(batches=batches))
        client.close = lambda: self.closed.append(True)
        monkeypatch.setattr(anthropic, "Anthropic", lambda: client)
        self.provider = AnthropicProvider()

    def create(self, *, requests: list[dict[str, Any]]) -> MessageBatch:
        self.record("batches.create", requests)
        self.uploaded.extend(requests)
        return anthropic_batch(batch_id="batch_created", status="in_progress")

    def retrieve(self, batch_id: str) -> Any:
        self.record("batches.retrieve", batch_id)
        return self.batch

    def stream(self, batch_id: str) -> Iterator[MessageBatchIndividualResponse]:
        self.record("batches.results", batch_id)
        return self.results()

    def cancel(self, batch_id: str) -> MessageBatch:
        self.record("batches.cancel", batch_id)
        return self.batch

    def list_batches(self, *, limit: int) -> SimpleNamespace:
        self.record("batches.list", limit)
        return SimpleNamespace(data=self.listed)

    def set_running(self) -> None:
        self.batch = anthropic_batch(status=self.running, processing=2)

    def serve(self, texts: list[str]) -> tuple[str, str]:
        records = [anthropic_result(str(i), text) for i, text in enumerate(texts)]
        self.results = lambda: iter(records)
        lines = [record.model_dump_json() for record in records]
        return "\n".join(lines), "".join(f"{line}\n" for line in lines)

    def interrupt_results(self) -> None:
        def interrupted() -> Iterator[MessageBatchIndividualResponse]:
            yield anthropic_result("one", "partial")
            raise OSError("stream interrupted")

        self.results = interrupted


class MistralHarness(Harness):
    name = "mistral"
    label = "Mistral"
    running = "RUNNING"
    upload_call = "files.upload"
    submit_kwargs = {"model": "mistral-small-latest"}
    # Downloads reuse retrieve_results, so they report retrieval failures.
    messages = {"download": "Failed to retrieve {} results"}
    sdk_calls = {
        "submit": "jobs.create",
        "status": "jobs.get",
        "retrieve": "files.download",
        "download": "files.download",
        "cancel": "jobs.cancel",
        "delete": "jobs.delete",
        "list": "jobs.list",
    }

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, *, supports_delete: bool = True
    ) -> None:
        super().__init__(monkeypatch)
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        self.job = mistral_job()
        self.listed: list[BatchJob] = []
        self.payload = b'{"ok": true}\n'
        # Per-file payloads; files not listed serve ``payload``.
        self.files: dict[str, bytes] = {}
        self.read: Callable[[], bytes] = lambda: self.payload
        self.created: dict[str, Any] = {}
        self.api_keys: list[str | None] = []
        jobs = SimpleNamespace(
            create=self.create, get=self.get, cancel=self.cancel, list=self.list_jobs
        )
        if supports_delete:
            jobs.delete = self.delete
        self.client = SimpleNamespace(
            batch=SimpleNamespace(jobs=jobs),
            files=SimpleNamespace(upload=self.files_upload, download=self.download),
        )

        def factory(*, api_key: str | None = None) -> SimpleNamespace:
            self.api_keys.append(api_key)
            return self.client

        monkeypatch.setattr(
            mistral_module, "_load_mistral_client_type", lambda: factory
        )
        self.provider = MistralProvider()

    def files_upload(self, *, file: dict[str, Any], purpose: str) -> SimpleNamespace:
        self.record("files.upload", (file["file_name"], purpose))
        self.upload(file["content"].read())
        return SimpleNamespace(id="file_in")

    def download(self, *, file_id: str) -> SimpleNamespace:
        self.record("files.download", file_id)
        if file_id in self.files:
            return SimpleNamespace(read=lambda: self.files[file_id])
        return SimpleNamespace(read=self.read)

    def create(self, **kwargs: Any) -> BatchJob:
        self.record("jobs.create", kwargs)
        self.created = kwargs
        return self.job

    def get(self, *, job_id: str) -> BatchJob:
        self.record("jobs.get", job_id)
        return self.job

    def cancel(self, *, job_id: str) -> BatchJob:
        self.record("jobs.cancel", job_id)
        return mistral_job(job_id=job_id, status="CANCELLATION_REQUESTED")

    def delete(self, *, job_id: str) -> DeleteBatchJobResponse:
        self.record("jobs.delete", job_id)
        return DeleteBatchJobResponse(id=job_id, object="batch", deleted=True)

    def list_jobs(self, **kwargs: Any) -> ListBatchJobsResponse:
        self.record("jobs.list", kwargs)
        return ListBatchJobsResponse(total=len(self.listed), data=self.listed)

    def set_running(self) -> None:
        self.job = mistral_job(status=self.running)

    def serve(self, texts: list[str]) -> tuple[str, str]:
        text = "".join(f"{line}\n" for line in texts)
        self.payload = text.encode("utf-8")
        return text, text

    def interrupt_results(self) -> None:
        def interrupted() -> bytes:
            raise OSError("download interrupted")

        self.read = interrupted


HARNESSES: dict[str, type[Harness]] = {
    "openai": OpenAIHarness,
    "anthropic": AnthropicHarness,
    "mistral": MistralHarness,
}
OPERATIONS = ("submit", "status", "retrieve", "download", "cancel", "delete", "list")


@pytest.fixture(params=list(HARNESSES))
def harness(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Harness:
    return HARNESSES[request.param](monkeypatch)


def run(harness: Harness, operation: str, destination: Path) -> Any:
    provider = harness.provider
    actions: dict[str, Callable[[], Any]] = {
        "submit": lambda: provider.submit_batch(
            io.BytesIO(PAYLOAD), **harness.submit_kwargs
        ),
        "status": lambda: provider.get_status("batch_123"),
        "retrieve": lambda: provider.retrieve_results("batch_123"),
        "download": lambda: provider.download_results("batch_123", str(destination)),
        "cancel": lambda: provider.cancel_batch("batch_123"),
        "delete": lambda: provider.delete_batch("batch_123"),
        "list": lambda: provider.list_batches(),
    }
    return actions[operation]()


# --- shared contract ----------------------------------------------------------


def test_batch_provider_requires_every_operation() -> None:
    assert BatchProvider.__abstractmethods__ == {
        "submit_batch",
        "get_status",
        "retrieve_results",
        "download_results",
        "cancel_batch",
        "delete_batch",
        "list_batches",
    }
    with pytest.raises(TypeError, match="abstract methods"):
        BatchProvider()  # ty: ignore[call-non-callable]


@pytest.mark.parametrize("operation", OPERATIONS)
def test_operation_wraps_sdk_failure(
    harness: Harness,
    operation: str,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    failure = ConnectionError("service unavailable")
    harness.failures[harness.sdk_calls[operation]] = failure
    destination = tmp_path / "results.jsonl"

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(Exception, match=harness.message(operation)) as caught:
            run(harness, operation, destination)

    assert caught.value.__cause__ is failure
    assert "service unavailable" in str(caught.value)
    if operation == "submit":
        assert type(caught.value) is RuntimeError
        # Logs name the failure type without echoing provider messages.
        assert "service unavailable" not in caplog.text
    assert not destination.exists()


@pytest.mark.parametrize("error", [ValueError("bad input"), TypeError("bad file")])
def test_submit_preserves_validation_errors(harness: Harness, error: Exception) -> None:
    harness.failures[harness.upload_call] = error

    with pytest.raises(type(error)) as caught:
        run(harness, "submit", Path())

    assert caught.value is error
    assert harness.called(harness.sdk_calls["submit"]) == []


def test_submit_rejects_unsupported_input(harness: Harness) -> None:
    with pytest.raises(ValueError, match="Unsupported file_path_or_buffer type"):
        harness.provider.submit_batch(cast(Any, 123), **harness.submit_kwargs)


@pytest.mark.parametrize("source", ["path", "buffer"])
def test_submit_uploads_path_or_rewound_buffer(
    harness: Harness, source: str, tmp_path: Path
) -> None:
    path = tmp_path / "requests.jsonl"
    path.write_bytes(PAYLOAD)
    buffer = io.BytesIO(PAYLOAD)
    buffer.seek(5)

    batch_id = harness.provider.submit_batch(
        str(path) if source == "path" else buffer, **harness.submit_kwargs
    )

    assert batch_id in {"batch_created", "batch_123"}
    assert harness.uploaded == [{"custom_id": "request-0"}]


@pytest.mark.parametrize("operation", ["retrieve", "download"])
def test_results_reject_running_batch(
    harness: Harness, operation: str, tmp_path: Path
) -> None:
    harness.set_running()
    destination = tmp_path / "results.jsonl"

    with pytest.raises(
        Exception, match=f"Batch not completed, status: {harness.running}"
    ):
        run(harness, operation, destination)

    assert harness.called(harness.sdk_calls[operation]) == []
    assert not destination.exists()


@pytest.mark.parametrize("operation", ["retrieve", "download"])
def test_results_return_every_record(
    harness: Harness, operation: str, tmp_path: Path
) -> None:
    retrieved, downloaded = harness.serve(['{"name": "Ada"}', '{"name": "Lin"}'])
    destination = tmp_path / "results.jsonl"

    result = run(harness, operation, destination)

    if operation == "retrieve":
        assert result == retrieved
    else:
        assert result is None
        assert destination.read_text(encoding="utf-8") == downloaded
    assert len(harness.called(harness.sdk_calls[operation])) == 1


def test_download_writes_utf8_when_platform_default_is_ascii(
    harness: Harness, tmp_path: Path
) -> None:
    harness.monkeypatch.setattr(
        f"instructor.batch.providers.{harness.name}.open",
        partial(open, encoding="ascii"),
        raising=False,
    )
    _, downloaded = harness.serve(['{"response": "Café ☕ 東京"}'])
    destination = tmp_path / "results.jsonl"

    run(harness, "download", destination)

    written = destination.read_text(encoding="utf-8")
    assert written == downloaded
    assert "Café ☕ 東京" in written


@pytest.mark.parametrize("fails", [False, True])
def test_download_replaces_destination_only_after_full_write(
    harness: Harness, fails: bool, tmp_path: Path
) -> None:
    _, downloaded = harness.serve(['{"new": true}'])
    if fails:
        harness.interrupt_results()
    destination = tmp_path / "results.jsonl"
    destination.write_text("previous\n", encoding="utf-8")

    if fails:
        with pytest.raises(Exception):  # noqa: B017 - each provider fails differently
            run(harness, "download", destination)
    else:
        run(harness, "download", destination)

    expected = "previous\n" if fails else downloaded
    assert destination.read_text(encoding="utf-8") == expected
    assert list(tmp_path.iterdir()) == [destination]


# --- provider registry --------------------------------------------------------


@pytest.fixture
def restore_registry() -> Iterator[None]:
    yield
    importlib.reload(batch_providers)


@pytest.mark.usefixtures("restore_registry")
@pytest.mark.parametrize(
    ("missing", "unavailable", "message"),
    [
        ("mistralai", "mistral", "Mistral is not installed"),
        ("anthropic", "anthropic", "Anthropic is not installed"),
        ("openai", "openai", "OpenAI is not installed"),
    ],
)
def test_missing_sdk_disables_only_its_own_provider(
    monkeypatch: pytest.MonkeyPatch, missing: str, unavailable: str, message: str
) -> None:
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args, **kwargs: None
        if name == missing
        else real_find_spec(name, *args, **kwargs),
    )
    module = importlib.reload(batch_providers)

    with pytest.raises(ValueError, match=message):
        module.get_provider(unavailable)
    for name in set(HARNESSES) - {unavailable}:
        assert isinstance(module.get_provider(name), module.BatchProvider)


def test_registry_exports_providers_and_rejects_unknown_names() -> None:
    assert "MistralProvider" in batch_providers.__all__
    for name, provider_type in [
        ("openai", OpenAIProvider),
        ("anthropic", AnthropicProvider),
        ("mistral", MistralProvider),
    ]:
        assert isinstance(batch_providers.get_provider(name), provider_type)
    with pytest.raises(ValueError, match="Unsupported provider: gemini"):
        batch_providers.get_provider("gemini")


# --- OpenAI-specific ----------------------------------------------------------


@pytest.fixture
def openai_harness(monkeypatch: pytest.MonkeyPatch) -> OpenAIHarness:
    return OpenAIHarness(monkeypatch)


def test_openai_submit_options_and_upload_names(
    openai_harness: OpenAIHarness, tmp_path: Path
) -> None:
    path = tmp_path / "requests.jsonl"
    path.write_bytes(PAYLOAD)
    provider = openai_harness.provider

    provider.submit_batch(str(path))
    provider.submit_batch(
        io.BytesIO(PAYLOAD), metadata={"source": "daily"}, completion_window="48h"
    )

    assert openai_harness.upload_names == [None, "batch.jsonl"]
    assert openai_harness.called("files.create") == ["batch", "batch"]
    assert openai_harness.created == [
        {
            "input_file_id": "file_input",
            "endpoint": "/v1/chat/completions",
            "completion_window": window,
            "metadata": metadata,
        }
        for window, metadata in [
            ("24h", {"description": "Instructor batch job"}),
            ("48h", {"source": "daily"}),
        ]
    ]


def test_openai_status_cancel_delete_and_list_map_sdk_results(
    openai_harness: OpenAIHarness,
) -> None:
    openai_harness.batches = [openai_batch(status="in_progress", counts=(4, 4, 0))]
    provider = openai_harness.provider

    assert provider.get_status("batch_123") == {
        "id": "batch_123",
        "status": "in_progress",
        "created_at": EPOCH,
        "request_counts": {"total": 4, "completed": 4, "failed": 0},
    }
    openai_harness.batches = [openai_batch(counts=None)]
    assert provider.get_status("batch_123")["request_counts"] == {
        "total": 0,
        "completed": 0,
        "failed": 0,
    }
    cancelled = provider.cancel_batch("batch_cancel")
    assert (cancelled["id"], cancelled["status"]) == ("batch_cancel", "cancelled")
    assert provider.delete_batch("batch_123") == {
        "id": "batch_123",
        "status": "completed",
        "message": "OpenAI does not support batch deletion",
    }
    jobs = provider.list_batches(limit=3)
    assert openai_harness.called("batches.list") == [3]
    assert [(job.provider, job.status) for job in jobs] == [
        ("openai", BatchStatus.COMPLETED)
    ]
    assert jobs[0].files.output_file_id == "file_output"


@pytest.mark.parametrize("operation", ["retrieve", "download"])
@pytest.mark.parametrize("status", ["completed", "expired", "cancelled"])
@pytest.mark.parametrize("delayed", [False, True])
def test_openai_combines_output_and_error_files(
    openai_harness: OpenAIHarness,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    operation: str,
    status: str,
    delayed: bool,
) -> None:
    ready = openai_batch(status=status, error="file_errors", counts=(2, 1, 1))
    openai_harness.batches = [ready]
    if delayed:
        waiting = openai_batch(status=status, output=None, counts=(2, 1, 1))
        openai_harness.batches.insert(0, waiting)

    destination = tmp_path / "results.jsonl"

    result = run(openai_harness, operation, destination)

    if operation == "download":
        result = destination.read_text()
    # Each file is JSONL; joining them keeps one record per line.
    assert result == '{"ok": true}\n{"ok": true}'
    assert openai_harness.called("files.content") == ["file_output", "file_errors"]
    assert openai_harness.waits == ([5] if delayed else [])
    output = capsys.readouterr().out
    assert ("waiting 5s (attempt 1/10)" in output) is delayed
    assert ("Result files now available" in output) is delayed


def test_openai_reads_error_file_when_every_request_failed(
    openai_harness: OpenAIHarness,
) -> None:
    openai_harness.batches = [
        openai_batch(output=None, error="file_error", counts=(3, 0, 3))
    ]

    assert openai_harness.provider.retrieve_results("batch_failed") == '{"ok": true}\n'
    assert openai_harness.called("files.content") == ["file_error"]


@pytest.mark.parametrize("status", ["cancelled", "expired"])
def test_openai_stopped_batch_without_finished_requests_is_empty(
    openai_harness: OpenAIHarness, status: str
) -> None:
    openai_harness.batches = [
        openai_batch(status=status, output=None, counts=(5, 0, 0))
    ]

    assert openai_harness.provider.retrieve_results("batch_stopped") == ""
    assert openai_harness.waits == []
    assert openai_harness.called("files.content") == []


@pytest.mark.parametrize("operation", ["retrieve", "download"])
@pytest.mark.parametrize(
    ("later", "message", "waits"),
    [
        ("failed", "Batch status changed to failed", [5]),
        ("completed", "No result files available after 10 retries", list(range(5, 15))),
    ],
)
def test_openai_stops_waiting_for_files_that_never_arrive(
    openai_harness: OpenAIHarness,
    tmp_path: Path,
    operation: str,
    later: str,
    message: str,
    waits: list[int],
) -> None:
    openai_harness.batches = [
        openai_batch(output=None),
        openai_batch(status=later, output=None),
    ]
    destination = tmp_path / "results.jsonl"

    with pytest.raises(Exception, match=message):
        run(openai_harness, operation, destination)

    assert openai_harness.waits == waits
    assert len(openai_harness.called("batches.retrieve")) == len(waits) + 1
    assert openai_harness.called("files.content") == []
    assert not destination.exists()


# --- Anthropic-specific -------------------------------------------------------


def test_anthropic_submit_parses_jsonl_and_reports_ignored_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    harness = AnthropicHarness(monkeypatch)
    path = tmp_path / "requests.jsonl"
    path.write_text('{"custom_id": "a"}\n  \n{"custom_id": "b"}\n')

    harness.provider.submit_batch(str(path), metadata={"source": "nightly"})

    assert harness.uploaded == [{"custom_id": "a"}, {"custom_id": "b"}]
    assert "Ignoring: {'source': 'nightly'}" in capsys.readouterr().out
    with pytest.raises(ValueError, match="Expecting value"):
        harness.provider.submit_batch(io.BytesIO(b"not-json\n"))


@pytest.mark.parametrize("beta", [False, True])
def test_anthropic_operations_use_stable_or_beta_batches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, beta: bool
) -> None:
    harness = AnthropicHarness(monkeypatch, beta=beta)
    batch = anthropic_batch(status="in_progress", succeeded=1, processing=2)
    harness.batch = batch
    harness.listed = [anthropic_batch(batch_id="batch_done"), batch]
    provider = harness.provider

    assert provider.submit_batch(io.BytesIO(PAYLOAD)) == "batch_created"
    assert provider.get_status("batch_123") == {
        "id": "batch_123",
        "status": "in_progress",
        "created_at": CREATED,
        "request_counts": batch.request_counts,
    }
    assert provider.cancel_batch("batch_123") == batch.model_dump()
    assert provider.delete_batch("batch_123") == {
        "id": "batch_123",
        "status": "in_progress",
        "message": "Anthropic does not support batch deletion",
    }
    jobs = provider.list_batches(limit=2)
    assert [(job.id, job.status, job.raw_status) for job in jobs] == [
        ("batch_done", BatchStatus.COMPLETED, "ended"),
        ("batch_123", BatchStatus.PROCESSING, "in_progress"),
    ]
    assert jobs[1].request_counts.total == 3
    assert str(jobs[0].files.results_url).endswith("/results")
    harness.batch = anthropic_batch()
    retrieved, downloaded = harness.serve(["First", "Second"])
    assert provider.retrieve_results("batch_123") == retrieved
    provider.download_results("batch_123", str(tmp_path / "out.jsonl"))
    assert (tmp_path / "out.jsonl").read_text() == downloaded
    assert harness.called("batches.list") == [2]


@pytest.mark.parametrize("operation", ["retrieve", "download"])
@pytest.mark.parametrize("status", ["failed", "cancelled", "expired"])
def test_anthropic_results_reject_terminal_failure_states(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, operation: str, status: str
) -> None:
    harness = AnthropicHarness(monkeypatch)
    harness.batch = SimpleNamespace(id="batch_123", processing_status=status)
    destination = tmp_path / "results.jsonl"

    with pytest.raises(Exception, match=f"Batch job failed with status: {status}"):
        run(harness, operation, destination)

    assert not destination.exists()


@pytest.mark.parametrize("operation", ["retrieve", "download"])
@pytest.mark.parametrize("counts", [None, (0, 3)], ids=["older-batch", "all-errored"])
def test_anthropic_reads_ended_batches_with_any_counts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    operation: str,
    counts: tuple[int, int] | None,
) -> None:
    harness = AnthropicHarness(monkeypatch)
    harness.batch = SimpleNamespace(
        id="older_batch", processing_status="ended", created_at=CREATED
    )
    if counts is not None:
        harness.batch.request_counts = SimpleNamespace(
            succeeded=counts[0], errored=counts[1]
        )
    record = anthropic_errored("failed_request")
    harness.results = lambda: iter([record])
    destination = tmp_path / "results.jsonl"

    result = run(harness, operation, destination)

    expected = record.model_dump_json()
    if operation == "retrieve":
        assert result == expected
    else:
        assert destination.read_text() == expected + "\n"
    assert harness.called("batches.results") == ["batch_123"]
    if counts is None:
        status = harness.provider.get_status("older_batch")
        assert status["request_counts"] == {}


def test_anthropic_result_stream_keeps_its_sdk_client_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = AnthropicHarness(monkeypatch)

    class OwnedClient:
        def __init__(self) -> None:
            # The results stream must not keep the client alive by itself.
            owner = weakref.ref(self)

            def results(_batch_id: str) -> Iterator[MessageBatchIndividualResponse]:
                gc.collect()
                assert owner() is not None and not harness.closed
                yield anthropic_result("one", "streamed")

            batches = SimpleNamespace(retrieve=harness.retrieve, results=results)
            self.messages = SimpleNamespace(batches=batches)

        def close(self) -> None:
            harness.closed.append(True)

    monkeypatch.setattr(anthropic, "Anthropic", OwnedClient)

    result = harness.provider.retrieve_results("batch_123")

    assert json.loads(result)["custom_id"] == "one"
    assert harness.closed == [True]


@pytest.mark.parametrize("command", ["download_file", "results"])
@pytest.mark.parametrize("stream_fails", [False, True])
def test_cli_anthropic_export_replaces_file_only_after_every_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, command: str, stream_fails: bool
) -> None:
    from instructor.cli import batch as batch_cli

    harness = AnthropicHarness(monkeypatch)
    if stream_fails:
        harness.interrupt_results()
    else:
        harness.results = lambda: iter(
            [anthropic_result("one", "a"), anthropic_result("two", "b")]
        )
    destination = tmp_path / "results.jsonl"
    destination.write_text("previous\n", encoding="utf-8")

    if command == "download_file":
        batch_cli.download_file(
            batch_id="batch_123",
            download_file_path=str(destination),
            provider="anthropic",
        )
    else:
        batch_cli.results(
            batch_id="batch_123",
            output_file=str(destination),
            model="anthropic/claude-sonnet-4-5",
        )

    lines = destination.read_text(encoding="utf-8").splitlines()
    if stream_fails:
        assert lines == ["previous"]
    else:
        assert [json.loads(line)["custom_id"] for line in lines] == ["one", "two"]
    assert list(tmp_path.iterdir()) == [destination]


# --- Mistral-specific ---------------------------------------------------------


@pytest.fixture
def mistral_harness(monkeypatch: pytest.MonkeyPatch) -> MistralHarness:
    return MistralHarness(monkeypatch)


def test_mistral_client_needs_sdk_and_api_key(
    mistral_harness: MistralHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    mistral_harness.provider.get_status("batch_123")
    assert mistral_harness.api_keys == ["test-key"]

    monkeypatch.delenv("MISTRAL_API_KEY")
    with pytest.raises(RuntimeError, match="MISTRAL_API_KEY"):
        mistral_harness.provider.get_status("batch_123")

    monkeypatch.setattr(mistral_module, "_load_mistral_client_type", lambda: None)
    with pytest.raises(RuntimeError, match="require the mistralai package"):
        mistral_harness.provider.get_status("batch_123")


def test_mistral_client_type_resolves_installed_sdk() -> None:
    client_type = mistral_module._load_mistral_client_type()

    assert client_type is not None
    assert client_type.__name__ == "Mistral"


def test_mistral_submit_creates_job_with_model_and_endpoint(
    mistral_harness: MistralHarness,
) -> None:
    provider = mistral_harness.provider

    provider.submit_batch(
        io.BytesIO(PAYLOAD), metadata={"description": "job"}, model="mistral-small"
    )
    first = mistral_harness.created
    provider.submit_batch(
        io.BytesIO(PAYLOAD), model="mistral-large", endpoint="/v1/embeddings"
    )

    assert mistral_harness.called("files.upload") == [("batch.jsonl", "batch")] * 2
    assert first == {
        "input_files": ["file_in"],
        "model": "mistral-small",
        "endpoint": "/v1/chat/completions",
        "metadata": {"description": "job"},
    }
    assert mistral_harness.created["endpoint"] == "/v1/embeddings"
    assert mistral_harness.created["metadata"] == {}
    with pytest.raises(ValueError, match="require a model"):
        provider.submit_batch("batch_requests.jsonl")


@pytest.mark.parametrize("explicit", [None, "mistral-large-latest"])
def test_mistral_processor_submits_with_its_model_unless_overridden(
    mistral_harness: MistralHarness, explicit: str | None
) -> None:
    class Person(BaseModel):
        name: str

    processor = BatchProcessor("mistral/mistral-small-latest", Person)
    if explicit:
        processor.submit_batch(io.BytesIO(PAYLOAD), model=explicit)
    else:
        processor.submit_batch(io.BytesIO(PAYLOAD))

    assert mistral_harness.created["model"] == (explicit or "mistral-small-latest")


def test_mistral_status_cancel_delete_and_list(
    mistral_harness: MistralHarness,
) -> None:
    mistral_harness.job = mistral_job(status="RUNNING", succeeded=1, failed=1)
    mistral_harness.listed = [mistral_job(job_id=f"batch_{i}") for i in range(5)]
    provider = mistral_harness.provider

    assert provider.get_status("batch_123") == {
        "id": "batch_123",
        "status": "RUNNING",
        "created_at": EPOCH,
        "request_counts": {"total": 2, "succeeded": 1, "failed": 1},
    }
    assert provider.cancel_batch("batch_123")["status"] == "CANCELLATION_REQUESTED"
    assert provider.delete_batch("batch_123") == {
        "id": "batch_123",
        "object": "batch",
        "deleted": True,
    }
    jobs = provider.list_batches(limit=2)
    assert mistral_harness.called("jobs.list") == [{"page_size": 2}]
    assert [(job.id, job.provider) for job in jobs] == [
        ("batch_0", "mistral"),
        ("batch_1", "mistral"),
    ]
    mistral_harness.listed = []
    assert provider.list_batches() == []


def test_mistral_delete_requires_sdk2(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = MistralHarness(monkeypatch, supports_delete=False)

    with pytest.raises(NotImplementedError, match="requires mistralai>=2.0.0"):
        harness.provider.delete_batch("batch_123")


def mistral_record(custom_id: str, **fields: Any) -> str:
    return json.dumps({"id": f"result_{custom_id}", "custom_id": custom_id, **fields})


MISTRAL_SUCCESS = mistral_record(
    "ok",
    response={
        "status_code": 200,
        "body": {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": '{"name": "Ada"}',
                    },
                    "finish_reason": "stop",
                }
            ]
        },
    },
    error=None,
)
MISTRAL_FAILURE = mistral_record(
    "bad",
    response={
        "status_code": 400,
        "body": {"error": {"type": "invalid_request_error", "message": "Bad input"}},
    },
    error=None,
)


@pytest.mark.parametrize("operation", ["retrieve", "download"])
@pytest.mark.parametrize("status", ["SUCCESS", "TIMEOUT_EXCEEDED", "CANCELLED"])
def test_mistral_combines_output_and_error_files(
    mistral_harness: MistralHarness, tmp_path: Path, operation: str, status: str
) -> None:
    mistral_harness.job = mistral_job(
        status=status, succeeded=1, failed=1, error="file_err"
    )
    # The output file has no trailing newline; the merge must still split lines.
    mistral_harness.files = {
        "file_out": MISTRAL_SUCCESS.encode(),
        "file_err": f"{MISTRAL_FAILURE}\n".encode(),
    }
    destination = tmp_path / "results.jsonl"

    result = run(mistral_harness, operation, destination)

    if operation == "download":
        result = destination.read_text(encoding="utf-8")
    assert result == f"{MISTRAL_SUCCESS}\n{MISTRAL_FAILURE}"
    assert mistral_harness.called("files.download") == ["file_out", "file_err"]


@pytest.mark.parametrize("status", ["SUCCESS", "FAILED"])
def test_mistral_returns_per_request_errors_when_every_request_failed(
    mistral_harness: MistralHarness, status: str
) -> None:
    class Person(BaseModel):
        name: str

    mistral_harness.job = mistral_job(
        status=status, succeeded=0, failed=1, output=None, error="file_err"
    )
    mistral_harness.files = {"file_err": f"{MISTRAL_FAILURE}\n".encode()}

    text = mistral_harness.provider.retrieve_results("batch_123")
    [failure] = BatchProcessor("mistral/mistral-small-latest", Person).parse_results(
        text
    )

    assert text == f"{MISTRAL_FAILURE}\n"
    assert mistral_harness.called("files.download") == ["file_err"]
    assert isinstance(failure, BatchError)
    assert (failure.custom_id, failure.error_type, failure.error_message) == (
        "bad",
        "invalid_request_error",
        "Bad input",
    )


def test_mistral_merged_results_parse_into_successes_and_errors(
    mistral_harness: MistralHarness,
) -> None:
    class Person(BaseModel):
        name: str

    mistral_harness.job = mistral_job(succeeded=1, failed=1, error="file_err")
    mistral_harness.files = {
        "file_out": f"{MISTRAL_SUCCESS}\n".encode(),
        "file_err": f"{MISTRAL_FAILURE}\n".encode(),
    }

    text = mistral_harness.provider.retrieve_results("batch_123")
    success, failure = BatchProcessor(
        "mistral/mistral-small-latest", Person
    ).parse_results(text)

    assert success == BatchSuccess(custom_id="ok", result=Person(name="Ada"))
    assert isinstance(failure, BatchError)
    assert (failure.custom_id, failure.error_message) == ("bad", "Bad input")


@pytest.mark.parametrize("status", ["CANCELLED", "TIMEOUT_EXCEEDED"])
def test_mistral_stopped_job_without_finished_requests_is_empty(
    mistral_harness: MistralHarness, status: str
) -> None:
    mistral_harness.job = mistral_job(
        status=status, succeeded=0, output=None, completed=0
    )

    assert mistral_harness.provider.retrieve_results("batch_123") == ""
    assert mistral_harness.called("files.download") == []


@pytest.mark.parametrize(
    ("job", "message"),
    [
        (
            mistral_job(status="QUEUED"),
            "Batch not completed, status: QUEUED",
        ),
        (
            mistral_job(status="CANCELLATION_REQUESTED"),
            "Batch not completed, status: CANCELLATION_REQUESTED",
        ),
        (
            mistral_job(status="FAILED", output=None),
            "no output or error file available, status: FAILED",
        ),
        (
            mistral_job(output=None),
            "no output or error file available, status: SUCCESS",
        ),
        (
            mistral_job(status="CANCELLED", succeeded=1, output=None),
            "no output or error file available, status: CANCELLED",
        ),
    ],
)
def test_mistral_results_refuse_jobs_without_usable_output(
    mistral_harness: MistralHarness, job: BatchJob, message: str
) -> None:
    mistral_harness.job = job

    with pytest.raises(RuntimeError, match=message):
        mistral_harness.provider.retrieve_results("batch_123")
    assert mistral_harness.called("files.download") == []
