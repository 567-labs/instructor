"""Behavior tests for the Anthropic batch provider."""

import io
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import anthropic
import pytest
from anthropic.types import Message, TextBlock, Usage
from anthropic.types.messages import (
    MessageBatch,
    MessageBatchIndividualResponse,
    MessageBatchRequestCounts,
    MessageBatchSucceededResult,
)

from instructor.batch.models import BatchStatus
from instructor.batch.providers.anthropic import AnthropicProvider

pytestmark = pytest.mark.unit


CREATED_AT = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def make_batch(
    batch_id="batch_123", status="ended", succeeded=2, errored=0, processing=0
):
    return MessageBatch(
        id=batch_id,
        archived_at=None,
        cancel_initiated_at=None,
        created_at=CREATED_AT,
        ended_at=CREATED_AT + timedelta(minutes=1) if status == "ended" else None,
        expires_at=CREATED_AT + timedelta(days=1),
        processing_status=status,
        request_counts=MessageBatchRequestCounts(
            canceled=0,
            errored=errored,
            expired=0,
            processing=processing,
            succeeded=succeeded,
        ),
        results_url="https://api.anthropic.com/v1/messages/batches/batch_123/results",
        type="message_batch",
    )


def make_result(custom_id, text):
    return MessageBatchIndividualResponse(
        custom_id=custom_id,
        result=MessageBatchSucceededResult(
            type="succeeded",
            message=Message(
                id=f"msg_{custom_id}",
                content=[TextBlock(type="text", text=text, citations=None)],
                model="claude-sonnet-4-5",
                role="assistant",
                stop_reason="end_turn",
                stop_sequence=None,
                type="message",
                usage=Usage(input_tokens=12, output_tokens=3),
            ),
        ),
    )


class RecordingBatches:
    def __init__(self, batch=None, results=(), listed=(), failures=None):
        self.batch = batch or make_batch()
        self.result_items = tuple(results)
        self.listed = tuple(listed)
        self.failures = failures or {}
        self.created_requests = None
        self.create_count = 0
        self.retrieved_ids = []
        self.result_ids = []
        self.cancelled_ids = []
        self.list_limits = []

    def _raise_if_needed(self, operation):
        if operation in self.failures:
            raise self.failures[operation]

    def create(self, *, requests):
        self._raise_if_needed("create")
        self.create_count += 1
        self.created_requests = requests
        return self.batch

    def retrieve(self, batch_id):
        self._raise_if_needed("retrieve")
        self.retrieved_ids.append(batch_id)
        return self.batch

    def results(self, batch_id):
        self._raise_if_needed("results")
        self.result_ids.append(batch_id)
        return iter(self.result_items)

    def cancel(self, batch_id):
        self._raise_if_needed("cancel")
        self.cancelled_ids.append(batch_id)
        return self.batch

    def list(self, *, limit):
        self._raise_if_needed("list")
        self.list_limits.append(limit)
        return SimpleNamespace(data=list(self.listed))


def install_client(monkeypatch, batches, *, beta=False):
    if beta:
        client = SimpleNamespace(
            messages=SimpleNamespace(),
            beta=SimpleNamespace(messages=SimpleNamespace(batches=batches)),
        )
    else:
        client = SimpleNamespace(messages=SimpleNamespace(batches=batches))
    monkeypatch.setattr(anthropic, "Anthropic", lambda: client)


def request(custom_id, content):
    return {
        "custom_id": custom_id,
        "params": {
            "model": "claude-sonnet-4-5",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": content}],
        },
    }


def test_submit_file_parses_jsonl_and_reports_ignored_metadata(
    monkeypatch, tmp_path, capsys
):
    batches = RecordingBatches(make_batch("batch_file"))
    install_client(monkeypatch, batches)
    first = request("first", "Summarize the first note")
    second = request("second", "Summarize the second note")
    path = tmp_path / "requests.jsonl"
    path.write_text(f"{json.dumps(first)}\n  \n{json.dumps(second)}\n")

    batch_id = AnthropicProvider().submit_batch(
        str(path), metadata={"source": "nightly"}, completion_window="24h"
    )

    assert batch_id == "batch_file"
    assert batches.created_requests == [first, second]
    assert batches.create_count == 1
    assert "Ignoring: {'source': 'nightly'}" in capsys.readouterr().out


def test_submit_bytesio_uses_beta_batches_and_rewinds_buffer(monkeypatch):
    batches = RecordingBatches(make_batch("batch_buffer"))
    install_client(monkeypatch, batches, beta=True)
    payload = request("buffered", "Extract the dates")
    buffer = io.BytesIO(f"\n{json.dumps(payload)}\n".encode())
    buffer.seek(4)

    assert AnthropicProvider().submit_batch(buffer) == "batch_buffer"
    assert batches.created_requests == [payload]
    assert batches.create_count == 1


@pytest.mark.parametrize(
    ("input_value", "message"),
    [
        (io.BytesIO(b"not-json\n"), "Expecting value"),
        (object(), "Unsupported file_path_or_buffer type"),
    ],
)
def test_submit_preserves_input_validation_errors(monkeypatch, input_value, message):
    install_client(monkeypatch, RecordingBatches())

    with pytest.raises(ValueError, match=message):
        AnthropicProvider().submit_batch(input_value)


def test_submit_wraps_sdk_failure(monkeypatch):
    batches = RecordingBatches(failures={"create": OSError("service unavailable")})
    install_client(monkeypatch, batches)

    with pytest.raises(RuntimeError, match="Failed to submit Anthropic batch") as exc:
        AnthropicProvider().submit_batch(io.BytesIO(b""))

    assert isinstance(exc.value.__cause__, OSError)


@pytest.mark.parametrize("beta", [False, True])
def test_get_status_returns_sdk_fields_and_supports_beta(monkeypatch, beta):
    batch = make_batch(status="in_progress", succeeded=1, processing=2)
    batches = RecordingBatches(batch)
    install_client(monkeypatch, batches, beta=beta)

    status = AnthropicProvider().get_status("batch_123")

    assert status == {
        "id": "batch_123",
        "status": "in_progress",
        "created_at": CREATED_AT,
        "request_counts": batch.request_counts,
    }
    assert batches.retrieved_ids == ["batch_123"]


def test_get_status_defaults_missing_request_counts(monkeypatch):
    batch = SimpleNamespace(
        id="older_batch", processing_status="ended", created_at=CREATED_AT
    )
    install_client(monkeypatch, RecordingBatches(batch))

    assert AnthropicProvider().get_status("older_batch")["request_counts"] == {}


def test_get_status_wraps_retrieve_failure(monkeypatch):
    batches = RecordingBatches(failures={"retrieve": ConnectionError("offline")})
    install_client(monkeypatch, batches)

    with pytest.raises(Exception, match="Failed to get Anthropic batch status") as exc:
        AnthropicProvider().get_status("batch_missing")

    assert isinstance(exc.value.__cause__, ConnectionError)


@pytest.mark.parametrize("beta", [False, True])
def test_retrieve_results_serializes_every_sdk_result(monkeypatch, beta):
    first = make_result("one", "First answer")
    second = make_result("two", "Second answer")
    batches = RecordingBatches(results=[first, second])
    install_client(monkeypatch, batches, beta=beta)

    output = AnthropicProvider().retrieve_results("batch_123")

    assert output == "\n".join([first.model_dump_json(), second.model_dump_json()])
    assert [json.loads(line)["custom_id"] for line in output.splitlines()] == [
        "one",
        "two",
    ]
    assert batches.result_ids == ["batch_123"]


@pytest.mark.parametrize("status", ["failed", "cancelled", "expired"])
def test_retrieve_results_rejects_terminal_failure_states(monkeypatch, status):
    batch = SimpleNamespace(id="batch_123", processing_status=status)
    install_client(monkeypatch, RecordingBatches(batch))

    with pytest.raises(Exception, match=f"Batch job failed with status: {status}"):
        AnthropicProvider().retrieve_results("batch_123")


def test_retrieve_results_rejects_in_progress_batch(monkeypatch):
    batch = make_batch(status="in_progress", processing=2)
    install_client(monkeypatch, RecordingBatches(batch))

    with pytest.raises(Exception, match="Batch not completed, status: in_progress"):
        AnthropicProvider().retrieve_results("batch_123")


def test_retrieve_results_reads_batch_where_every_request_errored(monkeypatch):
    batch = SimpleNamespace(
        id="batch_123",
        processing_status="ended",
        request_counts=SimpleNamespace(succeeded=0, errored=2, total=2),
    )
    result = MessageBatchIndividualResponse.model_validate(
        {
            "custom_id": "failed_request",
            "result": {
                "type": "errored",
                "error": {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Invalid model",
                    },
                },
            },
        }
    )
    batches = RecordingBatches(batch, results=[result])
    install_client(monkeypatch, batches)
    assert AnthropicProvider().retrieve_results("batch_123") == result.model_dump_json()
    assert batches.result_ids == ["batch_123"]


def test_retrieve_results_wraps_results_stream_failure(monkeypatch):
    batches = RecordingBatches(failures={"results": OSError("stream closed")})
    install_client(monkeypatch, batches)

    with pytest.raises(Exception, match="Failed to retrieve Anthropic results") as exc:
        AnthropicProvider().retrieve_results("batch_123")

    assert isinstance(exc.value.__cause__, OSError)


@pytest.mark.parametrize("beta", [False, True])
def test_download_results_writes_one_json_record_per_line(monkeypatch, tmp_path, beta):
    first = make_result("one", "First answer")
    second = make_result("two", "Second answer")
    batches = RecordingBatches(results=[first, second])
    install_client(monkeypatch, batches, beta=beta)
    path = tmp_path / "results.jsonl"

    AnthropicProvider().download_results("batch_123", str(path))

    assert path.read_text() == (
        f"{first.model_dump_json()}\n{second.model_dump_json()}\n"
    )
    assert batches.result_ids == ["batch_123"]


@pytest.mark.parametrize("status", ["failed", "cancelled", "expired"])
def test_download_results_rejects_terminal_failure_states(
    monkeypatch, tmp_path, status
):
    batch = SimpleNamespace(id="batch_123", processing_status=status)
    install_client(monkeypatch, RecordingBatches(batch))
    path = tmp_path / "results.jsonl"

    with pytest.raises(Exception, match=f"Batch job failed with status: {status}"):
        AnthropicProvider().download_results("batch_123", str(path))

    assert not path.exists()


def test_download_results_rejects_in_progress_batch(monkeypatch, tmp_path):
    batch = make_batch(status="in_progress", processing=1)
    install_client(monkeypatch, RecordingBatches(batch))
    path = tmp_path / "results.jsonl"

    with pytest.raises(Exception, match="Batch not completed, status: in_progress"):
        AnthropicProvider().download_results("batch_123", str(path))

    assert not path.exists()


def test_download_results_reads_batch_where_every_request_errored(
    monkeypatch, tmp_path
):
    batch = SimpleNamespace(
        id="batch_123",
        processing_status="ended",
        request_counts=SimpleNamespace(succeeded=0, errored=3, total=3),
    )
    result = MessageBatchIndividualResponse.model_validate(
        {
            "custom_id": "failed_request",
            "result": {
                "type": "errored",
                "error": {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Invalid model",
                    },
                },
            },
        }
    )
    batches = RecordingBatches(batch, results=[result])
    install_client(monkeypatch, batches)
    path = tmp_path / "results.jsonl"

    AnthropicProvider().download_results("batch_123", str(path))
    assert path.read_text() == result.model_dump_json() + "\n"
    assert batches.result_ids == ["batch_123"]


def test_download_results_wraps_results_stream_failure(monkeypatch, tmp_path):
    batches = RecordingBatches(failures={"results": OSError("stream closed")})
    install_client(monkeypatch, batches)

    with pytest.raises(Exception, match="Failed to download Anthropic results") as exc:
        AnthropicProvider().download_results("batch_123", str(tmp_path / "out"))

    assert isinstance(exc.value.__cause__, OSError)


@pytest.mark.parametrize("operation", ["retrieve", "download"])
def test_results_allow_older_batches_without_request_counts(
    monkeypatch, tmp_path, operation
):
    result = make_result("one", "First answer")
    batch = SimpleNamespace(id="older_batch", processing_status="ended")
    batches = RecordingBatches(batch=batch, results=[result])
    install_client(monkeypatch, batches)
    destination = tmp_path / "results.jsonl"

    if operation == "retrieve":
        assert AnthropicProvider().retrieve_results("older_batch") == (
            result.model_dump_json()
        )
    else:
        AnthropicProvider().download_results("older_batch", str(destination))
        assert destination.read_text() == f"{result.model_dump_json()}\n"

    assert batches.retrieved_ids == ["older_batch"]
    assert batches.result_ids == ["older_batch"]


@pytest.mark.parametrize("beta", [False, True])
def test_cancel_batch_returns_complete_sdk_payload(monkeypatch, beta):
    batch = make_batch(status="in_progress", processing=2)
    batches = RecordingBatches(batch)
    install_client(monkeypatch, batches, beta=beta)

    assert AnthropicProvider().cancel_batch("batch_123") == batch.model_dump()
    assert batches.cancelled_ids == ["batch_123"]


def test_cancel_batch_wraps_sdk_failure(monkeypatch):
    batches = RecordingBatches(failures={"cancel": PermissionError("read only")})
    install_client(monkeypatch, batches)

    with pytest.raises(Exception, match="Failed to cancel Anthropic batch") as exc:
        AnthropicProvider().cancel_batch("batch_123")

    assert isinstance(exc.value.__cause__, PermissionError)


@pytest.mark.parametrize("beta", [False, True])
def test_delete_batch_reports_that_deletion_is_unsupported(monkeypatch, beta):
    batches = RecordingBatches(make_batch())
    install_client(monkeypatch, batches, beta=beta)

    assert AnthropicProvider().delete_batch("batch_123") == {
        "id": "batch_123",
        "status": "ended",
        "message": "Anthropic does not support batch deletion",
    }
    assert batches.retrieved_ids == ["batch_123"]


def test_delete_batch_wraps_retrieve_failure(monkeypatch):
    batches = RecordingBatches(failures={"retrieve": LookupError("missing batch")})
    install_client(monkeypatch, batches)

    with pytest.raises(Exception, match="Failed to delete Anthropic batch") as exc:
        AnthropicProvider().delete_batch("batch_missing")

    assert isinstance(exc.value.__cause__, LookupError)


@pytest.mark.parametrize("beta", [False, True])
def test_list_batches_converts_sdk_batches_to_normalized_job_info(monkeypatch, beta):
    ended = make_batch("batch_done", status="ended", succeeded=2)
    active = make_batch("batch_active", status="in_progress", succeeded=1, processing=2)
    batches = RecordingBatches(listed=[ended, active])
    install_client(monkeypatch, batches, beta=beta)

    jobs = AnthropicProvider().list_batches(limit=2)

    assert [(job.id, job.status, job.raw_status) for job in jobs] == [
        ("batch_done", BatchStatus.COMPLETED, "ended"),
        ("batch_active", BatchStatus.PROCESSING, "in_progress"),
    ]
    assert jobs[1].request_counts.total == 3
    results_url = jobs[0].files.results_url
    assert results_url is not None
    assert results_url.endswith("/results")
    assert batches.list_limits == [2]


def test_list_batches_wraps_sdk_failure(monkeypatch):
    batches = RecordingBatches(failures={"list": TimeoutError("timed out")})
    install_client(monkeypatch, batches)

    with pytest.raises(Exception, match="Failed to list Anthropic batches") as exc:
        AnthropicProvider().list_batches()

    assert isinstance(exc.value.__cause__, TimeoutError)
