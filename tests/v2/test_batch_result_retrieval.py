"""Batch retrieval through native SDKs and loopback HTTP, without provider calls."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from instructor.batch import BatchProcessor
from instructor.batch.models import BatchError, BatchResult, BatchSuccess

pytestmark = pytest.mark.unit


class Person(BaseModel):
    name: str


class BatchEndpoint:
    def __init__(self) -> None:
        self.url = ""
        self.routes: dict[str, list[tuple[str, bytes]]] = {}
        self.requests: list[str] = []

    def json_states(self, path: str, *states: dict[str, Any]) -> None:
        self.routes[path] = [
            ("application/json", json.dumps(state).encode()) for state in states
        ]

    def jsonl(self, path: str, records: list[dict[str, Any]]) -> None:
        # Deliberately omit the final newline: separate files still need a boundary.
        body = "\n".join(json.dumps(record) for record in records).encode()
        self.routes[path] = [("application/jsonl", body)]


@pytest.fixture
def endpoint(monkeypatch: pytest.MonkeyPatch) -> Iterator[BatchEndpoint]:
    endpoint = BatchEndpoint()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            endpoint.requests.append(self.path)
            replies = endpoint.routes.get(self.path)
            if replies is None:
                status, content_type, body = 404, "application/json", b"{}"
            else:
                status = 200
                content_type, body = replies[0]
                if len(replies) > 1:
                    replies.pop(0)
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    endpoint.url = f"http://127.0.0.1:{server.server_port}"
    for provider in ("OPENAI", "ANTHROPIC"):
        monkeypatch.setenv(f"{provider}_API_KEY", "local-only-batch-test")
        monkeypatch.setenv(f"{provider}_BASE_URL", endpoint.url)
    for variable in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.delenv(variable, raising=False)
        monkeypatch.delenv(variable.lower(), raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}
    )
    thread.start()
    try:
        yield endpoint
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def openai_batch(
    *, completed: int, failed: int, output_file: str | None, error_file: str | None
) -> dict[str, Any]:
    return {
        "id": "batch-local",
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "input_file_id": "file-input",
        "completion_window": "24h",
        "created_at": 1,
        "completed_at": 2,
        "status": "completed",
        "request_counts": {
            "total": completed + failed,
            "completed": completed,
            "failed": failed,
        },
        "output_file_id": output_file,
        "error_file_id": error_file,
    }


def openai_success() -> dict[str, Any]:
    return {
        "id": "batch_req_success",
        "custom_id": "successful-request",
        "error": None,
        "response": {
            "status_code": 200,
            "request_id": "req_success",
            "body": {
                "id": "chatcmpl-local",
                "object": "chat.completion",
                "created": 1,
                "model": "local",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"name":"Ada"}'},
                        "finish_reason": "stop",
                    }
                ],
            },
        },
    }


def openai_error(custom_id: str, shape: str) -> tuple[dict[str, Any], str, str]:
    message = f"Invalid request for {custom_id}"
    record: dict[str, Any] = {"id": f"batch_req_{custom_id}", "custom_id": custom_id}
    if shape == "top-level":
        error_type = "batch_request_error"
        record.update(response=None, error={"code": error_type, "message": message})
    else:
        details = {"code": "invalid_parameter", "message": message}
        if shape == "body-type":
            details["type"] = "invalid_request_error"
        error_type = details.get("type", details["code"])
        record.update(
            error=None,
            response={
                "status_code": 400,
                "request_id": f"req_{custom_id}",
                "body": {"error": details},
            },
        )
    return record, error_type, message


def assert_retrieve_download_parity(
    processor: BatchProcessor[Person], tmp_path: Path
) -> list[BatchResult]:
    path = tmp_path / "results.jsonl"
    retrieved = processor.get_results("batch-local", file_path=str(path))
    downloaded = processor.parse_results(path.read_text(encoding="utf-8"))
    assert downloaded == retrieved
    assert [result.custom_id for result in downloaded] == [
        result.custom_id for result in retrieved
    ]
    return retrieved


@pytest.mark.parametrize("shape", ["top-level", "body-type", "body-code"])
def test_completed_openai_batch_combines_output_and_error_files(
    endpoint: BatchEndpoint, tmp_path: Path, shape: str
) -> None:
    record, error_type, message = openai_error("failed-request", shape)
    endpoint.json_states(
        "/batches/batch-local",
        openai_batch(
            completed=1, failed=1, output_file="file-output", error_file="file-errors"
        ),
    )
    endpoint.jsonl("/files/file-output/content", [openai_success()])
    endpoint.jsonl("/files/file-errors/content", [record])
    processor = BatchProcessor("openai/local", Person)

    results = assert_retrieve_download_parity(processor, tmp_path)

    assert results == [
        BatchSuccess(custom_id="successful-request", result=Person(name="Ada")),
        BatchError(
            custom_id="failed-request",
            error_type=error_type,
            error_message=message,
            raw_data=record,
        ),
    ]
    assert endpoint.requests.count("/files/file-output/content") == 2
    assert endpoint.requests.count("/files/file-errors/content") == 2


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_completed_all_failed_batch_retains_native_error_records(
    endpoint: BatchEndpoint, tmp_path: Path, provider: str
) -> None:
    expected: list[BatchError] = []
    records: list[dict[str, Any]] = []
    for custom_id, shape in (("failed-b", "top-level"), ("failed-a", "body-type")):
        if provider == "openai":
            record, error_type, message = openai_error(custom_id, shape)
            raw_data = record
        else:
            sdk_types = pytest.importorskip("anthropic.types.messages")
            error_type = "invalid_request_error"
            message = f"Invalid request for {custom_id}"
            record = {
                "custom_id": custom_id,
                "result": {
                    "type": "errored",
                    "error": {
                        "type": "error",
                        "error": {"type": error_type, "message": message},
                    },
                },
            }
            raw_data = sdk_types.MessageBatchIndividualResponse.model_validate(
                record
            ).model_dump(mode="json")
        records.append(record)
        expected.append(
            BatchError(
                custom_id=custom_id,
                error_type=error_type,
                error_message=message,
                raw_data=raw_data,
            )
        )
    if provider == "openai":
        batch_path = "/batches/batch-local"
        result_path = "/files/file-errors/content"
        state = openai_batch(
            completed=0, failed=2, output_file=None, error_file="file-errors"
        )
    else:
        batch_path = "/v1/messages/batches/batch-local"
        result_path = f"{batch_path}/results"
        state = {
            "id": "batch-local",
            "type": "message_batch",
            "processing_status": "ended",
            "created_at": "2026-01-01T00:00:00Z",
            "ended_at": "2026-01-01T00:01:00Z",
            "expires_at": "2026-01-02T00:00:00Z",
            "request_counts": {
                "processing": 0,
                "succeeded": 0,
                "errored": 2,
                "canceled": 0,
                "expired": 0,
            },
            "results_url": endpoint.url + result_path,
        }
    endpoint.json_states(batch_path, state)
    endpoint.jsonl(result_path, records)
    processor = BatchProcessor(f"{provider}/local", Person)

    assert assert_retrieve_download_parity(processor, tmp_path) == expected
    assert endpoint.requests.count(result_path) == 2


def test_openai_waits_for_expected_error_file_even_when_output_is_ready(
    endpoint: BatchEndpoint, tmp_path: Path
) -> None:
    record, error_type, message = openai_error("delayed-failure", "top-level")
    endpoint.json_states(
        "/batches/batch-local",
        openai_batch(completed=1, failed=1, output_file="file-output", error_file=None),
        openai_batch(
            completed=1, failed=1, output_file="file-output", error_file="file-errors"
        ),
    )
    endpoint.jsonl("/files/file-output/content", [openai_success()])
    endpoint.jsonl("/files/file-errors/content", [record])

    results = assert_retrieve_download_parity(
        BatchProcessor("openai/local", Person), tmp_path
    )

    assert results == [
        BatchSuccess(custom_id="successful-request", result=Person(name="Ada")),
        BatchError(
            custom_id="delayed-failure",
            error_type=error_type,
            error_message=message,
            raw_data=record,
        ),
    ]
    assert endpoint.requests.count("/batches/batch-local") >= 3
    assert endpoint.requests.count("/files/file-errors/content") == 2


@pytest.mark.parametrize("counts", ["missing", "null"])
@pytest.mark.parametrize("files", ["output", "error", "both"])
def test_openai_reads_available_files_when_request_counts_are_unknown(
    endpoint: BatchEndpoint, tmp_path: Path, counts: str, files: str
) -> None:
    has_output = files in {"output", "both"}
    has_error = files in {"error", "both"}
    record, error_type, message = openai_error("unknown-counts-failure", "top-level")
    state = openai_batch(
        completed=int(has_output),
        failed=int(has_error),
        output_file="file-output" if has_output else None,
        error_file="file-errors" if has_error else None,
    )
    if counts == "missing":
        state.pop("request_counts")
    else:
        state["request_counts"] = None
    endpoint.json_states("/batches/batch-local", state)
    expected: list[BatchResult] = []
    if has_output:
        endpoint.jsonl("/files/file-output/content", [openai_success()])
        expected.append(
            BatchSuccess(custom_id="successful-request", result=Person(name="Ada"))
        )
    if has_error:
        endpoint.jsonl("/files/file-errors/content", [record])
        expected.append(
            BatchError(
                custom_id="unknown-counts-failure",
                error_type=error_type,
                error_message=message,
                raw_data=record,
            )
        )

    results = assert_retrieve_download_parity(
        BatchProcessor("openai/local", Person), tmp_path
    )

    assert results == expected
    assert endpoint.requests.count("/batches/batch-local") == 2
    assert endpoint.requests.count("/files/file-output/content") == 2 * has_output
    assert endpoint.requests.count("/files/file-errors/content") == 2 * has_error


@pytest.mark.parametrize("missing", ["output", "both"])
def test_openai_waits_for_output_or_any_file_when_counts_are_unknown(
    endpoint: BatchEndpoint, tmp_path: Path, missing: str
) -> None:
    record, error_type, message = openai_error("delayed-file-failure", "top-level")
    first = openai_batch(
        completed=1,
        failed=1,
        output_file=None,
        error_file="file-errors" if missing == "output" else None,
    )
    ready = openai_batch(
        completed=1, failed=1, output_file="file-output", error_file="file-errors"
    )
    if missing == "both":
        first["request_counts"] = None
        ready["request_counts"] = None
    endpoint.json_states("/batches/batch-local", first, ready)
    endpoint.jsonl("/files/file-output/content", [openai_success()])
    endpoint.jsonl("/files/file-errors/content", [record])

    results = assert_retrieve_download_parity(
        BatchProcessor("openai/local", Person), tmp_path
    )

    assert results == [
        BatchSuccess(custom_id="successful-request", result=Person(name="Ada")),
        BatchError(
            custom_id="delayed-file-failure",
            error_type=error_type,
            error_message=message,
            raw_data=record,
        ),
    ]
    assert endpoint.requests.count("/batches/batch-local") == 3
    assert endpoint.requests.count("/files/file-output/content") == 2
    assert endpoint.requests.count("/files/file-errors/content") == 2


@pytest.mark.parametrize("contents", ["empty", "no-final-newline", "trailing-newlines"])
def test_openai_shared_output_and_error_file_is_downloaded_once_per_operation(
    endpoint: BatchEndpoint, tmp_path: Path, contents: str
) -> None:
    state = openai_batch(
        completed=1, failed=1, output_file="file-shared", error_file="file-shared"
    )
    endpoint.json_states("/batches/batch-local", state)
    record, error_type, message = openai_error("shared-file-failure", "top-level")
    records = [] if contents == "empty" else [openai_success(), record]
    path = "/files/file-shared/content"
    endpoint.jsonl(path, records)
    if contents == "trailing-newlines":
        content_type, body = endpoint.routes[path][0]
        endpoint.routes[path] = [(content_type, body + b"\n\n")]

    results = assert_retrieve_download_parity(
        BatchProcessor("openai/local", Person), tmp_path
    )

    expected: list[BatchResult] = []
    if records:
        expected = [
            BatchSuccess(custom_id="successful-request", result=Person(name="Ada")),
            BatchError(
                custom_id="shared-file-failure",
                error_type=error_type,
                error_message=message,
                raw_data=record,
            ),
        ]
    assert results == expected
    assert endpoint.requests.count(path) == 2


@pytest.mark.parametrize(
    "envelope",
    [
        {"error": "malformed error", "response": None},
        {"error": ["malformed error"], "response": None},
        {"error": None, "response": "malformed response"},
        {"error": None, "response": {"body": ["malformed body"]}},
        {"error": None, "response": {"body": {"error": "malformed error"}}},
    ],
    ids=[
        "string-error",
        "list-error",
        "string-response",
        "list-body",
        "string-body-error",
    ],
)
def test_openai_malformed_error_envelope_preserves_id_and_raw_record(
    endpoint: BatchEndpoint, tmp_path: Path, envelope: dict[str, Any]
) -> None:
    record = {"custom_id": "malformed-failure", **envelope}
    endpoint.json_states(
        "/batches/batch-local",
        openai_batch(completed=0, failed=1, output_file=None, error_file="file-errors"),
    )
    endpoint.jsonl("/files/file-errors/content", [record])

    results = assert_retrieve_download_parity(
        BatchProcessor("openai/local", Person), tmp_path
    )

    assert results == [
        BatchError(
            custom_id="malformed-failure",
            error_type="extraction_error",
            error_message="Unknown error",
            raw_data=record,
        )
    ]


@pytest.mark.parametrize(
    "envelope",
    [
        {"error": None, "response": {"body": {"error": {}}}},
        {"error": {"param": "model"}, "response": None},
    ],
    ids=["empty-body-error", "top-error-without-details"],
)
def test_openai_error_dictionary_without_details_uses_provider_fallbacks(
    endpoint: BatchEndpoint, tmp_path: Path, envelope: dict[str, Any]
) -> None:
    record = {"custom_id": "missing-details", **envelope}
    endpoint.json_states(
        "/batches/batch-local",
        openai_batch(completed=0, failed=1, output_file=None, error_file="file-errors"),
    )
    endpoint.jsonl("/files/file-errors/content", [record])

    results = assert_retrieve_download_parity(
        BatchProcessor("openai/local", Person), tmp_path
    )

    assert results == [
        BatchError(
            custom_id="missing-details",
            error_type="openai_error",
            error_message="Unknown OpenAI error",
            raw_data=record,
        )
    ]
