"""Batch result parsing and job normalization, table-driven across providers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from anthropic.types.messages import MessageBatch, MessageBatchIndividualResponse
from openai.types import Batch
from pydantic import BaseModel, RootModel

import instructor.batch.processor as processor_module
from instructor.batch import (
    BatchError,
    BatchJob,
    BatchJobInfo,
    BatchProcessor,
    BatchStatus,
    BatchSuccess,
    extract_results,
    filter_errors,
    filter_successful,
    get_results_by_custom_id,
)

pytestmark = pytest.mark.unit

PROVIDERS = ["openai", "anthropic"]


class Person(BaseModel):
    name: str
    age: int


class Document(BaseModel):
    root: str
    label: str = "document"


class RootDocument(RootModel[Document]):
    pass


class DefaultDocument(Document):
    root: str = "untitled"


class RootDefaultDocument(RootModel[DefaultDocument]):
    pass


def record(
    provider: str,
    payload: Any,
    *,
    custom_id: str = "request-1",
    block: str = "tool_use",
    stop: str | None = None,
) -> dict[str, Any]:
    """Build one provider result record carrying ``payload`` as structured output."""
    if provider == "anthropic":
        content = (
            {"type": "tool_use", "input": payload}
            if block == "tool_use"
            else {"type": "text", "text": json.dumps(payload)}
        )
        message: dict[str, Any] = {"content": [content]}
        if stop is not None:
            message["stop_reason"] = stop
        return {
            "custom_id": custom_id,
            "result": {"type": "succeeded", "message": message},
        }
    choice: dict[str, Any] = {"message": {"content": json.dumps(payload)}}
    if stop is not None:
        choice["finish_reason"] = stop
    return {"custom_id": custom_id, "response": {"body": {"choices": [choice]}}}


def parse(provider: str, response_model: Any, *records: dict[str, Any] | str) -> Any:
    lines = [r if isinstance(r, str) else json.dumps(r) for r in records]
    return BatchProcessor(f"{provider}/model", response_model).parse_results(
        "\n".join(lines)
    )


# --- successful records -------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "block"),
    [("openai", "text"), ("anthropic", "tool_use"), ("anthropic", "text")],
)
@pytest.mark.parametrize("payload", [{"root": "Ada"}, {"root": "Ada", "label": "x"}])
@pytest.mark.parametrize("response_model", [Document, RootDocument])
def test_processor_and_batch_job_validate_objects_and_root_models(
    provider: str, block: str, payload: dict[str, Any], response_model: Any
) -> None:
    line = json.dumps(record(provider, payload, block=block))
    expected = response_model.model_validate(payload)

    assert parse(provider, response_model, line) == [
        BatchSuccess(custom_id="request-1", result=expected)
    ]
    assert BatchJob.parse_from_string(line, response_model) == ([expected], [])


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize(
    "response_model", [DefaultDocument, RootDefaultDocument], ids=["model", "root"]
)
def test_empty_object_validates_against_defaults(
    provider: str, response_model: Any, tmp_path: Path
) -> None:
    line = json.dumps(record(provider, {}))
    path = tmp_path / "results.jsonl"
    path.write_text(line, encoding="utf-8")
    expected = response_model.model_validate({})

    assert parse(provider, response_model, line) == [
        BatchSuccess(custom_id="request-1", result=expected)
    ]
    assert BatchJob.parse_from_string(line, response_model) == ([expected], [])
    assert BatchJob.parse_from_file(str(path), response_model) == ([expected], [])


def test_anthropic_prefers_tool_use_then_first_parseable_text() -> None:
    tool = record("anthropic", {"name": "Ada", "age": 36}, custom_id="tool")
    tool["result"]["message"]["content"].insert(0, {"type": "text", "text": "x"})
    text = record(
        "anthropic", {"name": "Lin", "age": 28}, custom_id="text", block="text"
    )
    text["result"]["message"]["content"].insert(0, {"type": "image", "source": {}})
    text["result"]["message"]["content"].insert(0, {"type": "text", "text": "x"})

    assert parse("anthropic", Person, tool, text) == [
        BatchSuccess(custom_id="tool", result=Person(name="Ada", age=36)),
        BatchSuccess(custom_id="text", result=Person(name="Lin", age=28)),
    ]


def test_mistral_records_parse_like_openai() -> None:
    def mistral(custom_id: str, content: str) -> dict[str, Any]:
        return {
            "id": f"result_{custom_id}",
            "custom_id": custom_id,
            "response": {
                "status_code": 200,
                "body": {
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                },
            },
            "error": None,
        }

    success, invalid = parse(
        "mistral",
        Person,
        mistral("request-0", '{"name": "Ada", "age": 36}'),
        mistral("request-1", "not json"),
    )

    assert success == BatchSuccess(
        custom_id="request-0", result=Person(name="Ada", age=36)
    )
    assert isinstance(invalid, BatchError) and invalid.custom_id == "request-1"


# --- failed records -----------------------------------------------------------


def anthropic_sdk_error(error_type: str, message: str) -> dict[str, Any]:
    response = MessageBatchIndividualResponse.model_validate(
        {
            "custom_id": "failed",
            "result": {
                "type": "errored",
                "error": {
                    "type": "error",
                    "error": {"type": error_type, "message": message},
                },
            },
        }
    )
    return json.loads(response.model_dump_json())


PROVIDER_ERRORS = {
    "anthropic-sdk-invalid": (
        "anthropic",
        anthropic_sdk_error("invalid_request_error", "max_tokens must be > 0"),
        "invalid_request_error",
        "max_tokens must be > 0",
    ),
    "anthropic-sdk-rate-limit": (
        "anthropic",
        anthropic_sdk_error("rate_limit_error", "Too many requests"),
        "rate_limit_error",
        "Too many requests",
    ),
    "anthropic-legacy": (
        "anthropic",
        {
            "result": {
                "type": "error",
                "error": {"error": {"type": "rate_limit_error", "message": "slow"}},
            }
        },
        "rate_limit_error",
        "slow",
    ),
    "anthropic-string": (
        "anthropic",
        {"result": {"type": "error", "error": "service unavailable"}},
        "anthropic_error",
        "service unavailable",
    ),
    "anthropic-empty": (
        "anthropic",
        {"result": {"type": "succeeded", "message": {"content": []}}},
        "extraction_error",
        "Unknown error",
    ),
    "openai-string": (
        "openai",
        {"error": "service unavailable", "response": None},
        "openai_error",
        "service unavailable",
    ),
    "openai-body-string": (
        "openai",
        {"response": {"status_code": 400, "body": {"error": "bad request"}}},
        "openai_error",
        "bad request",
    ),
    "openai-http-status": (
        "openai",
        {"response": {"status_code": 503, "body": None}},
        "openai_error",
        "HTTP 503 error",
    ),
    "openai-empty-body": (
        "openai",
        {"response": {"status_code": 200, "body": {}}},
        "extraction_error",
        "Unknown error",
    ),
    "openai-missing-response": ("openai", {}, "extraction_error", "Unknown error"),
    "openai-code": (
        "openai",
        {"error": {"code": "expired", "message": "Expired"}},
        "expired",
        "Expired",
    ),
    "openai-body-type": (
        "openai",
        {
            "response": {
                "status_code": 400,
                "body": {
                    "error": {
                        "type": "invalid_request_error",
                        "code": "invalid_parameter",
                        "message": "Bad parameter",
                    }
                },
            }
        },
        "invalid_request_error",
        "Bad parameter",
    ),
    "mistral-dict": (
        "mistral",
        {
            "response": None,
            "error": {"message": "model not found", "type": "invalid_request_error"},
        },
        "invalid_request_error",
        "model not found",
    ),
    "mistral-string": (
        "mistral",
        {"error": "rate limited"},
        "mistral_error",
        "rate limited",
    ),
}


@pytest.mark.parametrize(
    ("provider", "failed", "error_type", "message"),
    PROVIDER_ERRORS.values(),
    ids=PROVIDER_ERRORS.keys(),
)
def test_failed_record_keeps_provider_details_next_to_successes(
    provider: str, failed: dict[str, Any], error_type: str, message: str
) -> None:
    failed = {"custom_id": "failed", **failed}
    success = record(provider, {"name": "Ada", "age": 36}, custom_id="ok", block="text")

    assert parse(provider, Person, success, failed) == [
        BatchSuccess(custom_id="ok", result=Person(name="Ada", age=36)),
        BatchError(
            custom_id="failed",
            error_type=error_type,
            error_message=message,
            raw_data=failed,
        ),
    ]


def test_openai_distinguishes_validation_and_json_errors() -> None:
    results = parse(
        "openai",
        Person,
        record("openai", {"name": "Ada", "age": "unknown"}, custom_id="invalid"),
        "   ",
        "not-json",
    )

    assert len(results) == 2
    invalid, broken = results
    assert isinstance(invalid, BatchError) and isinstance(broken, BatchError)
    assert (invalid.custom_id, invalid.error_type) == ("invalid", "parsing_error")
    assert "Failed to parse into Person" in invalid.error_message
    assert invalid.raw_data == {"name": "Ada", "age": "unknown"}
    assert (broken.custom_id, broken.error_type) == ("unknown", "json_parse_error")
    assert broken.raw_data == {"raw_line": "not-json"}


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("payload", [[1], "invalid", 17, {"root": 17}])
def test_invalid_payload_keeps_request_id(provider: str, payload: Any) -> None:
    failed = record(provider, payload, custom_id="bad", block="text")

    results = parse(provider, RootDocument, failed, record(provider, {"root": "Ada"}))

    assert isinstance(results[0], BatchError)
    assert (results[0].custom_id, results[0].error_type) == ("bad", "parsing_error")
    # Object payloads are kept as extracted; anything else keeps the full record.
    assert results[0].raw_data == (payload if isinstance(payload, dict) else failed)
    assert isinstance(results[1], BatchSuccess)


@pytest.mark.parametrize("input_present", [False, True])
def test_anthropic_missing_tool_input_is_not_an_empty_result(
    input_present: bool,
) -> None:
    failed = record("anthropic", {})
    tool = failed["result"]["message"]["content"][0]
    if input_present:
        tool["input"] = None
    else:
        del tool["input"]
    line = json.dumps(failed)

    (result,) = parse("anthropic", RootDefaultDocument, line)

    assert isinstance(result, BatchError) and result.raw_data == failed
    assert BatchJob.parse_from_string(line, RootDefaultDocument) == ([], [failed])


@pytest.mark.parametrize(
    ("provider", "block", "stop", "truncated"),
    [
        ("openai", "text", "stop", False),
        ("openai", "text", "length", True),
        ("anthropic", "tool_use", "tool_use", False),
        ("anthropic", "tool_use", "max_tokens", True),
        ("anthropic", "text", "end_turn", False),
        ("anthropic", "text", "max_tokens", True),
    ],
)
def test_token_limit_is_checked_before_defaults_apply(
    provider: str, block: str, stop: str, truncated: bool
) -> None:
    limited = record(provider, {}, block=block, stop=stop)

    results = parse(provider, DefaultDocument, limited)
    parsed = BatchJob.parse_from_string(json.dumps(limited), DefaultDocument)

    if truncated:
        (error,) = results
        assert isinstance(error, BatchError)
        assert error.error_type == "incomplete_output"
        assert "max_tokens" in error.error_message
        assert error.raw_data == limited
        assert parsed == ([], [limited])
    else:
        assert results == [
            BatchSuccess(custom_id="request-1", result=DefaultDocument())
        ]
        assert parsed == ([DefaultDocument()], [])


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"custom_id": "no-result"}, None),
        ({"result": {"type": "succeeded", "message": {"content": "x"}}}, None),
        ({"result": {"type": "succeeded"}}, None),
        ({"result": {"message": {"content": []}}}, None),
        ({"result": {"message": {"content": [{"type": "image"}]}}}, None),
        ({"response": {"body": {"choices": []}}}, None),
        ({"response": {"body": {"choices": [{"message": {}}]}}}, None),
        ({"anything": "goes"}, None),
        (
            {
                "response": {
                    "body": {
                        "choices": [
                            {
                                "message": {
                                    "content": {"ignored": True},
                                    "tool_calls": [
                                        {"function": {"arguments": '{"age": 36}'}}
                                    ],
                                }
                            }
                        ]
                    }
                }
            },
            {"age": 36},
        ),
    ],
)
@pytest.mark.parametrize(
    "extract",
    [processor_module._extract_structured_data, BatchJob._extract_structured_data],
    ids=["processor", "batch-job"],
)
def test_extract_structured_data_handles_unknown_shapes(
    extract: Any, data: dict[str, Any], expected: Any
) -> None:
    assert extract(data) == expected


def test_batch_job_file_parser_reads_every_provider_and_keeps_errors(
    tmp_path: Path,
) -> None:
    name = "Ren" + chr(0xE9) + "e"
    tool_call = {
        "custom_id": "openai-tool",
        "response": {
            "body": {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"function": {"arguments": '{"name":"Grace","age":42}'}}
                            ]
                        }
                    }
                ]
            }
        },
    }
    invalid = record("openai", {"name": "No age"})
    malformed = {
        "custom_id": "malformed",
        "response": {"body": {"choices": [{"message": {"content": "not json"}}]}},
    }
    unsupported = {"custom_id": "unsupported", "response": {"body": {}}}
    lines = [
        json.dumps(record("openai", {"name": name, "age": 30}), ensure_ascii=False),
        "   ",
        json.dumps(tool_call),
        json.dumps(record("anthropic", {"name": "Kat", "age": 51})),
        json.dumps(record("anthropic", {"name": "Meg", "age": 28}, block="text")),
        json.dumps(invalid),
        json.dumps(malformed),
        json.dumps(unsupported),
        "{not-valid-json",
    ]
    path = tmp_path / "results.jsonl"
    path.write_bytes("\n".join(lines).encode("utf-8"))

    results, errors = BatchJob.parse_from_file(str(path), Person)

    assert results == [
        Person(name=name, age=30),
        Person(name="Grace", age=42),
        Person(name="Kat", age=51),
        Person(name="Meg", age=28),
    ]
    assert errors == [
        invalid,
        malformed,
        unsupported,
        {"error": "Failed to parse JSON", "raw_line": "{not-valid-json"},
    ]


def test_result_helpers_keep_successes_errors_and_custom_ids() -> None:
    ada = BatchSuccess(custom_id="r1", result=Person(name="Ada", age=36))
    failure = BatchError(custom_id="r2", error_type="rate_limit", error_message="x")
    grace = BatchSuccess(custom_id="r3", result=Person(name="Grace", age=42))
    results = [ada, failure, grace]

    assert filter_successful(results) == [ada, grace]
    assert filter_errors(results) == [failure]
    assert extract_results(results) == [ada.result, grace.result]
    assert get_results_by_custom_id(results) == {"r1": ada, "r2": failure, "r3": grace}


# --- job normalization --------------------------------------------------------


def utc(timestamp: int) -> datetime:
    return datetime.fromtimestamp(timestamp, timezone.utc)


def test_openai_job_info_normalizes_every_field() -> None:
    payload = {
        "id": "batch-openai",
        "status": "failed",
        "created_at": 1_700_000_000,
        "in_progress_at": 1_700_000_010,
        "completed_at": 1_700_000_020,
        "failed_at": 1_700_000_021,
        "cancelled_at": 1_700_000_022,
        "expired_at": 1_700_000_023,
        "expires_at": 1_700_000_024,
        "request_counts": {"total": 12, "completed": 8, "failed": 4},
        "input_file_id": "file-input",
        "output_file_id": "file-output",
        "error_file_id": "file-error",
        "errors": {"type": "invalid_request_error", "message": "bad", "code": "x"},
        "metadata": {"tenant": "example"},
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
    }

    result = BatchJobInfo.from_openai(payload)

    assert (result.id, result.provider, result.status, result.raw_status) == (
        "batch-openai",
        "openai",
        BatchStatus.FAILED,
        "failed",
    )
    assert result.timestamps.model_dump() == {
        "created_at": utc(1_700_000_000),
        "started_at": utc(1_700_000_010),
        "completed_at": utc(1_700_000_020),
        "failed_at": utc(1_700_000_021),
        "cancelled_at": utc(1_700_000_022),
        "expired_at": utc(1_700_000_023),
        "expires_at": utc(1_700_000_024),
    }
    assert result.request_counts.model_dump(exclude_none=True) == {
        "total": 12,
        "completed": 8,
        "failed": 4,
    }
    assert result.files.model_dump() == {
        "input_file_id": "file-input",
        "output_file_id": "file-output",
        "error_file_id": "file-error",
        "results_url": None,
    }
    assert result.error is not None
    assert result.error.model_dump() == {
        "error_type": "invalid_request_error",
        "error_message": "bad",
        "error_code": "x",
    }
    assert result.metadata == {"tenant": "example"}
    assert result.raw_data == payload
    assert (result.endpoint, result.completion_window) == (
        "/v1/chat/completions",
        "24h",
    )
    minimal = BatchJobInfo.from_openai({"id": "batch-new", "status": "queued"})
    assert minimal.status is BatchStatus.PENDING
    assert minimal.timestamps == minimal.timestamps.__class__()
    assert minimal.error is None


def openai_sdk_payload(**fields: Any) -> dict[str, Any]:
    return Batch.model_validate(
        {
            "id": "batch-test",
            "object": "batch",
            "endpoint": "/v1/chat/completions",
            "input_file_id": "file-input",
            "completion_window": "24h",
            "created_at": 1_700_000_000,
            "status": "validating",
            **fields,
        }
    ).model_dump()


@pytest.mark.parametrize("counts", [None, {"total": 2, "completed": 1, "failed": 0}])
@pytest.mark.parametrize("metadata", [None, {"source": "example"}])
def test_openai_job_info_accepts_nullable_sdk_mappings(
    counts: dict[str, int] | None, metadata: dict[str, str] | None
) -> None:
    payload = openai_sdk_payload(request_counts=counts, metadata=metadata)

    result = BatchJobInfo.from_openai(payload)

    assert result.status is BatchStatus.PENDING
    for key in ("total", "completed", "failed"):
        assert getattr(result.request_counts, key) == (counts or {}).get(key)
    assert result.metadata == (metadata or {})
    assert result.raw_data == payload


def validation_errors(count: int) -> list[dict[str, Any]]:
    return [
        {"code": f"invalid_{i}", "message": f"Bad line {i}", "line": i, "param": "x"}
        for i in range(1, count + 1)
    ]


@pytest.mark.parametrize(
    ("errors", "expected"),
    [
        pytest.param("absent", None, id="absent"),
        pytest.param(None, None, id="null"),
        pytest.param(
            {"object": "list", "data": validation_errors(1)},
            {
                "error_type": None,
                "error_message": "Bad line 1",
                "error_code": "invalid_1",
            },
            id="one-validation-error",
        ),
        pytest.param(
            {"object": "list", "data": validation_errors(2)},
            {
                "error_type": None,
                "error_message": "Bad line 1",
                "error_code": "invalid_1",
            },
            id="first-validation-error",
        ),
        pytest.param(
            {"object": "list", "data": []},
            {"error_type": None, "error_message": None, "error_code": None},
            id="empty-list",
        ),
        pytest.param(
            {"type": "validation", "code": "invalid", "message": "Bad file"},
            {
                "error_type": "validation",
                "error_message": "Bad file",
                "error_code": "invalid",
            },
            id="flat-mapping",
        ),
    ],
)
def test_openai_job_errors_keep_sdk_details(errors: Any, expected: Any) -> None:
    payload = openai_sdk_payload(status="failed")
    if errors == "absent":
        payload.pop("errors")
    else:
        payload["errors"] = errors

    job = BatchJobInfo.from_openai(payload)

    assert (job.error.model_dump() if job.error else None) == expected
    assert job.raw_data == payload


def test_anthropic_job_info_accepts_timestamp_variants_and_counts() -> None:
    created_at = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    payload = {
        "id": "batch-anthropic",
        "processing_status": "ended",
        "created_at": created_at,
        "cancel_initiated_at": "not-a-timestamp",
        "ended_at": 17,
        "expires_at": "2025-01-02T12:00:00Z",
        "request_counts": {
            "processing": 1,
            "succeeded": 7,
            "errored": 2,
            "canceled": 3,
            "expired": 4,
        },
        "results_url": "https://example.test/results/batch-anthropic",
    }

    result = BatchJobInfo.from_anthropic(payload)

    assert (result.provider, result.status, result.raw_status) == (
        "anthropic",
        BatchStatus.COMPLETED,
        "ended",
    )
    assert result.timestamps.created_at == result.timestamps.started_at == created_at
    assert result.timestamps.cancelled_at is None
    assert result.timestamps.completed_at == utc(17)
    assert result.timestamps.expires_at == datetime(2025, 1, 2, 12, tzinfo=timezone.utc)
    assert result.request_counts.model_dump(exclude_none=True) == {
        "total": 17,
        "processing": 1,
        "succeeded": 7,
        "errored": 2,
        "cancelled": 3,
        "expired": 4,
    }
    assert result.files.results_url == "https://example.test/results/batch-anthropic"
    assert result.raw_data == payload
    minimal = BatchJobInfo.from_anthropic({"id": "new", "processing_status": "queued"})
    assert minimal.status is BatchStatus.PENDING
    assert minimal.timestamps.created_at is None
    assert minimal.request_counts.total == 0


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("in_progress", BatchStatus.PROCESSING),
        ("canceling", BatchStatus.CANCELLED),
        ("ended", BatchStatus.COMPLETED),
    ],
)
def test_anthropic_job_info_normalizes_sdk_statuses(
    status: str, expected: BatchStatus
) -> None:
    ended = status == "ended"
    batch = MessageBatch.model_validate(
        {
            "id": "msgbatch-local",
            "type": "message_batch",
            "processing_status": status,
            "created_at": "2026-10-24T00:00:00Z",
            "expires_at": "2026-10-25T00:00:00Z",
            "ended_at": "2026-10-24T00:00:20Z" if ended else None,
            "cancel_initiated_at": None
            if status == "in_progress"
            else "2026-10-24T00:00:10Z",
            "results_url": None,
            "request_counts": {
                "processing": 0 if ended else 2,
                "succeeded": 1,
                "errored": 0,
                "canceled": 2 if ended else 0,
                "expired": 0,
            },
        }
    )

    result = BatchJobInfo.from_anthropic(batch.model_dump())

    assert (result.status, result.raw_status) == (expected, status)
    assert result.timestamps.cancelled_at == batch.cancel_initiated_at
    assert result.request_counts.total == 3
    assert result.raw_data == batch.model_dump()


MISTRAL_CREATED = 1735784800  # 2025-01-02T02:26:40Z


@pytest.mark.parametrize(
    ("raw_status", "expected"),
    [
        ("QUEUED", BatchStatus.PENDING),
        ("RUNNING", BatchStatus.PROCESSING),
        ("SUCCESS", BatchStatus.COMPLETED),
        ("FAILED", BatchStatus.FAILED),
        ("TIMEOUT_EXCEEDED", BatchStatus.EXPIRED),
        ("CANCELLATION_REQUESTED", BatchStatus.CANCELLED),
        ("CANCELLED", BatchStatus.CANCELLED),
        ("SOMETHING_NEW", BatchStatus.PENDING),
    ],
)
def test_mistral_job_info_maps_every_status(
    raw_status: str, expected: BatchStatus
) -> None:
    info = BatchJobInfo.from_mistral({"id": "j", "status": raw_status})

    assert (info.status, info.raw_status) == (expected, raw_status)


def test_mistral_job_info_normalizes_fields_and_errors() -> None:
    info = BatchJobInfo.from_mistral(
        {
            "id": "batch_123",
            "status": "SUCCESS",
            "created_at": MISTRAL_CREATED,
            "completed_at": MISTRAL_CREATED + 65,
            "input_files": ["file_in"],
            "output_file": "file_out",
            "total_requests": 2,
            "succeeded_requests": 2,
            "failed_requests": 0,
            "model": "mistral-small-latest",
            "endpoint": "/v1/chat/completions",
        }
    )

    assert (info.provider, info.model, info.endpoint) == (
        "mistral",
        "mistral-small-latest",
        "/v1/chat/completions",
    )
    assert (info.files.input_file_id, info.files.output_file_id) == (
        "file_in",
        "file_out",
    )
    counts = info.request_counts
    assert (counts.total, counts.succeeded, counts.failed) == (2, 2, 0)
    assert info.timestamps.completed_at == utc(MISTRAL_CREATED + 65)
    assert info.error is None

    failed = BatchJobInfo.from_mistral(
        {
            "id": "batch_123",
            "status": "FAILED",
            "errors": [{"message": "model not found", "count": 2}],
            "input_files": [],
            "metadata": None,
        }
    )
    assert failed.error is not None
    assert (failed.error.error_type, failed.error.error_message) == (
        "mistral_error",
        "model not found",
    )
    assert failed.files.input_file_id is None
    assert failed.metadata == {}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2025-01-02T02:26:40Z", utc(MISTRAL_CREATED)),
        (MISTRAL_CREATED, utc(MISTRAL_CREATED)),
        (
            datetime(2025, 1, 2, tzinfo=timezone.utc),
            datetime(2025, 1, 2, tzinfo=timezone.utc),
        ),
        ("not-a-date", None),
        (10**100, None),
        (None, None),
    ],
)
def test_mistral_job_info_parses_timestamp_variants(
    value: Any, expected: datetime | None
) -> None:
    info = BatchJobInfo.from_mistral(
        {"id": "j", "status": "QUEUED", "created_at": value}
    )

    assert info.timestamps.created_at == expected
