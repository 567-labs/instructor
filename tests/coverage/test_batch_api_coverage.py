"""Coverage for the public batch parsing, request, and result APIs."""

from __future__ import annotations

import io
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from instructor.batch import (
    BatchError,
    BatchJob,
    BatchJobInfo,
    BatchRequest,
    BatchStatus,
    BatchSuccess,
    extract_results,
    filter_errors,
    filter_successful,
    get_results_by_custom_id,
)

pytestmark = pytest.mark.unit


class Person(BaseModel):
    name: str
    age: int


class PersonGroup(BaseModel):
    people: list[Person]


class TypelessResponse(BaseModel):
    value: str

    @classmethod
    def model_json_schema(cls, *_args, **_kwargs) -> dict:
        return {"properties": {"value": {"type": "string"}}, "required": ["value"]}


class RestrictedResponse(BaseModel):
    value: str

    @classmethod
    def model_json_schema(cls, *_args, **_kwargs) -> dict:
        return {
            "type": "object",
            "additionalProperties": True,
            "properties": {"value": {"type": "string"}, "forbidden": False},
            "required": ["value"],
        }


def test_legacy_batch_job_parses_provider_results_and_preserves_errors(
    tmp_path: Path,
) -> None:
    openai_json = {
        "custom_id": "openai-json",
        "response": {
            "body": {
                "choices": [
                    {"message": {"content": json.dumps({"name": "Ada", "age": 36})}}
                ]
            }
        },
    }
    openai_tool = {
        "custom_id": "openai-tool",
        "response": {
            "body": {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "arguments": json.dumps(
                                            {"name": "Grace", "age": 42}
                                        )
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        },
    }
    anthropic_tool = {
        "custom_id": "anthropic-tool",
        "result": {
            "message": {
                "content": [
                    {"type": "text", "text": "Using the extraction tool."},
                    {"type": "tool_use", "input": {"name": "Katherine", "age": 51}},
                ]
            }
        },
    }
    anthropic_text = {
        "custom_id": "anthropic-text",
        "result": {
            "message": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({"name": "Margaret", "age": 28}),
                    }
                ]
            }
        },
    }
    invalid_model = {
        "custom_id": "invalid-model",
        "response": {
            "body": {
                "choices": [{"message": {"content": json.dumps({"name": "No age"})}}]
            }
        },
    }
    malformed_content = {
        "custom_id": "malformed-content",
        "response": {"body": {"choices": [{"message": {"content": "not json"}}]}},
    }
    unsupported_shape = {"custom_id": "unsupported", "response": {"body": {}}}
    content = "\n".join(
        [
            json.dumps(openai_json),
            "   ",
            json.dumps(openai_tool),
            json.dumps(anthropic_tool),
            json.dumps(anthropic_text),
            json.dumps(invalid_model),
            json.dumps(malformed_content),
            json.dumps(unsupported_shape),
            "{not-valid-json",
        ]
    )
    batch_file = tmp_path / "batch-results.jsonl"
    batch_file.write_text(content)

    results, errors = BatchJob.parse_from_file(str(batch_file), Person)

    assert results == [
        Person(name="Ada", age=36),
        Person(name="Grace", age=42),
        Person(name="Katherine", age=51),
        Person(name="Margaret", age=28),
    ]
    assert errors[:3] == [invalid_model, malformed_content, unsupported_shape]
    assert errors[3] == {"error": "Failed to parse JSON", "raw_line": "{not-valid-json"}


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {
                "response": {
                    "body": {
                        "choices": [
                            {
                                "message": {
                                    "content": {"ignored": True},
                                    "tool_calls": [
                                        {
                                            "function": {
                                                "arguments": '{"name":"Ada","age":36}'
                                            }
                                        }
                                    ],
                                }
                            }
                        ]
                    }
                }
            },
            {"name": "Ada", "age": 36},
        ),
        (
            {"response": {"body": {"choices": [{"message": {"role": "assistant"}}]}}},
            None,
        ),
        ({"result": {"message": {"content": []}}}, None),
        (
            {"result": {"message": {"content": [{"type": "image", "source": {}}]}}},
            None,
        ),
        (
            {
                "result": {
                    "message": {
                        "content": [
                            {"type": "image", "source": {}},
                            {"type": "text", "text": '{"name":"Lin","age":28}'},
                        ]
                    }
                }
            },
            {"name": "Lin", "age": 28},
        ),
    ],
)
def test_legacy_batch_job_handles_empty_unknown_and_mixed_content_blocks(
    payload: dict, expected: dict | None
) -> None:
    assert BatchJob._extract_structured_data(payload) == expected


def test_openai_batch_job_info_normalizes_status_timestamps_counts_and_error() -> None:
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
        "errors": {
            "type": "invalid_request_error",
            "message": "bad input",
            "code": "bad",
        },
        "metadata": {"tenant": "example"},
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
    }

    result = BatchJobInfo.from_openai(payload)

    assert result.id == "batch-openai"
    assert result.provider == "openai"
    assert result.status is BatchStatus.FAILED
    assert result.raw_status == "failed"
    assert result.timestamps.created_at == datetime.fromtimestamp(
        1_700_000_000, timezone.utc
    )
    assert result.timestamps.started_at == datetime.fromtimestamp(
        1_700_000_010, timezone.utc
    )
    assert result.timestamps.completed_at == datetime.fromtimestamp(
        1_700_000_020, timezone.utc
    )
    assert result.timestamps.failed_at == datetime.fromtimestamp(
        1_700_000_021, timezone.utc
    )
    assert result.timestamps.cancelled_at == datetime.fromtimestamp(
        1_700_000_022, timezone.utc
    )
    assert result.timestamps.expired_at == datetime.fromtimestamp(
        1_700_000_023, timezone.utc
    )
    assert result.timestamps.expires_at == datetime.fromtimestamp(
        1_700_000_024, timezone.utc
    )
    assert result.request_counts.model_dump() == {
        "total": 12,
        "completed": 8,
        "failed": 4,
        "processing": None,
        "succeeded": None,
        "errored": None,
        "cancelled": None,
        "expired": None,
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
        "error_message": "bad input",
        "error_code": "bad",
    }
    assert result.metadata == {"tenant": "example"}
    assert result.raw_data == payload
    assert result.endpoint == "/v1/chat/completions"
    assert result.completion_window == "24h"

    minimal = BatchJobInfo.from_openai({"id": "batch-new", "status": "queued"})
    assert minimal.status is BatchStatus.PENDING
    assert minimal.timestamps == minimal.timestamps.__class__()
    assert minimal.error is None


def test_anthropic_batch_job_info_accepts_timestamp_variants_and_counts() -> None:
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

    assert result.id == "batch-anthropic"
    assert result.provider == "anthropic"
    assert result.status is BatchStatus.COMPLETED
    assert result.raw_status == "ended"
    assert result.timestamps.created_at == created_at
    assert result.timestamps.started_at == created_at
    assert result.timestamps.cancelled_at is None
    assert result.timestamps.completed_at is None
    assert result.timestamps.expires_at == datetime(
        2025, 1, 2, 12, 0, tzinfo=timezone.utc
    )
    assert result.request_counts.model_dump() == {
        "total": 17,
        "completed": None,
        "failed": None,
        "processing": 1,
        "succeeded": 7,
        "errored": 2,
        "cancelled": 3,
        "expired": 4,
    }
    assert result.files.results_url == "https://example.test/results/batch-anthropic"
    assert result.raw_data == payload

    minimal = BatchJobInfo.from_anthropic(
        {"id": "batch-new", "processing_status": "queued"}
    )
    assert minimal.status is BatchStatus.PENDING
    assert minimal.timestamps.created_at is None
    assert minimal.request_counts.total == 0


def test_openai_batch_request_makes_nested_array_and_definition_schemas_strict() -> (
    None
):
    request = BatchRequest[PersonGroup](
        custom_id="group-1",
        messages=[{"role": "user", "content": "Extract the group."}],
        response_model=PersonGroup,
        model="gpt-4o-mini",
    )

    result = request.to_openai_format()
    schema = result["body"]["response_format"]["json_schema"]["schema"]

    assert result["custom_id"] == "group-1"
    assert result["method"] == "POST"
    assert schema["additionalProperties"] is False
    assert schema["properties"]["people"]["type"] == "array"
    assert schema["$defs"]["Person"]["additionalProperties"] is False


def test_openai_batch_request_preserves_boolean_property_schema() -> None:
    request = BatchRequest[RestrictedResponse](
        custom_id="restricted-1",
        messages=[{"role": "user", "content": "Extract the value."}],
        response_model=RestrictedResponse,
        model="gpt-4o-mini",
    )

    schema = request.to_openai_format()["body"]["response_format"]["json_schema"][
        "schema"
    ]

    assert schema["additionalProperties"] is False
    assert schema["properties"]["forbidden"] is False


@pytest.mark.parametrize("nested", [False, True], ids=["root", "nested-array"])
def test_openai_batch_request_requires_fields_with_defaults(nested: bool) -> None:
    class Contact(BaseModel):
        name: str
        phone: str | None = Field(default=None, alias="phone_number")

    class Contacts(BaseModel):
        contacts: list[Contact] = Field(default_factory=list)

    request = BatchRequest(
        custom_id="contacts-1",
        messages=[{"role": "user", "content": "Extract contact details."}],
        response_model=Contacts if nested else Contact,
        model="gpt-4o-mini",
    )
    buffer = io.BytesIO()
    request.save_to_file(buffer, provider="openai")
    result = json.loads(buffer.getvalue())
    response_format = result["body"]["response_format"]["json_schema"]
    schema = response_format["schema"]

    assert response_format["strict"] is True
    if nested:
        assert schema["required"] == ["contacts"]
        schema = schema["$defs"]["Contact"]

    assert schema["required"] == ["name", "phone_number"]
    assert {"type": "null"} in schema["properties"]["phone_number"]["anyOf"]
    # Requiring fields in the request must not change local Pydantic defaults.
    assert Contact(name="Ada").phone is None
    assert Contacts().contacts == []


def test_anthropic_batch_request_extracts_system_message_and_completes_schema() -> None:
    request = BatchRequest[TypelessResponse](
        custom_id="anthropic-1",
        messages=[
            {"role": "system", "content": "Return one value."},
            {"role": "user", "content": "Extract the value."},
        ],
        response_model=TypelessResponse,
        model="claude-sonnet",
    )

    result = request.to_anthropic_format()

    assert result["custom_id"] == "anthropic-1"
    assert result["params"]["system"] == "Return one value."
    assert result["params"]["messages"] == [
        {"role": "user", "content": "Extract the value."}
    ]
    assert result["params"]["tools"][0]["input_schema"] == {
        "type": "object",
        "additionalProperties": False,
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }


def test_anthropic_batch_request_combines_multiple_system_messages() -> None:
    request = BatchRequest[TypelessResponse](
        custom_id="anthropic-multiple-system-1",
        messages=[
            {"role": "system", "content": "Follow the extraction policy."},
            {"role": "user", "content": "Extract the value."},
            {"role": "system", "content": "Return exactly one value."},
        ],
        response_model=TypelessResponse,
        model="claude-sonnet",
    )

    result = request.to_anthropic_format()

    assert result["params"]["system"] == (
        "Follow the extraction policy.\n\nReturn exactly one value."
    )
    assert result["params"]["messages"] == [
        {"role": "user", "content": "Extract the value."}
    ]


def test_anthropic_batch_request_combines_string_and_block_system_messages() -> None:
    cached_system_block = {
        "type": "text",
        "text": "Use the cached extraction policy.",
        "cache_control": {"type": "ephemeral"},
    }
    request = BatchRequest[TypelessResponse](
        custom_id="anthropic-system-blocks-1",
        messages=[
            {"role": "system", "content": "Extract conservatively."},
            {"role": "system", "content": [cached_system_block]},
            {"role": "user", "content": "Extract the value."},
            {"role": "system", "content": "Return exactly one value."},
        ],
        response_model=TypelessResponse,
        model="claude-sonnet",
    )
    original_messages = deepcopy(request.messages)

    result = request.to_anthropic_format()

    assert request.messages == original_messages
    assert result["params"]["system"] == [
        {"type": "text", "text": "Extract conservatively."},
        cached_system_block,
        {"type": "text", "text": "Return exactly one value."},
    ]
    assert result["params"]["messages"] == [
        {"role": "user", "content": "Extract the value."}
    ]


def test_anthropic_batch_request_preserves_explicit_additional_properties() -> None:
    request = BatchRequest[RestrictedResponse](
        custom_id="anthropic-restricted-1",
        messages=[{"role": "user", "content": "Extract the value."}],
        response_model=RestrictedResponse,
        model="claude-sonnet",
    )

    schema = request.to_anthropic_format()["params"]["tools"][0]["input_schema"]

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is True
    assert schema["properties"]["forbidden"] is False


def test_batch_request_rejects_an_unsupported_provider() -> None:
    request = BatchRequest[Person](
        custom_id="unknown-1",
        messages=[{"role": "user", "content": "Extract a person."}],
        response_model=Person,
        model="model",
    )

    with pytest.raises(ValueError, match="Unsupported provider: unknown"):
        request.save_to_file(io.BytesIO(), "unknown")


def test_batch_result_helpers_keep_successes_errors_and_custom_ids() -> None:
    ada = BatchSuccess(custom_id="request-1", result=Person(name="Ada", age=36))
    failure = BatchError(
        custom_id="request-2",
        error_type="rate_limit",
        error_message="try later",
        raw_data={"status": 429},
    )
    grace = BatchSuccess(custom_id="request-3", result=Person(name="Grace", age=42))
    results = [ada, failure, grace]

    assert filter_successful(results) == [ada, grace]
    assert filter_errors(results) == [failure]
    assert extract_results(results) == [ada.result, grace.result]
    assert get_results_by_custom_id(results) == {
        "request-1": ada,
        "request-2": failure,
        "request-3": grace,
    }
