"""Anthropic batch error records must retain their provider details."""

import json
from typing import Any

import pytest
from anthropic.types.messages import MessageBatchIndividualResponse
from pydantic import BaseModel

from instructor.batch import BatchProcessor
from instructor.batch.models import BatchError, BatchSuccess

pytestmark = pytest.mark.unit


class Person(BaseModel):
    name: str


def errored_result(error_type: str, message: str) -> MessageBatchIndividualResponse:
    return MessageBatchIndividualResponse.model_validate(
        {
            "custom_id": "failed-request",
            "result": {
                "type": "errored",
                "error": {
                    "type": "error",
                    "error": {"type": error_type, "message": message},
                },
            },
        }
    )


@pytest.mark.parametrize(
    ("error_type", "message"),
    [
        ("invalid_request_error", "max_tokens must be greater than zero"),
        ("rate_limit_error", "Too many requests"),
    ],
)
def test_parse_anthropic_sdk_error_record(error_type: str, message: str) -> None:
    record = errored_result(error_type, message)
    processor = BatchProcessor("anthropic/claude-sonnet-4-5", Person)

    results = processor.parse_results(record.model_dump_json())

    assert results == [
        BatchError(
            custom_id=record.custom_id,
            error_type=error_type,
            error_message=message,
            raw_data=record.model_dump(mode="json"),
        )
    ]


def test_parse_mixed_anthropic_batch_preserves_success_and_error_details() -> None:
    failed = errored_result("invalid_request_error", "Invalid request")
    succeeded: dict[str, Any] = {
        "custom_id": "successful-request",
        "result": {
            "type": "succeeded",
            "message": {
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5",
                "content": [{"type": "text", "text": '{"name": "Ada"}'}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        },
    }
    success_record = MessageBatchIndividualResponse.model_validate(succeeded)
    result_lines = success_record.model_dump_json() + "\n" + failed.model_dump_json()
    processor = BatchProcessor("anthropic/claude-sonnet-4-5", Person)

    results = processor.parse_results(result_lines)

    assert len(results) == 2
    assert results[0] == BatchSuccess(
        custom_id="successful-request", result=Person(name="Ada")
    )
    assert isinstance(results[1], BatchError)
    assert results[1].custom_id == "failed-request"
    assert results[1].error_type == "invalid_request_error"
    assert results[1].error_message == "Invalid request"
    assert results[1].raw_data == failed.model_dump(mode="json")


def test_parse_legacy_anthropic_error_record() -> None:
    record = {
        "custom_id": "legacy-request",
        "result": {
            "type": "error",
            "error": {
                "error": {"type": "invalid_request_error", "message": "Invalid request"}
            },
        },
    }
    processor = BatchProcessor("anthropic/claude-sonnet-4-5", Person)

    results = processor.parse_results(json.dumps(record))

    assert results == [
        BatchError(
            custom_id="legacy-request",
            error_type="invalid_request_error",
            error_message="Invalid request",
            raw_data=record,
        )
    ]
