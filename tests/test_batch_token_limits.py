"""Batch parsing must reject token-limited responses instead of
marking them successful when model defaults fill the missing fields."""

import json

from pydantic import BaseModel

from instructor.batch import BatchProcessor
from instructor.batch.models import BatchError, BatchSuccess

import pytest

pytestmark = pytest.mark.unit


class Result(BaseModel):
    status: str = "ok"


def openai_record(finish_reason: str, content: str, custom_id: str = "request-1"):
    return {
        "custom_id": custom_id,
        "response": {
            "body": {
                "choices": [
                    {"finish_reason": finish_reason, "message": {"content": content}}
                ]
            }
        },
    }


def anthropic_record(stop_reason: str, content: list, custom_id: str = "request-2"):
    return {
        "custom_id": custom_id,
        "result": {
            "type": "succeeded",
            "message": {"stop_reason": stop_reason, "content": content},
        },
    }


def test_openai_truncated_response_is_rejected() -> None:
    record = openai_record("length", "{}")

    results = BatchProcessor("openai/test-model", Result).parse_results(
        json.dumps(record)
    )

    assert len(results) == 1
    error = results[0]
    assert isinstance(error, BatchError)
    assert error.custom_id == "request-1"
    assert error.error_type == "token_limit"
    assert error.raw_data == record


def test_anthropic_truncated_tool_input_is_rejected() -> None:
    record = anthropic_record(
        "max_tokens", [{"type": "tool_use", "input": {}}]
    )

    results = BatchProcessor("anthropic/claude-3-sonnet", Result).parse_results(
        json.dumps(record)
    )

    assert isinstance(results[0], BatchError)
    assert results[0].error_type == "token_limit"


def test_anthropic_truncated_json_text_is_rejected() -> None:
    record = anthropic_record("max_tokens", [{"type": "text", "text": "{}"}])

    results = BatchProcessor("anthropic/claude-3-sonnet", Result).parse_results(
        json.dumps(record)
    )

    assert isinstance(results[0], BatchError)
    assert results[0].error_type == "token_limit"


def test_complete_responses_still_succeed() -> None:
    results = BatchProcessor("openai/test-model", Result).parse_results(
        json.dumps(openai_record("stop", '{"status": "done"}'))
    )
    assert isinstance(results[0], BatchSuccess)
    assert results[0].result.status == "done"

    results = BatchProcessor("anthropic/claude-3-sonnet", Result).parse_results(
        json.dumps(
            anthropic_record(
                "end_turn", [{"type": "text", "text": '{"status": "done"}'}]
            )
        )
    )
    assert isinstance(results[0], BatchSuccess)
    assert results[0].result.status == "done"
