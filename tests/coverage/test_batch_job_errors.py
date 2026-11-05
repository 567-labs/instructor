"""OpenAI batch-level errors retain their SDK error details."""

from typing import Any

import pytest
from openai.types import Batch
from openai.types.batch import BatchRequestCounts

from instructor.batch.models import BatchJobInfo, BatchStatus


def batch_payload(errors: Any) -> dict[str, Any]:
    return Batch(
        id="batch-test",
        completion_window="24h",
        created_at=1720000000,
        endpoint="/v1/chat/completions",
        input_file_id="file-test",
        object="batch",
        status="failed",
        request_counts=BatchRequestCounts(total=1, completed=0, failed=1),
        metadata={},
        errors=errors,
    ).model_dump()


@pytest.mark.parametrize("count", [1, 2])
def test_openai_batch_uses_first_validation_error(count: int) -> None:
    errors = [
        {
            "code": f"invalid_request_{index}",
            "message": f"Invalid request at line {index}",
            "line": index,
            "param": "messages",
        }
        for index in range(1, count + 1)
    ]
    payload = batch_payload({"object": "list", "data": errors})

    job = BatchJobInfo.from_openai(payload)

    assert job.status == BatchStatus.FAILED
    assert job.error is not None
    assert job.error.error_code == "invalid_request_1"
    assert job.error.error_message == "Invalid request at line 1"
    assert job.error.error_type is None
    assert job.raw_data == payload
    assert job.raw_data is not None
    assert job.raw_data["errors"]["data"] == errors


@pytest.mark.parametrize("include_errors", [False, True])
def test_openai_batch_without_errors(include_errors: bool) -> None:
    payload = batch_payload(None)
    if not include_errors:
        payload.pop("errors")

    assert BatchJobInfo.from_openai(payload).error is None


def test_openai_batch_preserves_flat_error_mapping() -> None:
    payload = batch_payload(None)
    payload["errors"] = {"type": "validation", "code": "invalid", "message": "Bad file"}

    job = BatchJobInfo.from_openai(payload)

    assert job.error is not None
    assert job.error.error_type == "validation"
    assert job.error.error_code == "invalid"
    assert job.error.error_message == "Bad file"


def test_openai_batch_empty_error_list_retains_existing_behavior() -> None:
    job = BatchJobInfo.from_openai(batch_payload({"object": "list", "data": []}))

    assert job.error is not None
    assert job.error.model_dump() == {
        "error_type": None,
        "error_message": None,
        "error_code": None,
    }
