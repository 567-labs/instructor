"""Completed OpenAI batches must retain failed requests alongside successes."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import openai
import pytest
from pydantic import BaseModel

from instructor.batch.models import BatchError, BatchSuccess
from instructor.batch.processor import BatchProcessor

pytestmark = pytest.mark.unit


class Answer(BaseModel):
    name: str


@pytest.mark.parametrize("all_failed", [False, True])
@pytest.mark.parametrize("operation", ["retrieve", "download", "processor"])
def test_completed_batch_preserves_error_file_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    all_failed: bool,
    operation: str,
) -> None:
    success = {
        "id": "batch_req_ok",
        "custom_id": "request-ok",
        "response": {
            "status_code": 200,
            "request_id": "req_ok",
            "body": {"choices": [{"message": {"content": '{"name":"Ada"}'}}]},
        },
        "error": None,
    }
    failure = {
        "id": "batch_req_failed",
        "custom_id": "request-failed",
        "response": None,
        "error": {"code": "request_timeout", "message": "Request timed out."},
    }
    http_failure = {
        "id": "batch_req_http_failed",
        "custom_id": "request-http-failed",
        "response": {
            "status_code": 400,
            "request_id": "req_http_failed",
            "body": {
                "error": {
                    "code": "context_length_exceeded",
                    "type": "invalid_request_error",
                    "message": "Input exceeds the context window.",
                }
            },
        },
        "error": None,
    }
    expected = ([] if all_failed else [success]) + [failure, http_failure]
    files = {
        "file_output": json.dumps(success),  # Deliberately no trailing newline.
        "file_errors": "\n".join(map(json.dumps, [failure, http_failure])) + "\n",
    }
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        requests.append(path)
        if path == "/v1/batches/batch_test":
            return httpx.Response(
                200,
                json={
                    "id": "batch_test",
                    "object": "batch",
                    "completion_window": "24h",
                    "created_at": 0,
                    "endpoint": "/v1/chat/completions",
                    "input_file_id": "file_input",
                    "status": "completed",
                    "output_file_id": None if all_failed else "file_output",
                    "error_file_id": "file_errors",
                    "request_counts": {
                        "total": len(expected),
                        "completed": int(not all_failed),
                        "failed": 2,
                    },
                },
            )
        for file_id, content in files.items():
            if path == f"/v1/files/{file_id}/content":
                return httpx.Response(200, text=content)
        raise AssertionError(f"Unexpected SDK request: {path}")

    with openai.OpenAI(
        api_key="synthetic-test-key",
        base_url="https://batch-test.invalid/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    ) as client:
        monkeypatch.setattr(openai, "OpenAI", lambda: client)
        processor = BatchProcessor("openai/test-model", Answer)
        if operation == "processor":
            results = processor.retrieve_results("batch_test")
            assert [result.custom_id for result in results] == [
                row["custom_id"] for row in expected
            ]
            if not all_failed:
                assert isinstance(results[0], BatchSuccess)
                assert results[0].result == Answer(name="Ada")
            errors = [result for result in results if isinstance(result, BatchError)]
            assert len(errors) == 2
            assert [result.error_type for result in errors] == [
                "request_timeout",
                "context_length_exceeded",
            ]
            assert [result.error_message for result in errors] == [
                failure["error"]["message"],
                http_failure["response"]["body"]["error"]["message"],
            ]
            assert [result.raw_data for result in errors] == [failure, http_failure]
        else:
            if operation == "download":
                destination = tmp_path / "results.jsonl"
                processor.provider.download_results("batch_test", str(destination))
                content = destination.read_text()
            else:
                content = processor.provider.retrieve_results("batch_test")
            assert [
                json.loads(line) for line in content.splitlines() if line
            ] == expected

    assert requests == ["/v1/batches/batch_test"] + [
        f"/v1/files/{file_id}/content"
        for file_id in (
            ["file_errors"] if all_failed else ["file_output", "file_errors"]
        )
    ]
