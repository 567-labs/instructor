"""Focused tests for the public batch processor API and result parsing."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

import instructor.batch.processor as processor_module
from instructor.batch.models import BatchError, BatchJobInfo, BatchSuccess
from instructor.batch.processor import BatchProcessor


pytestmark = pytest.mark.unit


class Person(BaseModel):
    name: str
    age: int


class DefaultedResult(BaseModel):
    status: str = "ok"


class RecordingProvider:
    def __init__(self, results: str = "") -> None:
        self.results = results
        self.calls: list[tuple[str, Any]] = []

    def submit_batch(
        self,
        file_path_or_buffer: str | io.BytesIO,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        self.calls.append(("submit", (file_path_or_buffer, metadata, kwargs)))
        return "batch-123"

    def get_status(self, batch_id: str) -> dict[str, Any]:
        self.calls.append(("status", batch_id))
        return {"id": batch_id, "status": "completed"}

    def retrieve_results(self, batch_id: str) -> str:
        self.calls.append(("retrieve", batch_id))
        return self.results

    def download_results(self, batch_id: str, file_path: str) -> None:
        self.calls.append(("download", (batch_id, file_path)))
        Path(file_path).write_text(self.results)

    def cancel_batch(self, batch_id: str) -> dict[str, Any]:
        self.calls.append(("cancel", batch_id))
        return {"id": batch_id, "status": "cancelled"}

    def delete_batch(self, batch_id: str) -> dict[str, Any]:
        self.calls.append(("delete", batch_id))
        return {"id": batch_id, "deleted": True}

    def list_batches(self, limit: int = 10) -> list[BatchJobInfo]:
        self.calls.append(("list", limit))
        return [
            BatchJobInfo.from_openai(
                {"id": "batch-123", "status": "completed", "metadata": {}}
            )
        ]


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> RecordingProvider:
    provider = RecordingProvider()
    monkeypatch.setattr(processor_module, "get_provider", lambda _: provider)
    return provider


def openai_result(custom_id: str, content: str) -> str:
    return json.dumps(
        {
            "custom_id": custom_id,
            "response": {"body": {"choices": [{"message": {"content": content}}]}},
        }
    )


def test_init_splits_provider_and_model_and_rejects_invalid_model(
    provider: RecordingProvider,
) -> None:
    processor = BatchProcessor("openai/gpt-4.1-mini", Person)

    assert processor.provider is provider
    assert processor.provider_name == "openai"
    assert processor.model_name == "gpt-4.1-mini"
    assert processor.response_model is Person

    with pytest.raises(
        ValueError, match='Model string must be in format "provider/model-name"'
    ):
        BatchProcessor("gpt-4.1-mini", Person)


def test_create_batch_file_replaces_existing_contents_and_serializes_requests(
    provider: RecordingProvider, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    del provider
    batch_file = tmp_path / "requests.jsonl"
    batch_file.write_text("stale request\n")
    processor = BatchProcessor("openai/gpt-4.1-mini", Person)

    returned = processor.create_batch_from_messages(
        [
            [{"role": "user", "content": "Ada is 36"}],
            [{"role": "user", "content": "Lin is 28"}],
        ],
        str(batch_file),
        max_tokens=321,
        temperature=0.3,
    )

    lines = [json.loads(line) for line in batch_file.read_text().splitlines()]
    assert returned == str(batch_file)
    assert [line["custom_id"] for line in lines] == ["request-0", "request-1"]
    assert [line["body"]["model"] for line in lines] == [
        "gpt-4.1-mini",
        "gpt-4.1-mini",
    ]
    assert lines[0]["body"]["max_tokens"] == 321
    assert lines[0]["body"]["temperature"] == 0.3
    assert "Created batch file" in capsys.readouterr().out


def test_create_batch_file_creates_a_new_output_file(
    provider: RecordingProvider, tmp_path: Path
) -> None:
    del provider
    batch_file = tmp_path / "new-requests.jsonl"
    processor = BatchProcessor("openai/gpt-4.1-mini", Person)

    returned = processor.create_batch_from_messages(
        [[{"role": "user", "content": "Ada is 36"}]], str(batch_file)
    )

    assert returned == str(batch_file)
    request = json.loads(batch_file.read_text())
    assert request["custom_id"] == "request-0"
    assert request["body"]["messages"] == [{"role": "user", "content": "Ada is 36"}]


def test_create_batch_buffer_is_readable_from_start(
    provider: RecordingProvider, capsys: pytest.CaptureFixture[str]
) -> None:
    del provider
    processor = BatchProcessor("anthropic/claude-sonnet", Person)

    buffer = processor.create_batch_from_messages(
        [[{"role": "user", "content": "Ada is 36"}]], max_tokens=90, temperature=0.2
    )

    assert isinstance(buffer, io.BytesIO)
    assert buffer.tell() == 0
    request = json.loads(buffer.read().decode())
    assert request["custom_id"] == "request-0"
    assert request["params"]["model"] == "claude-sonnet"
    assert request["params"]["max_tokens"] == 90
    assert request["params"]["temperature"] == 0.2
    assert "Created batch buffer with 1 requests" in capsys.readouterr().out


@pytest.mark.parametrize("model", ["openai/gpt-4.1-mini", "anthropic/claude-sonnet"])
def test_create_batch_uses_custom_ids(provider: RecordingProvider, model: str) -> None:
    del provider
    processor = BatchProcessor(model, Person)

    buffer = processor.create_batch_from_messages(
        [
            [{"role": "user", "content": "Ada is 36"}],
            [{"role": "user", "content": "Lin is 28"}],
        ],
        custom_ids=["user_ada", "user-lin"],
    )

    assert isinstance(buffer, io.BytesIO)
    lines = [json.loads(line) for line in buffer.read().decode().splitlines()]
    assert [line["custom_id"] for line in lines] == ["user_ada", "user-lin"]


@pytest.mark.parametrize(
    ("model", "custom_ids", "match"),
    [
        ("openai/gpt-4.1-mini", ["only-one"], "has 1 items but messages_list has 2"),
        ("openai/gpt-4.1-mini", ["same", "same"], "must be unique"),
        ("anthropic/claude-sonnet", ["ok", "user:42"], "Anthropic custom_ids"),
        ("anthropic/claude-sonnet", ["ok", "x" * 65], "Anthropic custom_ids"),
    ],
)
def test_create_batch_rejects_invalid_custom_ids_before_writing(
    provider: RecordingProvider,
    tmp_path: Path,
    model: str,
    custom_ids: list[str],
    match: str,
) -> None:
    del provider
    batch_file = tmp_path / "requests.jsonl"
    batch_file.write_text("existing\n")
    processor = BatchProcessor(model, Person)

    with pytest.raises(ValueError, match=match):
        processor.create_batch_from_messages(
            [
                [{"role": "user", "content": "Ada is 36"}],
                [{"role": "user", "content": "Lin is 28"}],
            ],
            str(batch_file),
            custom_ids=custom_ids,
        )

    assert batch_file.read_text() == "existing\n"


def test_provider_operations_forward_arguments_and_parse_downloaded_results(
    provider: RecordingProvider, tmp_path: Path
) -> None:
    provider.results = openai_result("request-0", '{"name": "Ada", "age": 36}')
    processor = BatchProcessor("openai/gpt-4.1-mini", Person)
    request_buffer = io.BytesIO(b"request")

    assert (
        processor.submit_batch(request_buffer, completion_window="24h") == "batch-123"
    )
    assert provider.calls[-1] == (
        "submit",
        (
            request_buffer,
            {"description": "Instructor batch job"},
            {"completion_window": "24h"},
        ),
    )
    assert processor.submit_batch("requests.jsonl", metadata={"team": "search"}) == (
        "batch-123"
    )
    assert provider.calls[-1] == (
        "submit",
        ("requests.jsonl", {"team": "search"}, {}),
    )
    assert processor.get_batch_status("batch-123") == {
        "id": "batch-123",
        "status": "completed",
    }
    jobs = processor.list_batches(limit=2)
    assert len(jobs) == 1
    assert jobs[0].id == "batch-123"
    assert provider.calls[-1] == ("list", 2)

    results_file = tmp_path / "results.jsonl"
    results = processor.get_results("batch-123", str(results_file))
    assert isinstance(results[0], BatchSuccess)
    assert results[0].result == Person(name="Ada", age=36)
    assert results_file.read_text() == provider.results
    assert ("retrieve", "batch-123") in provider.calls
    assert ("download", ("batch-123", str(results_file))) in provider.calls

    calls_before = len(provider.calls)
    in_memory_results = processor.get_results("batch-123")
    assert isinstance(in_memory_results[0], BatchSuccess)
    assert provider.calls[calls_before:] == [("retrieve", "batch-123")]
    assert processor.cancel_batch("batch-123") == {
        "id": "batch-123",
        "status": "cancelled",
    }
    assert processor.delete_batch("batch-123") == {
        "id": "batch-123",
        "deleted": True,
    }
    assert provider.calls == [
        (
            "submit",
            (
                request_buffer,
                {"description": "Instructor batch job"},
                {"completion_window": "24h"},
            ),
        ),
        ("submit", ("requests.jsonl", {"team": "search"}, {})),
        ("status", "batch-123"),
        ("list", 2),
        ("retrieve", "batch-123"),
        ("download", ("batch-123", str(results_file))),
        ("retrieve", "batch-123"),
        ("cancel", "batch-123"),
        ("delete", "batch-123"),
    ]


def test_openai_results_distinguish_success_validation_extraction_and_json_errors(
    provider: RecordingProvider,
) -> None:
    del provider
    processor = BatchProcessor("openai/gpt-4.1-mini", Person)
    content = "\n".join(
        [
            openai_result("ok", '{"name": "Ada", "age": 36}'),
            "   ",
            openai_result("invalid-model", '{"name": "Ada", "age": "unknown"}'),
            json.dumps({"custom_id": "missing-response"}),
            "not-json",
        ]
    )

    results = processor.parse_results(content)

    assert len(results) == 4
    assert isinstance(results[0], BatchSuccess)
    assert results[0].custom_id == "ok"
    assert results[0].result == Person(name="Ada", age=36)
    assert isinstance(results[1], BatchError)
    assert results[1].custom_id == "invalid-model"
    assert results[1].error_type == "parsing_error"
    assert "Failed to parse into Person" in results[1].error_message
    assert results[1].raw_data == {"name": "Ada", "age": "unknown"}
    assert isinstance(results[2], BatchError)
    assert results[2].custom_id == "missing-response"
    assert results[2].error_type == "extraction_error"
    assert results[2].error_message == "Unknown error"
    assert isinstance(results[3], BatchError)
    assert results[3].custom_id == "unknown"
    assert results[3].error_type == "json_parse_error"
    assert results[3].raw_data == {"raw_line": "not-json"}


@pytest.mark.parametrize(
    ("model", "content"),
    [
        ("openai/gpt-4.1-mini", openai_result("empty", "{}")),
        (
            "anthropic/claude-sonnet",
            json.dumps(
                {
                    "custom_id": "empty",
                    "result": {
                        "type": "succeeded",
                        "message": {"content": [{"type": "tool_use", "input": {}}]},
                    },
                }
            ),
        ),
    ],
)
def test_parse_results_accepts_empty_object_for_defaulted_model(
    provider: RecordingProvider,
    model: str,
    content: str,
) -> None:
    del provider
    processor = BatchProcessor(model, DefaultedResult)

    results = processor.parse_results(content)

    assert results == [
        BatchSuccess(custom_id="empty", result=DefaultedResult(status="ok"))
    ]


def test_anthropic_results_support_tool_use_and_text_fallback(
    provider: RecordingProvider,
) -> None:
    del provider
    processor = BatchProcessor("anthropic/claude-sonnet", Person)
    tool_result = {
        "custom_id": "tool",
        "result": {
            "type": "succeeded",
            "message": {
                "content": [
                    {"type": "text", "text": "not json"},
                    {"type": "tool_use", "input": {"name": "Ada", "age": 36}},
                ]
            },
        },
    }
    text_result = {
        "custom_id": "text",
        "result": {
            "type": "succeeded",
            "message": {
                "content": [
                    {"type": "text", "text": "not json"},
                    {"type": "text", "text": '{"name": "Lin", "age": 28}'},
                ]
            },
        },
    }

    results = processor.parse_results(
        "\n".join([json.dumps(tool_result), json.dumps(text_result)])
    )

    assert len(results) == 2
    assert isinstance(results[0], BatchSuccess)
    assert isinstance(results[1], BatchSuccess)
    assert results[0].custom_id == "tool"
    assert results[0].result == Person(name="Ada", age=36)
    assert results[1].custom_id == "text"
    assert results[1].result == Person(name="Lin", age=28)


@pytest.mark.parametrize(
    ("result", "expected_type", "expected_message"),
    [
        (
            {
                "type": "error",
                "error": {
                    "error": {"type": "rate_limit_error", "message": "slow down"}
                },
            },
            "rate_limit_error",
            "slow down",
        ),
        (
            {"type": "error", "error": "service unavailable"},
            "anthropic_error",
            "service unavailable",
        ),
        (
            {"type": "succeeded", "message": {"content": []}},
            "extraction_error",
            "Unknown error",
        ),
    ],
)
def test_anthropic_error_and_empty_results_keep_provider_details(
    provider: RecordingProvider,
    result: dict[str, Any],
    expected_type: str,
    expected_message: str,
) -> None:
    del provider
    processor = BatchProcessor("anthropic/claude-sonnet", Person)

    parsed = processor.parse_results(
        json.dumps({"custom_id": "request-7", "result": result})
    )

    assert len(parsed) == 1
    assert isinstance(parsed[0], BatchError)
    assert parsed[0].custom_id == "request-7"
    assert parsed[0].error_type == expected_type
    assert parsed[0].error_message == expected_message


def test_extract_returns_none_for_missing_malformed_or_unknown_provider_responses(
    provider: RecordingProvider,
) -> None:
    del provider
    anthropic = BatchProcessor("anthropic/claude-sonnet", Person)
    openai = BatchProcessor("openai/gpt-4.1-mini", Person)
    unknown = BatchProcessor("local/model", Person)

    assert anthropic._extract_from_response({"custom_id": "no-result"}) is None
    assert (
        anthropic._extract_from_response(
            {"result": {"type": "succeeded", "message": {"content": "not-a-list"}}}
        )
        is None
    )
    assert anthropic._extract_from_response({"result": {"type": "succeeded"}}) is None
    assert (
        anthropic._extract_from_response(
            {
                "result": {
                    "type": "succeeded",
                    "message": {"content": [{"type": "image", "source": {}}]},
                }
            }
        )
        is None
    )
    assert anthropic._extract_from_response(
        {
            "result": {
                "type": "succeeded",
                "message": {
                    "content": [
                        {"type": "image", "source": {}},
                        {"type": "text", "text": '{"name":"Ada","age":36}'},
                    ]
                },
            }
        }
    ) == {"name": "Ada", "age": 36}
    assert (
        openai._extract_from_response({"response": {"body": {"choices": []}}}) is None
    )
    assert unknown._extract_from_response({"anything": "goes"}) is None
