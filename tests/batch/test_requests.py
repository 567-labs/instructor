"""Batch request serialization and the BatchProcessor API, across providers."""

from __future__ import annotations

import io
import json
import stat
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field

import instructor.batch.processor as processor_module
from instructor.batch import BatchProcessor, BatchRequest, BatchSuccess
from instructor.batch.models import BatchJobInfo

pytestmark = pytest.mark.unit

ADA = [{"role": "user", "content": "Ada is 36"}]
LIN = [{"role": "user", "content": "Lin is 28"}]


class Person(BaseModel):
    name: str
    age: int


class Team(BaseModel):
    members: list[Person]


class TypelessResponse(BaseModel):
    value: str

    @classmethod
    def model_json_schema(cls, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"properties": {"value": {"type": "string"}}, "required": ["value"]}


class RestrictedResponse(BaseModel):
    value: str

    @classmethod
    def model_json_schema(cls, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": True,
            "properties": {"value": {"type": "string"}, "forbidden": False},
            "required": ["value"],
        }


def request(
    response_model: Any = Person, messages: Any = ADA, **options: Any
) -> BatchRequest[Any]:
    return BatchRequest[Any](
        custom_id="request-1",
        messages=messages,
        response_model=response_model,
        model="test-model",
        **options,
    )


def rows(output: str | io.BytesIO) -> list[dict[str, Any]]:
    if isinstance(output, io.BytesIO):
        text = output.getvalue().decode("utf-8")
    else:
        text = Path(output).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# --- request formats ----------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "envelope"),
    [
        ("openai", {"custom_id", "method", "url", "body"}),
        ("anthropic", {"custom_id", "params"}),
        ("mistral", {"custom_id", "body"}),
    ],
)
@pytest.mark.parametrize("target", ["buffer", "path"])
def test_save_to_file_appends_one_row_per_request(
    provider: str, envelope: set[str], target: str, tmp_path: Path
) -> None:
    path = tmp_path / "requests.jsonl"
    buffer = io.BytesIO()
    for index in range(3):
        BatchRequest[Person](
            custom_id=f"request-{index}",
            messages=ADA,
            response_model=Person,
            model="test-model",
        ).save_to_file(buffer if target == "buffer" else str(path), provider)

    saved = rows(buffer if target == "buffer" else str(path))

    assert [row["custom_id"] for row in saved] == [
        "request-0",
        "request-1",
        "request-2",
    ]
    assert all(set(row) == envelope for row in saved)


@pytest.mark.parametrize("target", [123, "unknown-provider"])
def test_save_to_file_rejects_unknown_targets(target: Any) -> None:
    if target == 123:
        with pytest.raises(ValueError, match="Unsupported file_path_or_buffer type"):
            request().save_to_file(target, "openai")
    else:
        with pytest.raises(ValueError, match="Unsupported provider: unknown"):
            request().save_to_file(io.BytesIO(), "unknown")


def test_openai_format_makes_nested_schemas_strict_and_keeps_booleans() -> None:
    result = request(Team).to_openai_format()
    schema = result["body"]["response_format"]["json_schema"]["schema"]
    restricted = request(RestrictedResponse).to_openai_format()["body"]

    assert (result["method"], result["url"]) == ("POST", "/v1/chat/completions")
    assert result["body"]["model"] == "test-model"
    assert schema["additionalProperties"] is False
    assert schema["properties"]["members"]["type"] == "array"
    assert schema["$defs"]["Person"]["additionalProperties"] is False
    restricted_schema = restricted["response_format"]["json_schema"]["schema"]
    assert restricted_schema["additionalProperties"] is False
    assert restricted_schema["properties"]["forbidden"] is False


def test_openai_non_strict_format_keeps_mapping_value_schema() -> None:
    class Inventory(BaseModel):
        counts: dict[str, int]

    class Warehouse(BaseModel):
        inventories: list[Inventory]

    body = request(Warehouse, strict=False).to_openai_format()["body"]
    schema = body["response_format"]["json_schema"]["schema"]

    assert "additionalProperties" not in schema
    assert "additionalProperties" not in schema["$defs"]["Inventory"]
    # Mapping values live in additionalProperties and must survive.
    counts = schema["$defs"]["Inventory"]["properties"]["counts"]
    assert counts["additionalProperties"] == {"type": "integer"}


@pytest.mark.parametrize(
    ("temperature", "anthropic_has", "openai_value"),
    [(None, False, None), (0.0, True, 0.0), (0.1, True, 0.1), (1.0, True, 1.0)],
)
def test_temperature_is_omitted_only_for_anthropic_none(
    temperature: float | None, anthropic_has: bool, openai_value: float | None
) -> None:
    params = request(temperature=temperature).to_anthropic_format()["params"]

    assert ("temperature" in params) is anthropic_has
    if anthropic_has:
        assert params["temperature"] == temperature
    assert (
        request(temperature=temperature).to_openai_format()["body"]["temperature"]
        == openai_value
    )
    assert request().to_anthropic_format()["params"]["temperature"] == 0.1


@pytest.mark.parametrize(
    ("messages", "system"),
    [
        (
            [
                {"role": "system", "content": "Return one value."},
                {"role": "user", "content": "Extract."},
            ],
            "Return one value.",
        ),
        (
            [
                {"role": "system", "content": "Follow policy."},
                {"role": "user", "content": "Extract."},
                {"role": "system", "content": "Return one value."},
            ],
            "Follow policy.\n\nReturn one value.",
        ),
        (
            [
                {"role": "system", "content": "Be careful."},
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "Cached policy.",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
                {"role": "user", "content": "Extract."},
                {"role": "system", "content": "Return one value."},
            ],
            [
                {"type": "text", "text": "Be careful."},
                {
                    "type": "text",
                    "text": "Cached policy.",
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": "Return one value."},
            ],
        ),
    ],
    ids=["single", "multiple-strings", "strings-and-blocks"],
)
def test_anthropic_format_moves_system_messages_out_of_messages(
    messages: list[dict[str, Any]], system: Any
) -> None:
    batch_request = request(TypelessResponse, messages=messages)
    original = deepcopy(batch_request.messages)

    params = batch_request.to_anthropic_format()["params"]

    assert params["system"] == system
    assert params["messages"] == [{"role": "user", "content": "Extract."}]
    assert batch_request.messages == original
    assert params["tools"][0]["input_schema"] == {
        "type": "object",
        "additionalProperties": False,
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }


def test_anthropic_format_keeps_explicit_additional_properties() -> None:
    params = request(RestrictedResponse).to_anthropic_format()["params"]
    schema = params["tools"][0]["input_schema"]

    assert params["model"] == "test-model"
    assert (schema["type"], schema["additionalProperties"]) == ("object", True)
    assert schema["properties"]["forbidden"] is False


def test_mistral_format_carries_strict_schema_but_no_model() -> None:
    body = request(Team, max_tokens=90, temperature=0.2).to_mistral_format()["body"]
    json_schema = body["response_format"]["json_schema"]
    schema = json_schema["schema"]

    # Mistral sets the model on the job, not per request.
    assert "model" not in body
    assert (body["messages"], body["max_tokens"], body["temperature"]) == (ADA, 90, 0.2)
    assert body["response_format"]["type"] == "json_schema"
    assert (json_schema["name"], json_schema["strict"]) == ("Team", True)
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["Person"]["additionalProperties"] is False


def test_mistral_sdk_one_response_format_serializes_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mistralai import extra

    class LegacyFormat(BaseModel):
        schema_value: dict[str, str] = Field(alias="schema")

    legacy = LegacyFormat.model_validate({"schema": {"type": "object"}})
    monkeypatch.setattr(
        extra, "response_format_from_pydantic_model", lambda _model: legacy
    )

    body = request().to_mistral_format()["body"]

    assert body["response_format"] == {"schema": {"type": "object"}}


# --- BatchProcessor -----------------------------------------------------------


class RecordingProvider:
    """A BatchProvider that records each forwarded call."""

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
        return [BatchJobInfo.from_openai({"id": "batch-123", "status": "completed"})]


def test_processor_splits_model_string_and_rejects_bad_format() -> None:
    processor = BatchProcessor("openai/gpt-4.1-mini", Person)

    assert (processor.provider_name, processor.model_name) == ("openai", "gpt-4.1-mini")
    assert processor.response_model is Person
    with pytest.raises(ValueError, match='format "provider/model-name"'):
        BatchProcessor("gpt-4.1-mini", Person)


@pytest.mark.parametrize(
    ("model", "section", "model_field"),
    [
        ("openai/gpt-4.1-mini", "body", "gpt-4.1-mini"),
        ("anthropic/claude-sonnet", "params", "claude-sonnet"),
        ("mistral/mistral-small", "body", None),
    ],
)
@pytest.mark.parametrize("target", ["buffer", "new-file", "existing-file"])
@pytest.mark.parametrize("custom_ids", [None, ["user_ada", "user-lin"]])
def test_create_batch_serializes_every_conversation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    model: str,
    section: str,
    model_field: str | None,
    target: str,
    custom_ids: list[str] | None,
) -> None:
    path = tmp_path / "requests.jsonl"
    if target == "existing-file":
        path.write_text("stale request\n")
        path.chmod(0o644)
    processor = BatchProcessor(model, Person)

    output = processor.create_batch_from_messages(
        [ADA, LIN],
        None if target == "buffer" else str(path),
        max_tokens=321,
        temperature=0.3,
        custom_ids=custom_ids,
    )

    saved = rows(output)
    assert [row["custom_id"] for row in saved] == (
        custom_ids or ["request-0", "request-1"]
    )
    assert [row[section]["messages"] for row in saved] == [ADA, LIN]
    assert [row[section].get("model") for row in saved] == [model_field] * 2
    assert saved[0][section]["max_tokens"] == 321
    assert saved[0][section]["temperature"] == 0.3
    printed = capsys.readouterr().out
    if target == "buffer":
        assert isinstance(output, io.BytesIO) and output.tell() == 0
        assert "Created batch buffer with 2 requests" in printed
    else:
        assert output == str(path)
        assert "Created batch file" in printed
        assert list(tmp_path.iterdir()) == [path]
    if target == "existing-file":
        assert stat.S_IMODE(path.stat().st_mode) == 0o644


@pytest.mark.parametrize("file_backed", [False, True])
def test_anthropic_batch_omits_none_temperature(
    file_backed: bool, tmp_path: Path
) -> None:
    path = tmp_path / "batch.jsonl"
    output = BatchProcessor("anthropic/test-model", Person).create_batch_from_messages(
        [ADA], str(path) if file_backed else None, temperature=None
    )

    (row,) = rows(output)
    assert "temperature" not in row["params"]
    assert row["params"]["max_tokens"] == 1000


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
    tmp_path: Path, model: str, custom_ids: list[str], match: str
) -> None:
    path = tmp_path / "requests.jsonl"
    path.write_text("existing\n")

    with pytest.raises(ValueError, match=match):
        BatchProcessor(model, Person).create_batch_from_messages(
            [ADA, LIN], str(path), custom_ids=custom_ids
        )

    assert path.read_text() == "existing\n"


@pytest.mark.parametrize("existing", [False, True])
def test_create_batch_file_is_atomic_on_serialization_failure(
    tmp_path: Path, existing: bool
) -> None:
    path = tmp_path / "requests.jsonl"
    if existing:
        path.write_text("previous batch\n")

    with pytest.raises(TypeError):
        BatchProcessor("openai/gpt-4.1-mini", Person).create_batch_from_messages(
            [ADA, [{"role": "user", "content": object()}]], str(path)
        )

    assert list(tmp_path.iterdir()) == ([path] if existing else [])
    if existing:
        assert path.read_text() == "previous batch\n"


def test_processor_forwards_operations_and_parses_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider = RecordingProvider(
        json.dumps(
            {
                "custom_id": "request-0",
                "response": {
                    "body": {
                        "choices": [{"message": {"content": '{"name":"Ada","age":36}'}}]
                    }
                },
            }
        )
    )
    monkeypatch.setattr(processor_module, "get_provider", lambda _name: provider)
    processor = BatchProcessor("openai/gpt-4.1-mini", Person)
    assert processor.provider is provider
    buffer = io.BytesIO(b"request")
    results_file = tmp_path / "results.jsonl"
    expected = [BatchSuccess(custom_id="request-0", result=Person(name="Ada", age=36))]

    assert processor.submit_batch(buffer, completion_window="24h") == "batch-123"
    assert processor.submit_batch("requests.jsonl", metadata={"team": "x"}) == (
        "batch-123"
    )
    assert processor.get_batch_status("batch-123")["status"] == "completed"
    assert [job.id for job in processor.list_batches(limit=2)] == ["batch-123"]
    assert processor.get_results("batch-123", str(results_file)) == expected
    assert results_file.read_text() == provider.results
    assert processor.get_results("batch-123") == expected
    assert processor.cancel_batch("batch-123")["status"] == "cancelled"
    assert processor.delete_batch("batch-123")["deleted"] is True
    assert provider.calls == [
        (
            "submit",
            (
                buffer,
                {"description": "Instructor batch job"},
                {"completion_window": "24h"},
            ),
        ),
        ("submit", ("requests.jsonl", {"team": "x"}, {})),
        ("status", "batch-123"),
        ("list", 2),
        ("retrieve", "batch-123"),
        ("download", ("batch-123", str(results_file))),
        ("retrieve", "batch-123"),
        ("cancel", "batch-123"),
        ("delete", "batch-123"),
    ]
