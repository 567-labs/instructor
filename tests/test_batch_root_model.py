"""Batch parsing preserves object payloads when validating Pydantic root models."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, RootModel

from instructor.batch import BatchError, BatchJob, BatchProcessor, BatchSuccess


pytestmark = pytest.mark.unit


class Document(BaseModel):
    root: str
    label: str = "document"


class RootDocument(RootModel[Document]):
    pass


class DefaultDocument(Document):
    root: str = "untitled"


class RootDefaultDocument(RootModel[DefaultDocument]):
    pass


def batch_result(provider: str, payload: dict[str, object]) -> str:
    if provider == "openai":
        return json.dumps(
            {
                "custom_id": "document-1",
                "response": {
                    "body": {"choices": [{"message": {"content": json.dumps(payload)}}]}
                },
            }
        )
    return json.dumps(
        {
            "custom_id": "document-1",
            "result": {
                "type": "succeeded",
                "message": {"content": [{"type": "tool_use", "input": payload}]},
            },
        }
    )


@pytest.mark.parametrize("input_present", [False, True])
def test_anthropic_missing_tool_input_is_not_an_empty_result(
    input_present: bool,
) -> None:
    record = json.loads(batch_result("anthropic", {}))
    tool = record["result"]["message"]["content"][0]
    if input_present:
        tool["input"] = None
    else:
        del tool["input"]
    content = json.dumps(record)

    results = BatchProcessor("anthropic/model", RootDefaultDocument).parse_results(
        content
    )

    assert len(results) == 1
    assert isinstance(results[0], BatchError)
    assert results[0].custom_id == "document-1"
    assert results[0].raw_data == record
    assert BatchJob.parse_from_string(content, RootDefaultDocument) == ([], [record])


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("payload", [[1], "invalid", 17])
def test_invalid_non_object_payload_preserves_request_id(
    provider: str, payload: object
) -> None:
    record = json.loads(batch_result(provider, {}))
    if provider == "openai":
        record["response"]["body"]["choices"][0]["message"]["content"] = json.dumps(
            payload
        )
    else:
        record["result"]["message"]["content"] = [
            {"type": "text", "text": json.dumps(payload)}
        ]
    results = BatchProcessor(f"{provider}/model", RootDocument).parse_results(
        json.dumps(record) + "\n" + batch_result(provider, {"root": "Ada"})
    )

    assert len(results) == 2
    assert isinstance(results[0], BatchError)
    assert results[0].custom_id == "document-1"
    assert results[0].error_type == "parsing_error"
    assert results[0].raw_data == record
    assert isinstance(results[1], BatchSuccess)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("payload", [{"root": "Ada"}, {"root": "Ada", "label": "note"}])
@pytest.mark.parametrize("response_model", [Document, RootDocument])
def test_batch_parsing_validates_root_model_object(
    provider: str,
    payload: dict[str, object],
    response_model: type[Document] | type[RootDocument],
) -> None:
    processor = BatchProcessor(f"{provider}/model", response_model)

    results = processor.parse_results(batch_result(provider, payload))

    assert len(results) == 1
    assert isinstance(results[0], BatchSuccess)
    assert results[0].custom_id == "document-1"
    assert results[0].result == response_model.model_validate(payload)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_batch_parsing_preserves_invalid_root_model_error(provider: str) -> None:
    payload: dict[str, object] = {"root": 17}
    processor = BatchProcessor(f"{provider}/model", RootDocument)

    results = processor.parse_results(batch_result(provider, payload))

    assert len(results) == 1
    assert isinstance(results[0], BatchError)
    assert results[0].custom_id == "document-1"
    assert results[0].error_type == "parsing_error"
    assert results[0].raw_data == payload


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("payload", [{"root": "Ada"}, {"root": "Ada", "label": "note"}])
@pytest.mark.parametrize("response_model", [Document, RootDocument])
def test_batch_job_string_parser_validates_root_model_object(
    provider: str,
    payload: dict[str, object],
    response_model: type[Document] | type[RootDocument],
) -> None:
    results, errors = BatchJob.parse_from_string(
        batch_result(provider, payload), response_model
    )

    assert results == [response_model.model_validate(payload)]
    assert errors == []


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_batch_job_file_parser_preserves_root_model_and_invalid_record(
    provider: str, tmp_path: Path
) -> None:
    valid = batch_result(provider, {"root": "Ada"})
    invalid = batch_result(provider, {"root": 17})
    result_file = tmp_path / "results.jsonl"
    result_file.write_text(f"{valid}\n{invalid}\n", encoding="utf-8")

    results, errors = BatchJob.parse_from_file(str(result_file), RootDocument)

    assert results == [RootDocument(root=Document(root="Ada"))]
    assert errors == [json.loads(invalid)]


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("response_model", [DefaultDocument, RootDefaultDocument])
def test_batch_parsing_validates_empty_object_with_defaults(
    provider: str,
    response_model: type[DefaultDocument] | type[RootDefaultDocument],
) -> None:
    processor = BatchProcessor(f"{provider}/model", response_model)

    results = processor.parse_results(batch_result(provider, {}))

    assert len(results) == 1
    assert isinstance(results[0], BatchSuccess)
    assert results[0].result == response_model.model_validate({})


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("response_model", [DefaultDocument, RootDefaultDocument])
def test_batch_job_parses_empty_object_with_defaults(
    provider: str,
    response_model: type[DefaultDocument] | type[RootDefaultDocument],
    tmp_path: Path,
) -> None:
    record = batch_result(provider, {})
    result_file = tmp_path / "results.jsonl"
    result_file.write_text(record, encoding="utf-8")

    expected = ([response_model.model_validate({})], [])
    assert BatchJob.parse_from_string(record, response_model) == expected
    assert BatchJob.parse_from_file(str(result_file), response_model) == expected
