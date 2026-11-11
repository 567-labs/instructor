"""Parse real Mistral SDK text blocks without treating reasoning as output."""

from __future__ import annotations

import importlib
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import BaseModel

from instructor import Partial
from instructor.batch import BatchProcessor, BatchSuccess
from instructor.v2.providers.mistral.handlers import (
    MistralJSONSchemaHandler,
    MistralMDJSONHandler,
)

pytest.importorskip("mistralai")
try:
    models = importlib.import_module("mistralai.client.models")
except ImportError:
    models = importlib.import_module("mistralai.models")


class Person(BaseModel):
    name: str


def completion(content: Any) -> Any:
    return models.ChatCompletionResponse.model_validate(
        {
            "id": "completion-1",
            "object": "chat.completion",
            "model": "magistral-small-latest",
            "created": 0,
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
        }
    )


def thinking_blocks() -> list[dict[str, Any]]:
    # Older supported SDKs predate reasoning models, but still support text blocks.
    if not hasattr(models, "ThinkChunk"):
        return []
    return [
        {
            "type": "thinking",
            "thinking": [{"type": "text", "text": '{"name":"Wrong"}'}],
        },
    ]


def blocks(text: str) -> list[dict[str, Any]]:
    return [
        *thinking_blocks(),
        {"type": "text", "text": text[: len(text) // 2]},
        {"type": "text", "text": text[len(text) // 2 :]},
    ]


@pytest.mark.parametrize(
    "handler_type", [MistralJSONSchemaHandler, MistralMDJSONHandler]
)
@pytest.mark.parametrize("blocked", [False, True], ids=["string", "blocks"])
def test_complete_text_content(handler_type: Any, blocked: bool) -> None:
    text = '{"name":"Ada"}'
    if handler_type is MistralMDJSONHandler:
        text = f"```json\n{text}\n```"
    response = completion(blocks(text) if blocked else text)
    original = response.model_dump()
    parsed = handler_type().parse_response(response, Person)
    assert parsed == Person(name="Ada")
    assert response.model_dump() == original


def stream_events(text: str, blocked: bool) -> list[Any]:
    content_chunks: list[Any] = [None, ""]
    if blocked:
        content_chunks.append(thinking_blocks())
    for char in text:
        content_chunks.append([{"type": "text", "text": char}] if blocked else char)
    content_chunks.append(None)
    return [
        models.CompletionEvent.model_validate(
            {
                "data": {
                    "id": "completion-1",
                    "model": "magistral-small-latest",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": None,
                            "delta": {"content": content},
                        }
                    ],
                }
            }
        )
        for content in content_chunks
    ]


@pytest.mark.parametrize(
    "handler_type", [MistralJSONSchemaHandler, MistralMDJSONHandler]
)
@pytest.mark.parametrize("blocked", [False, True], ids=["string", "blocks"])
def test_partial_stream_text_content(handler_type: Any, blocked: bool) -> None:
    text = '{"name":"Ada"}'
    if handler_type is MistralMDJSONHandler:
        text = f"```json\n{text}\n```"
    results = handler_type().parse_response(
        iter(stream_events(text, blocked)), Partial[Person], stream=True
    )
    assert results[-1].name == "Ada"
    assert all(result.name != "Wrong" for result in results)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "handler_type", [MistralJSONSchemaHandler, MistralMDJSONHandler]
)
@pytest.mark.parametrize("blocked", [False, True], ids=["string", "blocks"])
async def test_async_partial_stream_text_content(
    handler_type: Any, blocked: bool
) -> None:
    text = '{"name":"Ada"}'
    if handler_type is MistralMDJSONHandler:
        text = f"```json\n{text}\n```"

    async def events() -> AsyncIterator[Any]:
        for event in stream_events(text, blocked):
            yield event

    results = [
        result
        async for result in handler_type().parse_response(
            events(), Partial[Person], stream=True, is_async=True
        )
    ]
    assert results[-1].name == "Ada"
    assert all(result.name != "Wrong" for result in results)


@pytest.mark.parametrize("blocked", [False, True], ids=["string", "blocks"])
def test_batch_text_content(blocked: bool) -> None:
    text = '{"name":"Ada"}'
    response = completion(blocks(text) if blocked else text)
    record = {
        "custom_id": "request-1",
        "response": {"status_code": 200, "body": response.model_dump(mode="json")},
        "error": None,
    }
    results = BatchProcessor("mistral/magistral-small-latest", Person).parse_results(
        json.dumps(record)
    )
    assert len(results) == 1
    assert isinstance(results[0], BatchSuccess)
    assert results[0].custom_id == "request-1"
    assert results[0].result == Person(name="Ada")
