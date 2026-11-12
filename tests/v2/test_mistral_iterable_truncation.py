"""Preserve complete streamed items before reporting Mistral token exhaustion."""

from __future__ import annotations

import importlib
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from pydantic import BaseModel

from instructor import Partial
from instructor.v2.core.errors import IncompleteOutputException
from instructor.v2.dsl.iterable import IterableModel
from instructor.v2.providers.mistral.handlers import (
    MistralJSONSchemaHandler,
    MistralMDJSONHandler,
    MistralToolsHandler,
)

pytest.importorskip("mistralai")
try:
    models = importlib.import_module("mistralai.client.models")
except ImportError:
    models = importlib.import_module("mistralai.models")


class Person(BaseModel):
    name: str


HANDLERS = [MistralJSONSchemaHandler, MistralMDJSONHandler, MistralToolsHandler]
CASES = [
    (handler, location)
    for handler in HANDLERS
    for location in ["after-items", "after-null", "terminal"]
]
CASES += [(MistralMDJSONHandler, "bare-terminal")]
CASES += [
    (handler, "between-items")
    for handler in [MistralJSONSchemaHandler, MistralMDJSONHandler]
]
TEXT = '{"tasks":[{"name":"Ada"},{"name":"Grace"}]}'


def event(content: str | None, finish_reason: str | None, *, tools: bool) -> Any:
    delta: dict[str, Any] = {"content": content}
    if tools:
        delta = {}
        if content:
            delta["tool_calls"] = [
                {
                    "index": 0,
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "IterablePerson", "arguments": content},
                }
            ]
    return models.CompletionEvent.model_validate(
        {
            "data": {
                "id": "local-completion",
                "model": "mistral-small-latest",
                "choices": [
                    {"index": 0, "delta": delta, "finish_reason": finish_reason}
                ],
            }
        }
    )


def metadata_event() -> Any:
    return models.CompletionEvent.model_validate(
        {
            "data": {
                "id": "local-completion",
                "model": "mistral-small-latest",
                "choices": [],
            }
        }
    )


def events(handler_type: Any, location: str, finish_reason: str) -> list[Any]:
    tools = handler_type is MistralToolsHandler
    text = TEXT
    if handler_type is MistralMDJSONHandler and location != "bare-terminal":
        text = f"```json\n{text}\n```"
    chunks = [metadata_event()]
    if location in {"after-items", "after-null"}:
        chunks.append(event(text, None, tools=tools))
        chunks.append(
            event(None if location == "after-null" else "", finish_reason, tools=tools)
        )
    elif location == "between-items":
        first = '{"tasks":[{"name":"Ada"},'
        if handler_type is MistralMDJSONHandler:
            first = "```json\n" + first
        chunks.append(event(first, None, tools=False))
        chunks.append(event('{"name":"Grace"},', finish_reason, tools=False))
    else:
        chunks.append(event(text, finish_reason, tools=tools))
    chunks.append(metadata_event())
    return chunks


@pytest.mark.parametrize("handler_type,location", CASES)
@pytest.mark.parametrize("finish_reason", ["length", "stop"])
def test_iterable_stream_reports_truncation_after_complete_items(
    handler_type: Any, location: str, finish_reason: str
) -> None:
    chunks = events(handler_type, location, finish_reason)
    closed = False

    def stream() -> Iterator[Any]:
        nonlocal closed
        try:
            yield from chunks
        finally:
            closed = True

    result = handler_type().parse_response(stream(), IterableModel(Person), stream=True)
    assert next(result) == Person(name="Ada")
    assert next(result) == Person(name="Grace")
    if finish_reason == "length":
        with pytest.raises(IncompleteOutputException) as exc_info:
            next(result)
        assert exc_info.value.last_completion is chunks[-2]
    else:
        with pytest.raises(StopIteration):
            next(result)
    assert closed


@pytest.mark.asyncio
@pytest.mark.parametrize("handler_type,location", CASES)
@pytest.mark.parametrize("finish_reason", ["length", "stop"])
async def test_async_iterable_stream_reports_truncation_after_complete_items(
    handler_type: Any, location: str, finish_reason: str
) -> None:
    chunks = events(handler_type, location, finish_reason)

    closed = False

    async def stream() -> AsyncIterator[Any]:
        nonlocal closed
        try:
            for chunk in chunks:
                yield chunk
        finally:
            closed = True

    result = handler_type().parse_response(
        stream(), IterableModel(Person), stream=True, is_async=True
    )
    assert await anext(result) == Person(name="Ada")
    assert await anext(result) == Person(name="Grace")
    if finish_reason == "length":
        with pytest.raises(IncompleteOutputException) as exc_info:
            await anext(result)
        assert exc_info.value.last_completion is chunks[-2]
    else:
        with pytest.raises(StopAsyncIteration):
            await anext(result)
    assert closed


@pytest.mark.parametrize(
    "handler_type", [MistralJSONSchemaHandler, MistralMDJSONHandler]
)
def test_partial_stream_keeps_tolerating_length(handler_type: Any) -> None:
    text = '{"name":"Ada"'
    if handler_type is MistralMDJSONHandler:
        text = f"```json\n{text}\n```"
    result = handler_type().parse_response(
        iter([event(text, "length", tools=False)]), Partial[Person], stream=True
    )
    assert result[-1].name == "Ada"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "handler_type", [MistralJSONSchemaHandler, MistralMDJSONHandler]
)
async def test_async_partial_stream_keeps_tolerating_length(handler_type: Any) -> None:
    text = '{"name":"Ada"'
    if handler_type is MistralMDJSONHandler:
        text = f"```json\n{text}\n```"

    async def stream() -> AsyncIterator[Any]:
        yield event(text, "length", tools=False)

    result = [
        item
        async for item in handler_type().parse_response(
            stream(), Partial[Person], stream=True, is_async=True
        )
    ]
    assert result[-1].name == "Ada"
