"""Mistral reasoning models return content as [ThinkChunk, TextChunk] lists."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import pytest
from pydantic import BaseModel

from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.registry import mode_registry
from instructor.v2.dsl.iterable import IterableModel

mistral_models = pytest.importorskip("mistralai.client.models")


class User(BaseModel):
    name: str
    age: int


def _thinking_then(text: str) -> list[Any]:
    return [
        mistral_models.ThinkChunk(
            thinking=[mistral_models.TextChunk(text='draft: {"name": "x"}')]
        ),
        mistral_models.TextChunk(text=text),
    ]


def _response(content: list[Any]) -> Any:
    return mistral_models.ChatCompletionResponse(
        id="r",
        object="chat.completion",
        model="magistral-medium-latest",
        created=1,
        usage=mistral_models.UsageInfo(
            prompt_tokens=1, completion_tokens=1, total_tokens=2
        ),
        choices=[
            mistral_models.ChatCompletionChoice(
                index=0,
                finish_reason="stop",
                message=mistral_models.AssistantMessage(content=content),
            )
        ],
    )


@pytest.mark.parametrize(
    ("mode", "text"),
    [
        (Mode.JSON_SCHEMA, '{"name": "Ada", "age": 36}'),
        (Mode.MD_JSON, '```json\n{"name": "Ada", "age": 36}\n```'),
    ],
)
def test_parse_list_content_skips_thinking(mode: Mode, text: str) -> None:
    handler = mode_registry.get_handlers(Provider.MISTRAL, mode)

    result = handler.response_parser(_response(_thinking_then(text)), User)

    assert result == User(name="Ada", age=36)


def _stream_events() -> list[Any]:
    def event(content: Any) -> Any:
        return mistral_models.CompletionEvent(
            data=mistral_models.CompletionChunk(
                id="c",
                model="magistral-medium-latest",
                choices=[
                    mistral_models.CompletionResponseStreamChoice(
                        index=0,
                        delta=mistral_models.DeltaMessage(content=content),
                        finish_reason=None,
                    )
                ],
            )
        )

    return [
        event(
            [
                mistral_models.ThinkChunk(
                    thinking=[mistral_models.TextChunk(text="thinking")]
                )
            ]
        ),
        event([mistral_models.TextChunk(text='{"tasks": [{"name": "Ada", ')]),
        event(None),
        event('"age": 36}]}'),
    ]


async def _aiter(items: list[Any]) -> AsyncGenerator[Any, None]:
    for item in items:
        yield item


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.asyncio
async def test_stream_list_deltas_skip_thinking(is_async: bool) -> None:
    handler = mode_registry.get_handlers(Provider.MISTRAL, Mode.JSON_SCHEMA)
    model = IterableModel(User)

    if is_async:
        stream: Any = handler.response_parser(
            _aiter(_stream_events()), model, stream=True, is_async=True
        )
        received = [item async for item in stream]
    else:
        received = list(
            handler.response_parser(iter(_stream_events()), model, stream=True)
        )

    assert received == [User(name="Ada", age=36)]
