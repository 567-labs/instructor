"""GenAI Iterable streams: token limits, empty chunks, and thought parts.

Chunks are real ``google.genai.types.GenerateContentResponse`` objects, the
same type the SDK yields from ``generate_content_stream``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from pydantic import BaseModel

from instructor.v2.core.errors import IncompleteOutputException
from instructor.v2.core.mode import Mode
from instructor.v2.dsl.iterable import IterableModel
from instructor.v2.providers.genai.handlers import (
    GenAIStructuredOutputsHandler,
    GenAIToolsHandler,
)

types = pytest.importorskip("google.genai.types")


class Item(BaseModel):
    name: str


MODEL = IterableModel(Item)


def _chunk(
    parts: list[Any] | None = None,
    finish_reason: Any = None,
    *,
    no_content: bool = False,
) -> Any:
    content = None if no_content else types.Content(role="model", parts=parts)
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=content, finish_reason=finish_reason)]
    )


def _tool_part(names: list[str]) -> Any:
    return types.Part(
        function_call=types.FunctionCall(
            name=MODEL.__name__, args={"tasks": [{"name": n} for n in names]}
        )
    )


def _handler(mode: Mode) -> Any:
    if mode == Mode.TOOLS:
        return GenAIToolsHandler(mode=Mode.TOOLS)
    return GenAIStructuredOutputsHandler(mode=Mode.JSON)


def _item_chunks(mode: Mode) -> list[Any]:
    if mode == Mode.TOOLS:
        return [_chunk([_tool_part(["Ada"])])]
    return [
        _chunk([types.Part(text='{"tasks": [{"name": "Ada"}')]),
        _chunk([types.Part(text=", ")]),
    ]


async def _collect(
    mode: Mode, chunks: list[Any], is_async: bool
) -> tuple[list[Item], Exception | None]:
    handler = _handler(mode)
    received: list[Item] = []
    try:
        if is_async:

            async def events() -> AsyncGenerator[Any, None]:
                for chunk in chunks:
                    yield chunk

            async for item in handler.parse_response(
                events(), MODEL, stream=True, is_async=True
            ):
                received.append(item)
        else:
            for item in handler.parse_response(iter(chunks), MODEL, stream=True):
                received.append(item)
    except Exception as exc:  # noqa: BLE001
        return received, exc
    return received, None


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("mode", [Mode.TOOLS, Mode.JSON])
@pytest.mark.parametrize("no_content", [False, True])
async def test_iterable_stream_reports_max_tokens_after_complete_items(
    mode: Mode, is_async: bool, no_content: bool
) -> None:
    terminal = _chunk(
        [], finish_reason=types.FinishReason.MAX_TOKENS, no_content=no_content
    )

    received, error = await _collect(mode, [*_item_chunks(mode), terminal], is_async)

    assert received == [Item(name="Ada")]
    assert isinstance(error, IncompleteOutputException)
    assert error.last_completion is terminal


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("mode", [Mode.TOOLS, Mode.JSON])
async def test_iterable_stream_finishing_normally_does_not_raise(
    mode: Mode, is_async: bool
) -> None:
    terminal = _chunk([], finish_reason=types.FinishReason.STOP)
    tail = [] if mode == Mode.TOOLS else [_chunk([types.Part(text="]}")])]

    received, error = await _collect(
        mode, [*_item_chunks(mode), *tail, terminal], is_async
    )

    assert error is None
    assert received == [Item(name="Ada")]


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("mode", [Mode.TOOLS, Mode.JSON])
async def test_stream_skips_chunks_without_parts_or_text(
    mode: Mode, is_async: bool
) -> None:
    empty = [
        _chunk(None),  # content.parts is None
        _chunk([]),
        _chunk([types.Part(thought=True, text="thinking")]),
    ]
    if mode == Mode.TOOLS:
        chunks = [*empty, _chunk([_tool_part(["Ada", "Bob"])])]
    else:
        chunks = [
            empty[0],
            _chunk([types.Part(text='{"tasks": [{"name": "Ada"}, ')]),
            empty[1],
            _chunk([types.Part(text='{"name": "Bob"}]}')]),
        ]

    received, error = await _collect(mode, chunks, is_async)

    assert error is None
    assert received == [Item(name="Ada"), Item(name="Bob")]


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("placement", ["before", "after"])
async def test_tools_stream_uses_function_call_next_to_thought_part(
    is_async: bool, placement: str
) -> None:
    thought = types.Part(thought=True, text="checking")
    call = _tool_part(["Ada"])
    parts = [thought, call] if placement == "before" else [call, thought]

    received, error = await _collect(Mode.TOOLS, [_chunk(parts)], is_async)

    assert error is None
    assert received == [Item(name="Ada")]


def test_extract_streaming_json_skips_thought_part() -> None:
    handler = GenAIToolsHandler(mode=Mode.TOOLS)
    chunk = _chunk([types.Part(thought=True, text="x"), _tool_part(["Ada"])])

    assert list(handler.extract_streaming_json([chunk])) == [
        json.dumps({"tasks": [{"name": "Ada"}]})
    ]
