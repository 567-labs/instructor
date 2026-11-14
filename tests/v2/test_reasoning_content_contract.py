"""Contract: reasoning/thinking content is skipped when extracting structured output.

Covers Mistral ThinkChunk content (non-streaming and streaming), GenAI thought
parts and empty chunks, and OpenAI Responses ``commentary``-phase messages.
All inputs are real SDK objects fed straight into the registered parser.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import pytest
from pydantic import BaseModel

from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.registry import mode_registry
from instructor.v2.dsl.iterable import IterableModel
from instructor.v2.dsl.partial import Partial


class Item(BaseModel):
    name: str


async def _aiter(chunks: list[Any]) -> AsyncGenerator[Any, None]:
    for chunk in chunks:
        yield chunk


async def _stream(
    provider: Provider, mode: Mode, model: Any, chunks: list[Any], is_async: bool
) -> list[Any]:
    parser: Any = mode_registry.get_handlers(provider, mode).response_parser
    if is_async:
        stream = parser(_aiter(chunks), model, stream=True, is_async=True)
        return [item async for item in stream]
    return list(parser(iter(chunks), model, stream=True))


sync_and_async = pytest.mark.parametrize(
    "is_async", [False, True], ids=["sync", "async"]
)

# --- Mistral: content as [ThinkChunk, TextChunk] ---------------------------


@pytest.mark.parametrize(
    ("mode", "text"),
    [
        (Mode.JSON_SCHEMA, '{"name": "Ada"}'),
        (Mode.MD_JSON, '```json\n{"name": "Ada"}\n```'),
    ],
)
def test_mistral_list_content_skips_thinking(mode: Mode, text: str) -> None:
    m = pytest.importorskip("mistralai.client.models")
    think = m.ThinkChunk(thinking=[m.TextChunk(text='draft: {"name": "x"}')])
    response = m.ChatCompletionResponse(
        id="r",
        object="chat.completion",
        model="magistral",
        created=1,
        usage=m.UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        choices=[
            m.ChatCompletionChoice(
                index=0,
                finish_reason="stop",
                message=m.AssistantMessage(content=[think, m.TextChunk(text=text)]),
            )
        ],
    )

    parser = mode_registry.get_handlers(Provider.MISTRAL, mode).response_parser
    assert parser(response, Item) == Item(name="Ada")


@sync_and_async
@pytest.mark.asyncio
async def test_mistral_stream_list_deltas_skip_thinking(is_async: bool) -> None:
    m = pytest.importorskip("mistralai.client.models")

    def event(content: Any) -> Any:
        choice = m.CompletionResponseStreamChoice(
            index=0, delta=m.DeltaMessage(content=content), finish_reason=None
        )
        return m.CompletionEvent(
            data=m.CompletionChunk(id="c", model="magistral", choices=[choice])
        )

    chunks = [
        event([m.ThinkChunk(thinking=[m.TextChunk(text="thinking")])]),
        event([m.TextChunk(text='{"tasks": [{"name": ')]),
        event(None),
        event('"Ada"}]}'),
    ]

    received = await _stream(
        Provider.MISTRAL, Mode.JSON_SCHEMA, IterableModel(Item), chunks, is_async
    )
    assert received == [Item(name="Ada")]


# --- GenAI: thought parts and chunks without parts -------------------------


def _genai_chunk(parts: list[Any] | None) -> Any:
    types = pytest.importorskip("google.genai.types")
    content = types.Content(role="model", parts=parts)
    return types.GenerateContentResponse(candidates=[types.Candidate(content=content)])


def _genai_call(names: list[str]) -> Any:
    types = pytest.importorskip("google.genai.types")
    args = {"tasks": [{"name": n} for n in names]}
    return types.Part(function_call=types.FunctionCall(name="IterableItem", args=args))


def _genai_thought() -> Any:
    types = pytest.importorskip("google.genai.types")
    return types.Part(thought=True, text='checking {"name": "x"}')


def _genai_text(text: str) -> Any:
    types = pytest.importorskip("google.genai.types")
    return types.Part(text=text)


@sync_and_async
@pytest.mark.parametrize("mode", [Mode.TOOLS, Mode.JSON])
@pytest.mark.asyncio
async def test_genai_stream_skips_thought_parts_and_empty_chunks(
    mode: Mode, is_async: bool
) -> None:
    empty = [_genai_chunk(None), _genai_chunk([]), _genai_chunk([_genai_thought()])]
    if mode is Mode.TOOLS:
        chunks = [*empty, _genai_chunk([_genai_call(["Ada", "Bob"])])]
    else:
        chunks = [
            empty[0],
            _genai_chunk([_genai_text('{"tasks": [{"name": "Ada"}, ')]),
            empty[1],
            empty[2],
            _genai_chunk([_genai_text('{"name": "Bob"}]}')]),
        ]

    received = await _stream(
        Provider.GENAI, mode, IterableModel(Item), chunks, is_async
    )
    assert received == [Item(name="Ada"), Item(name="Bob")]


@sync_and_async
@pytest.mark.parametrize("thought_first", [True, False], ids=["before", "after"])
@pytest.mark.asyncio
async def test_genai_tools_stream_uses_call_next_to_thought_part(
    thought_first: bool, is_async: bool
) -> None:
    parts = [_genai_thought(), _genai_call(["Ada"])]
    chunk = _genai_chunk(parts if thought_first else parts[::-1])

    received = await _stream(
        Provider.GENAI, Mode.TOOLS, IterableModel(Item), [chunk], is_async
    )
    assert received == [Item(name="Ada")]


# --- OpenAI Responses: commentary-phase message precedes the answer --------


def test_openai_responses_stream_skips_commentary_message() -> None:
    from openai.types.responses import (
        ResponseOutputItemAddedEvent,
        ResponseOutputMessage,
        ResponseTextDeltaEvent,
    )

    def message(index: int, phase: Any, text: str) -> list[Any]:
        item = ResponseOutputMessage(
            id=f"msg-{index}",
            type="message",
            role="assistant",
            status="in_progress",
            content=[],
            phase=phase,
        )
        return [
            ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                output_index=index,
                sequence_number=index * 2,
                item=item,
            ),
            ResponseTextDeltaEvent(
                type="response.output_text.delta",
                item_id=f"msg-{index}",
                output_index=index,
                content_index=0,
                sequence_number=index * 2 + 1,
                delta=text,
                logprobs=[],
            ),
        ]

    events = [
        *message(0, "commentary", 'Checking {"name": "x"} first.'),
        *message(1, "final_answer", '{"name": "Ada"}'),
    ]
    handlers = mode_registry.get_handlers(Provider.OPENAI, Mode.RESPONSES_TOOLS)
    parser: Any = handlers.response_parser
    results = list(parser(iter(events), Partial[Item], stream=True))

    assert results[-1].name == "Ada"
