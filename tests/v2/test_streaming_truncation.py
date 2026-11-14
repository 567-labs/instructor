"""Iterable streams must report a token limit that lands between complete items.

When a provider stops for its token limit right after an item closes, every
item that reached the caller parses fine and the stream used to end silently,
so callers could not tell the list was cut short. Iterable streams now raise
``IncompleteOutputException`` after flushing the complete items. Partial
streams keep their best-effort behavior.

Covers OpenAI chat, OpenAI Responses, Cohere and Mistral. Anthropic is covered
in ``test_anthropic_streaming_truncation.py``.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from typing import Any

import pytest
from pydantic import BaseModel

from instructor.v2.core.errors import IncompleteOutputException
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.registry import mode_registry
from instructor.v2.dsl.iterable import IterableModel
from instructor.v2.dsl.partial import Partial


class Item(BaseModel):
    name: str


ITEMS_JSON = '{"tasks": [{"name": "Ada"}'


# --- chunk builders (real SDK objects) -------------------------------------


def _openai_chat_chunks(mode: Mode, stop: str) -> list[Any]:
    from openai.types.chat import ChatCompletionChunk

    def chunk(delta: dict[str, Any], finish_reason: str | None) -> Any:
        return ChatCompletionChunk.model_validate(
            {
                "id": "c",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "m",
                "choices": [
                    {"index": 0, "delta": delta, "finish_reason": finish_reason}
                ],
            }
        )

    if mode is Mode.TOOLS:
        delta = {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call-0",
                    "type": "function",
                    "function": {"name": "Item", "arguments": ITEMS_JSON},
                }
            ]
        }
    elif mode is Mode.MD_JSON:
        delta = {"content": "```json\n" + ITEMS_JSON}
    else:
        delta = {"content": ITEMS_JSON}
    return [chunk(delta, None), chunk({}, stop)]


def _openai_responses_chunks(mode: Mode, stop: str) -> list[Any]:  # noqa: ARG001
    from openai.types.responses import (
        Response,
        ResponseCompletedEvent,
        ResponseFunctionCallArgumentsDeltaEvent,
        ResponseFunctionToolCall,
        ResponseIncompleteEvent,
        ResponseOutputItemAddedEvent,
    )
    from openai.types.responses.response import IncompleteDetails

    events: list[Any] = [
        ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            output_index=0,
            sequence_number=0,
            item=ResponseFunctionToolCall(
                type="function_call",
                call_id="call-0",
                name="IterableItem",
                arguments="",
            ),
        ),
        ResponseFunctionCallArgumentsDeltaEvent(
            type="response.function_call_arguments.delta",
            item_id="fc-0",
            output_index=0,
            sequence_number=1,
            delta=ITEMS_JSON,
        ),
    ]
    if stop == "length":
        response = Response.model_construct(
            status="incomplete",
            incomplete_details=IncompleteDetails(reason="max_output_tokens"),
        )
        events.append(
            ResponseIncompleteEvent(
                type="response.incomplete", sequence_number=2, response=response
            )
        )
    else:
        events.append(
            ResponseCompletedEvent(
                type="response.completed",
                sequence_number=2,
                response=Response.model_construct(status="completed"),
            )
        )
    return events


def _cohere_chunks(mode: Mode, stop: str) -> list[Any]:  # noqa: ARG001
    from cohere import (
        ChatContentDeltaEventDelta,
        ChatContentDeltaEventDeltaMessage,
        ChatContentDeltaEventDeltaMessageContent,
        ChatMessageEndEventDelta,
        ContentDeltaStreamedChatResponseV2,
        MessageEndStreamedChatResponseV2,
    )

    return [
        ContentDeltaStreamedChatResponseV2(
            type="content-delta",
            index=0,
            delta=ChatContentDeltaEventDelta(
                message=ChatContentDeltaEventDeltaMessage(
                    content=ChatContentDeltaEventDeltaMessageContent(text=ITEMS_JSON)
                )
            ),
        ),
        MessageEndStreamedChatResponseV2(
            type="message-end",
            delta=ChatMessageEndEventDelta(
                finish_reason="MAX_TOKENS" if stop == "length" else "COMPLETE"
            ),
        ),
    ]


def _mistral_chunks(mode: Mode, stop: str) -> list[Any]:
    from mistralai.client.models import CompletionEvent

    def chunk(delta: dict[str, Any], finish_reason: str | None) -> Any:
        return CompletionEvent.model_validate(
            {
                "data": {
                    "id": "c",
                    "model": "m",
                    "choices": [
                        {"index": 0, "delta": delta, "finish_reason": finish_reason}
                    ],
                }
            }
        )

    if mode is Mode.TOOLS:
        # Mistral streams whole tool-call arguments per chunk.
        arguments = ITEMS_JSON + "]}"
        delta = {
            "tool_calls": [
                {"id": "t", "function": {"name": "Item", "arguments": arguments}}
            ]
        }
    elif mode is Mode.MD_JSON:
        delta = {"content": "```json\n" + ITEMS_JSON}
    else:
        delta = {"content": ITEMS_JSON}
    # The final chunk carries no content (delta.content is None).
    return [chunk(delta, None), chunk({}, stop)]


CASES: list[tuple[Provider, Mode, Callable[[Mode, str], list[Any]], str]] = [
    (Provider.OPENAI, Mode.TOOLS, _openai_chat_chunks, "stop"),
    (Provider.OPENAI, Mode.JSON, _openai_chat_chunks, "stop"),
    (Provider.OPENAI, Mode.JSON_SCHEMA, _openai_chat_chunks, "stop"),
    (Provider.OPENAI, Mode.MD_JSON, _openai_chat_chunks, "stop"),
    (Provider.OPENAI, Mode.RESPONSES_TOOLS, _openai_responses_chunks, "stop"),
    (Provider.COHERE, Mode.TOOLS, _cohere_chunks, "stop"),
    (Provider.MISTRAL, Mode.TOOLS, _mistral_chunks, "stop"),
    (Provider.MISTRAL, Mode.JSON_SCHEMA, _mistral_chunks, "stop"),
    (Provider.MISTRAL, Mode.MD_JSON, _mistral_chunks, "stop"),
]
CASE_IDS = [f"{p.value}-{m.value}" for p, m, _, _ in CASES]


async def _aiter(chunks: list[Any]) -> AsyncGenerator[Any, None]:
    for chunk in chunks:
        yield chunk


async def _collect(
    provider: Provider, mode: Mode, model: Any, chunks: list[Any], is_async: bool
) -> list[Any]:
    parser = mode_registry.get_handlers(provider, mode).response_parser
    received: list[Any] = []
    if is_async:
        result = parser(_aiter(chunks), model, stream=True, is_async=True)
        async for item in result:
            received.append(item)
    else:
        for item in parser(iter(chunks), model, stream=True):
            received.append(item)
    return received


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(("provider", "mode", "build", "ok_stop"), CASES, ids=CASE_IDS)
@pytest.mark.asyncio
async def test_iterable_raises_on_token_limit_after_complete_items(
    provider: Provider,
    mode: Mode,
    build: Callable[[Mode, str], list[Any]],
    ok_stop: str,  # noqa: ARG001
    is_async: bool,
) -> None:
    chunks = build(mode, "length")
    received: list[Any] = []

    with pytest.raises(IncompleteOutputException) as exc_info:
        parser: Any = mode_registry.get_handlers(provider, mode).response_parser
        if is_async:
            async for item in parser(
                _aiter(chunks), IterableModel(Item), stream=True, is_async=True
            ):
                received.append(item)
        else:
            for item in parser(iter(chunks), IterableModel(Item), stream=True):
                received.append(item)

    assert received == [Item(name="Ada")]
    last = exc_info.value.last_completion
    assert last is not None
    if mode is Mode.RESPONSES_TOOLS:
        assert last is chunks[-1].response
        assert "max_output_tokens" in str(exc_info.value)
    else:
        assert last is chunks[-1]


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(("provider", "mode", "build", "ok_stop"), CASES, ids=CASE_IDS)
@pytest.mark.asyncio
async def test_iterable_normal_stop_ends_cleanly(
    provider: Provider,
    mode: Mode,
    build: Callable[[Mode, str], list[Any]],
    ok_stop: str,
    is_async: bool,
) -> None:
    chunks = build(mode, ok_stop)
    received = await _collect(provider, mode, IterableModel(Item), chunks, is_async)
    assert received == [Item(name="Ada")]


PARTIAL_CASES = [
    (Provider.OPENAI, Mode.JSON, _openai_chat_chunks),
    (Provider.OPENAI, Mode.TOOLS, _openai_chat_chunks),
    (Provider.COHERE, Mode.TOOLS, _cohere_chunks),
    (Provider.MISTRAL, Mode.JSON_SCHEMA, _mistral_chunks),
]


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("provider", "mode", "build"),
    PARTIAL_CASES,
    ids=[f"{p.value}-{m.value}" for p, m, _ in PARTIAL_CASES],
)
@pytest.mark.asyncio
async def test_partial_keeps_best_effort_on_token_limit(
    provider: Provider,
    mode: Mode,
    build: Callable[[Mode, str], list[Any]],
    is_async: bool,
) -> None:
    chunks = build(mode, "length")
    received = await _collect(provider, mode, Partial[Item], chunks, is_async)
    assert received
