"""Contract: a provider token limit raises instead of returning truncated output.

Streams: Iterable streams flush every complete item, then raise
``IncompleteOutputException`` when the provider stopped for its token limit.
Partial streams stay best-effort. Non-streaming JSON/tool modes raise on the
token-limit stop reason. Most cases use real SDK objects fed straight into the
registered response parser; the Google-style cases use duck-typed chunks, and
one Gemini case calls the handler directly.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import BaseModel

from instructor.v2.core.errors import IncompleteOutputException
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.registry import mode_registry
from instructor.v2.dsl.iterable import IterableModel
from instructor.v2.dsl.partial import Partial
from tests.coverage._openai import chat_completion
from tests.v2.test_bedrock_handlers import (
    Answer,
    _bedrock_text_response,
    _bedrock_tool_response,
)


class Item(BaseModel):
    name: str


class Items(BaseModel):
    tasks: list[Item]


ITEMS_JSON = '{"tasks": [{"name": "Ada"}'

# --- stream builders: one complete item, then a limit or normal stop -------


def _openai_chat(mode: Mode) -> Callable[[bool], list[Any]]:
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
        call = {"name": "Item", "arguments": ITEMS_JSON}
        delta: dict[str, Any] = {
            "tool_calls": [
                {"index": 0, "id": "c0", "type": "function", "function": call}
            ]
        }
    else:
        delta = {"content": ITEMS_JSON}
    return lambda limit: [chunk(delta, None), chunk({}, "length" if limit else "stop")]


def _openai_responses(limit: bool) -> list[Any]:
    from openai.types.responses import (
        Response,
        ResponseCompletedEvent,
        ResponseFunctionCallArgumentsDeltaEvent,
        ResponseIncompleteEvent,
    )
    from openai.types.responses.response import IncompleteDetails

    end: Any = (
        ResponseIncompleteEvent(
            type="response.incomplete",
            sequence_number=2,
            response=Response.model_construct(
                status="incomplete",
                incomplete_details=IncompleteDetails(reason="max_output_tokens"),
            ),
        )
        if limit
        else ResponseCompletedEvent(
            type="response.completed",
            sequence_number=2,
            response=Response.model_construct(status="completed"),
        )
    )
    return [
        ResponseFunctionCallArgumentsDeltaEvent(
            type="response.function_call_arguments.delta",
            item_id="fc-0",
            output_index=0,
            sequence_number=1,
            delta=ITEMS_JSON,
        ),
        end,
    ]


def _anthropic(limit: bool) -> list[Any]:
    from anthropic.types import (
        InputJSONDelta,
        RawContentBlockDeltaEvent,
        RawMessageDeltaEvent,
    )
    from anthropic.types.raw_message_delta_event import Delta

    stop: Any = "max_tokens" if limit else "end_turn"
    return [
        RawContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=InputJSONDelta(type="input_json_delta", partial_json=ITEMS_JSON),
        ),
        RawMessageDeltaEvent(
            type="message_delta",
            delta=Delta(stop_reason=stop, stop_sequence=None),
            usage=cast(Any, {"output_tokens": 1}),
        ),
    ]


def _genai(mode: Mode, *, no_content: bool = False) -> Callable[[bool], list[Any]]:
    def build(limit: bool) -> list[Any]:
        types = pytest.importorskip("google.genai.types")

        def chunk(parts: list[Any] | None, finish: Any = None, empty: bool = False):
            content = None if empty else types.Content(role="model", parts=parts)
            return types.GenerateContentResponse(
                candidates=[types.Candidate(content=content, finish_reason=finish)]
            )

        if mode is Mode.TOOLS:
            call = types.FunctionCall(
                name="IterableItem", args={"tasks": [{"name": "Ada"}]}
            )
            body = [chunk([types.Part(function_call=call)])]
        else:
            tail = ", " if limit else "]}"
            body = [
                chunk([types.Part(text=ITEMS_JSON)]),
                chunk([types.Part(text=tail)]),
            ]
        finish = types.FinishReason.MAX_TOKENS if limit else types.FinishReason.STOP
        return [*body, chunk([], finish, empty=no_content)]

    return build


def _cohere(limit: bool) -> list[Any]:
    cohere = pytest.importorskip("cohere")
    text = cohere.ChatContentDeltaEventDeltaMessageContent(text=ITEMS_JSON)
    return [
        cohere.ContentDeltaStreamedChatResponseV2(
            type="content-delta",
            index=0,
            delta=cohere.ChatContentDeltaEventDelta(
                message=cohere.ChatContentDeltaEventDeltaMessage(content=text)
            ),
        ),
        cohere.MessageEndStreamedChatResponseV2(
            type="message-end",
            delta=cohere.ChatMessageEndEventDelta(
                finish_reason="MAX_TOKENS" if limit else "COMPLETE"
            ),
        ),
    ]


def _mistral(mode: Mode) -> Callable[[bool], list[Any]]:
    def build(limit: bool) -> list[Any]:
        models = pytest.importorskip("mistralai.client.models")

        def chunk(delta: dict[str, Any], finish_reason: str | None) -> Any:
            choice = {"index": 0, "delta": delta, "finish_reason": finish_reason}
            return models.CompletionEvent.model_validate(
                {"data": {"id": "c", "model": "m", "choices": [choice]}}
            )

        if mode is Mode.TOOLS:
            # Mistral streams whole tool-call arguments per chunk.
            call = {"name": "Item", "arguments": ITEMS_JSON + "]}"}
            delta: dict[str, Any] = {"tool_calls": [{"id": "t", "function": call}]}
        else:
            delta = {"content": ITEMS_JSON}
        # The terminal chunk carries no content (delta.content is None).
        return [chunk(delta, None), chunk({}, "length" if limit else "stop")]

    return build


def _gemini_like() -> Callable[[bool], list[Any]]:
    """Duck-typed Google-style chunks with both text and parts accessors."""

    def build(limit: bool) -> list[Any]:
        reason = SimpleNamespace(name="MAX_TOKENS" if limit else "STOP")

        def chunk(text: str, finish: Any = None) -> Any:
            part = SimpleNamespace(text=text)
            return SimpleNamespace(
                text=text,
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(parts=[part]), finish_reason=finish
                    )
                ],
            )

        tail = ", " if limit else "]}"
        return [chunk(ITEMS_JSON), chunk(tail), chunk("", reason)]

    return build


def _gemini_like() -> Callable[[bool], list[Any]]:
    """Duck-typed Google-style chunks with both text and parts accessors."""

    def build(limit: bool) -> list[Any]:
        reason = SimpleNamespace(name="MAX_TOKENS" if limit else "STOP")

        def chunk(text: str, finish: Any = None) -> Any:
            part = SimpleNamespace(text=text)
            return SimpleNamespace(
                text=text,
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(parts=[part]), finish_reason=finish
                    )
                ],
            )

        tail = ", " if limit else "]}"
        return [chunk(ITEMS_JSON), chunk(tail), chunk("", reason)]

    return build


STREAM_CASES: list[tuple[str, Provider, Mode, Callable[[bool], list[Any]]]] = [
    ("openai-tools", Provider.OPENAI, Mode.TOOLS, _openai_chat(Mode.TOOLS)),
    ("openai-json", Provider.OPENAI, Mode.JSON, _openai_chat(Mode.JSON)),
    ("openai-responses", Provider.OPENAI, Mode.RESPONSES_TOOLS, _openai_responses),
    ("anthropic-tools", Provider.ANTHROPIC, Mode.TOOLS, _anthropic),
    (
        "vertexai-md-json",
        Provider.VERTEXAI,
        Mode.MD_JSON,
        _gemini_like(),
    ),
    ("genai-tools", Provider.GENAI, Mode.TOOLS, _genai(Mode.TOOLS)),
    ("genai-json", Provider.GENAI, Mode.JSON, _genai(Mode.JSON)),
    (
        "genai-tools-no-content",
        Provider.GENAI,
        Mode.TOOLS,
        _genai(Mode.TOOLS, no_content=True),
    ),
    (
        "genai-json-no-content",
        Provider.GENAI,
        Mode.JSON,
        _genai(Mode.JSON, no_content=True),
    ),
    ("cohere-tools", Provider.COHERE, Mode.TOOLS, _cohere),
    ("mistral-tools", Provider.MISTRAL, Mode.TOOLS, _mistral(Mode.TOOLS)),
    (
        "mistral-json-schema",
        Provider.MISTRAL,
        Mode.JSON_SCHEMA,
        _mistral(Mode.JSON_SCHEMA),
    ),
]
stream_cases = pytest.mark.parametrize(
    ("provider", "mode", "build"),
    [c[1:] for c in STREAM_CASES],
    ids=[c[0] for c in STREAM_CASES],
)
sync_and_async = pytest.mark.parametrize(
    "is_async", [False, True], ids=["sync", "async"]
)


async def _aiter(chunks: list[Any]) -> AsyncGenerator[Any, None]:
    for chunk in chunks:
        yield chunk


async def _collect(
    provider: Provider,
    mode: Mode,
    model: Any,
    chunks: list[Any],
    is_async: bool,
    received: list[Any],
) -> None:
    parser: Any = mode_registry.get_handlers(provider, mode).response_parser
    if is_async:
        async for item in parser(_aiter(chunks), model, stream=True, is_async=True):
            received.append(item)
    else:
        received.extend(parser(iter(chunks), model, stream=True))


@sync_and_async
@stream_cases
@pytest.mark.asyncio
async def test_iterable_stream_raises_on_token_limit_after_complete_items(
    provider: Provider, mode: Mode, build: Callable[[bool], list[Any]], is_async: bool
) -> None:
    chunks = build(True)
    received: list[Any] = []

    with pytest.raises(IncompleteOutputException) as exc_info:
        await _collect(provider, mode, IterableModel(Item), chunks, is_async, received)

    assert received == [Item(name="Ada")]
    if mode is Mode.RESPONSES_TOOLS:
        assert exc_info.value.last_completion is chunks[-1].response
        assert "max_output_tokens" in str(exc_info.value)
    else:
        assert exc_info.value.last_completion is chunks[-1]


@sync_and_async
@stream_cases
@pytest.mark.asyncio
async def test_iterable_stream_normal_stop_ends_cleanly(
    provider: Provider, mode: Mode, build: Callable[[bool], list[Any]], is_async: bool
) -> None:
    received: list[Any] = []
    await _collect(
        provider, mode, IterableModel(Item), build(False), is_async, received
    )
    assert received == [Item(name="Ada")]


@sync_and_async
@stream_cases
@pytest.mark.asyncio
async def test_partial_stream_stays_best_effort_on_token_limit(
    provider: Provider, mode: Mode, build: Callable[[bool], list[Any]], is_async: bool
) -> None:
    received: list[Any] = []
    await _collect(provider, mode, Partial[Items], build(True), is_async, received)
    assert received


# --- non-streaming ---------------------------------------------------------


def _anthropic_message(beta: bool) -> Any:
    from anthropic.types import Message
    from anthropic.types.beta import BetaMessage

    cls: Any = BetaMessage if beta else Message
    return cls.model_validate(
        {
            "id": "msg",
            "type": "message",
            "role": "assistant",
            "model": "claude",
            "content": [
                {
                    "type": "tool_use",
                    "id": "t",
                    "name": "Answer",
                    "input": {"answer": 4},
                }
            ],
            "stop_reason": "max_tokens",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )


def _bedrock(response: dict[str, Any]) -> dict[str, Any]:
    return {**response, "stopReason": "max_tokens"}


def _openai_responses_incomplete() -> Any:
    from openai.types.responses import Response

    # No incomplete_details: the error message falls back to a generic reason.
    return Response.model_construct(status="incomplete", incomplete_details=None)


def _genai_max_tokens() -> Any:
    types = pytest.importorskip("google.genai.types")
    content = types.Content(role="model", parts=[types.Part(text='{"answer": 4')])
    candidate = types.Candidate(
        content=content, finish_reason=types.FinishReason.MAX_TOKENS
    )
    return types.GenerateContentResponse(candidates=[candidate])


NONSTREAM_CASES: list[tuple[str, Provider, Mode, Callable[[], Any]]] = [
    (
        "bedrock-tools",
        Provider.BEDROCK,
        Mode.TOOLS,
        lambda: _bedrock(_bedrock_tool_response({"answer": 4.0})),
    ),
    (
        "bedrock-md-json",
        Provider.BEDROCK,
        Mode.MD_JSON,
        lambda: _bedrock(_bedrock_text_response('```json\n{"answer": 4')),
    ),
    (
        "bedrock-json-schema",
        Provider.BEDROCK,
        Mode.JSON_SCHEMA,
        lambda: _bedrock(_bedrock_text_response('{"answer": 4')),
    ),
    (
        "writer-md-json",
        Provider.WRITER,
        Mode.MD_JSON,
        lambda: chat_completion(
            content='```json\n{"answer": 4}\n```', finish_reason="length"
        ),
    ),
    (
        "writer-json-schema",
        Provider.WRITER,
        Mode.JSON_SCHEMA,
        lambda: chat_completion(content='{"answer": 4}', finish_reason="length"),
    ),
    (
        "anthropic-message",
        Provider.ANTHROPIC,
        Mode.TOOLS,
        lambda: _anthropic_message(beta=False),
    ),
    (
        "anthropic-beta-message",
        Provider.ANTHROPIC,
        Mode.TOOLS,
        lambda: _anthropic_message(beta=True),
    ),
    (
        "openai-responses",
        Provider.OPENAI,
        Mode.RESPONSES_TOOLS,
        _openai_responses_incomplete,
    ),
    ("genai-json", Provider.GENAI, Mode.JSON, _genai_max_tokens),
]


@pytest.mark.parametrize(
    ("provider", "mode", "build"),
    [c[1:] for c in NONSTREAM_CASES],
    ids=[c[0] for c in NONSTREAM_CASES],
)
def test_non_streaming_token_limit_raises(
    provider: Provider, mode: Mode, build: Callable[[], Any]
) -> None:
    response = build()

    with pytest.raises(IncompleteOutputException) as exc_info:
        mode_registry.get_handlers(provider, mode).response_parser(response, Answer)

    assert exc_info.value.last_completion is response


def test_gemini_iterable_stream_raises_on_max_tokens_after_complete_item() -> None:
    """Gemini's streaming extractor must surface MAX_TOKENS for iterables."""
    from instructor.v2.providers.gemini.handlers import GeminiJSONHandler

    complete = SimpleNamespace(text=ITEMS_JSON, candidates=[])
    terminal = SimpleNamespace(
        text="",
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="MAX_TOKENS"))],
    )
    handler = GeminiJSONHandler()

    generator = handler.extract_streaming_json(
        [complete, terminal], require_complete=True
    )
    assert next(generator) == ITEMS_JSON
    with pytest.raises(IncompleteOutputException) as exc_info:
        next(generator)
    assert exc_info.value.last_completion is terminal
