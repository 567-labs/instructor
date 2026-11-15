"""Contract: a provider token limit raises instead of returning truncated output.

Streams: Iterable streams flush every complete item, then raise
``IncompleteOutputException`` when the provider stopped for its token limit.
Partial streams stay best-effort. Non-streaming JSON/tool modes raise on the
token-limit stop reason. All chunks/responses are real SDK objects fed straight
into the registered response parser.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from enum import IntEnum
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import BaseModel

from instructor.v2.core.errors import IncompleteOutputException
from instructor.v2.core.mode import Mode
from instructor.v2.core.patch import patch
from instructor.v2.core.providers import Provider
from instructor.v2.core.registry import mode_registry
from instructor.v2.dsl.iterable import IterableModel
from instructor.v2.dsl.partial import Partial
from instructor.v2.providers.gemini.utils import is_truncated_at_max_tokens
from tests._streams import async_items
from tests.coverage._openai import chat_completion


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


STREAM_CASES: list[tuple[str, Provider, Mode, Callable[[bool], list[Any]]]] = [
    ("openai-tools", Provider.OPENAI, Mode.TOOLS, _openai_chat(Mode.TOOLS)),
    ("openai-json", Provider.OPENAI, Mode.JSON, _openai_chat(Mode.JSON)),
    ("openai-responses", Provider.OPENAI, Mode.RESPONSES_TOOLS, _openai_responses),
    ("anthropic-tools", Provider.ANTHROPIC, Mode.TOOLS, _anthropic),
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
        async for item in parser(
            async_items(chunks), model, stream=True, is_async=True
        ):
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
# Payloads are complete and valid: the stop reason alone decides truncation,
# and it is checked before parsing so defaults never stand in for cut fields.


class Answer(BaseModel):
    answer: float


ANSWER_JSON = '{"answer": 4}'


def _bedrock(content: dict[str, Any]) -> Callable[[bool], Any]:
    return lambda limit: {
        "output": {"message": {"content": [content]}},
        "stopReason": "max_tokens" if limit else "end_turn",
    }


def _writer(content: str) -> Callable[[bool], Any]:
    return lambda limit: chat_completion(
        content=content, finish_reason="length" if limit else "stop"
    )


def _anthropic_message(beta: bool) -> Callable[[bool], Any]:
    def build(limit: bool) -> Any:
        from anthropic.types import Message
        from anthropic.types.beta import BetaMessage

        cls: Any = BetaMessage if beta else Message
        tool = {"type": "tool_use", "id": "t", "name": "Answer", "input": {"answer": 4}}
        return cls.model_validate(
            {
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "model": "claude",
                "content": [tool],
                "stop_reason": "max_tokens" if limit else "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )

    return build


def _openai_response(limit: bool) -> Any:
    from openai.types.responses import Response

    call = {
        "type": "function_call",
        "id": "fc",
        "call_id": "c",
        "name": "Answer",
        "arguments": ANSWER_JSON,
        "status": "completed",
    }
    # No incomplete_details: the error message falls back to a generic reason.
    return Response.model_validate(
        {
            "id": "r",
            "created_at": 0,
            "object": "response",
            "model": "m",
            "output": [call],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "status": "incomplete" if limit else "completed",
            "incomplete_details": None,
        }
    )


def _genai(tools: bool) -> Callable[[bool], Any]:
    def build(limit: bool) -> Any:
        types = pytest.importorskip("google.genai.types")
        part = (
            types.Part.from_function_call(name="Answer", args={"answer": 4})
            if tools
            else types.Part(text=ANSWER_JSON)
        )
        finish = types.FinishReason.MAX_TOKENS if limit else types.FinishReason.STOP
        candidate = types.Candidate(
            content=types.Content(role="model", parts=[part]), finish_reason=finish
        )
        return types.GenerateContentResponse(candidates=[candidate])

    return build


class GoogleFinishReason(IntEnum):
    """Integer enum like the legacy Gemini/Vertex SDK finish reasons."""

    STOP = 1
    MAX_TOKENS = 2
    SAFETY = 3


def google_response(reason: Any, payload: dict[str, Any]) -> Any:
    """In-repo stand-in for the legacy Gemini/Vertex response shape."""
    part = SimpleNamespace(
        function_call=SimpleNamespace(args=payload), text=json.dumps(payload)
    )
    candidate = SimpleNamespace(
        finish_reason=reason, content=SimpleNamespace(parts=[part])
    )
    return SimpleNamespace(candidates=[candidate], text=part.text)


def _legacy_google(limit: bool) -> Any:
    reason = GoogleFinishReason.MAX_TOKENS if limit else GoogleFinishReason.STOP
    return google_response(reason, {"answer": 4})


NONSTREAM_CASES: list[tuple[str, Provider, Mode, Callable[[bool], Any]]] = [
    (
        "bedrock-tools",
        Provider.BEDROCK,
        Mode.TOOLS,
        _bedrock(
            {"toolUse": {"toolUseId": "t", "name": "Answer", "input": {"answer": 4}}}
        ),
    ),
    (
        "bedrock-md-json",
        Provider.BEDROCK,
        Mode.MD_JSON,
        _bedrock({"text": f"```json\n{ANSWER_JSON}\n```"}),
    ),
    (
        "bedrock-json-schema",
        Provider.BEDROCK,
        Mode.JSON_SCHEMA,
        _bedrock({"text": ANSWER_JSON}),
    ),
    (
        "writer-md-json",
        Provider.WRITER,
        Mode.MD_JSON,
        _writer(f"```json\n{ANSWER_JSON}\n```"),
    ),
    ("writer-json-schema", Provider.WRITER, Mode.JSON_SCHEMA, _writer(ANSWER_JSON)),
    ("anthropic-message", Provider.ANTHROPIC, Mode.TOOLS, _anthropic_message(False)),
    (
        "anthropic-beta-message",
        Provider.ANTHROPIC,
        Mode.TOOLS,
        _anthropic_message(True),
    ),
    ("openai-responses", Provider.OPENAI, Mode.RESPONSES_TOOLS, _openai_response),
    ("genai-tools", Provider.GENAI, Mode.TOOLS, _genai(tools=True)),
    ("genai-json", Provider.GENAI, Mode.JSON, _genai(tools=False)),
    ("gemini-tools", Provider.GEMINI, Mode.TOOLS, _legacy_google),
    ("gemini-md-json", Provider.GEMINI, Mode.MD_JSON, _legacy_google),
    ("vertexai-tools", Provider.VERTEXAI, Mode.TOOLS, _legacy_google),
    ("vertexai-md-json", Provider.VERTEXAI, Mode.MD_JSON, _legacy_google),
]


@pytest.mark.parametrize("limit", [True, False], ids=["max-tokens", "normal-stop"])
@pytest.mark.parametrize(
    ("provider", "mode", "build"),
    [c[1:] for c in NONSTREAM_CASES],
    ids=[c[0] for c in NONSTREAM_CASES],
)
def test_non_streaming_token_limit_raises(
    provider: Provider, mode: Mode, build: Callable[[bool], Any], limit: bool
) -> None:
    response = build(limit)
    parse = mode_registry.get_handlers(provider, mode).response_parser

    if not limit:
        assert parse(response, Answer) == Answer(answer=4)
        return
    with pytest.raises(IncompleteOutputException) as exc_info:
        parse(response, Answer)
    assert exc_info.value.last_completion is response


def _sdk_reason(module: str, name: str) -> Callable[[], Any]:
    def build() -> Any:
        return getattr(pytest.importorskip(module).FinishReason, name)

    return build


TRUNCATION_CASES: list[tuple[str, Callable[[], Any], bool]] = [
    ("int-enum-max-tokens", lambda: GoogleFinishReason.MAX_TOKENS, True),
    ("int-enum-stop", lambda: GoogleFinishReason.STOP, False),
    ("int-enum-safety", lambda: GoogleFinishReason.SAFETY, False),
    ("unset", lambda: None, False),
    ("string-max-tokens", lambda: "MAX_TOKENS", True),
    ("qualified-string-max-tokens", lambda: "FinishReason.MAX_TOKENS", True),
    ("string-stop", lambda: "STOP", False),
    *[
        (f"{module}-{name.lower()}", _sdk_reason(module, name), name == "MAX_TOKENS")
        for module in (
            "google.generativeai.types.answer_types",
            "vertexai.generative_models",
            "google.genai.types",
        )
        for name in ("MAX_TOKENS", "STOP", "SAFETY")
    ],
]


@pytest.mark.parametrize(
    ("reason", "truncated"),
    [c[1:] for c in TRUNCATION_CASES],
    ids=[c[0] for c in TRUNCATION_CASES],
)
def test_is_truncated_at_max_tokens(reason: Callable[[], Any], truncated: bool) -> None:
    assert is_truncated_at_max_tokens(google_response(reason(), {})) is truncated


def test_missing_candidates_are_not_truncated() -> None:
    assert not is_truncated_at_max_tokens(None)
    assert not is_truncated_at_max_tokens(SimpleNamespace(candidates=[]))


@pytest.mark.parametrize("mode", [Mode.TOOLS, Mode.JSON])
def test_genai_empty_candidates_are_not_reported_as_truncation(mode: Mode) -> None:
    types = pytest.importorskip("google.genai.types")
    parse = mode_registry.get_handlers(Provider.GENAI, mode).response_parser
    with pytest.raises(Exception) as exc_info:
        parse(types.GenerateContentResponse(candidates=[]), Answer)
    assert not isinstance(exc_info.value, (IncompleteOutputException, IndexError))


# --- retry: a token limit is final, never retried or wrapped ----------------


@sync_and_async
@pytest.mark.parametrize("source", ["api-error", "parsed-response", "stream-response"])
@pytest.mark.parametrize("max_retries", [0, 3])
@pytest.mark.asyncio
async def test_token_limit_is_raised_once_without_retry_wrapping(
    source: str, max_retries: int, is_async: bool
) -> None:
    truncated = chat_completion(content='{"answer": 4', finish_reason="length")
    calls = 0

    def respond() -> Any:
        nonlocal calls
        calls += 1
        if source == "api-error":
            raise IncompleteOutputException(last_completion=None)
        return truncated

    async def acreate(**_kwargs: Any) -> Any:
        return respond()

    def create(**_kwargs: Any) -> Any:
        return respond()

    patched: Any = patch(
        create=acreate if is_async else create,
        provider=Provider.OPENAI,
        mode=Mode.JSON,
    )
    kwargs: dict[str, Any] = {
        "response_model": Answer,
        "messages": [],
        "max_retries": max_retries,
    }
    if source == "stream-response":
        kwargs["stream"] = True
    with pytest.raises(IncompleteOutputException) as exc_info:
        result = patched(**kwargs)
        if is_async:
            await result

    assert type(exc_info.value) is IncompleteOutputException
    assert exc_info.value.last_completion is (
        None if source == "api-error" else truncated
    )
    assert calls == 1
