"""Request-local routing with real SDK payloads, without network calls."""

from __future__ import annotations

import asyncio
import importlib
from typing import Any

import pytest
from anthropic.types import (
    Message,
    RawContentBlockDeltaEvent,
    TextBlock,
    ToolUseBlock,
    Usage,
    TextDelta,
    InputJSONDelta,
)
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel, ValidationError, ValidationInfo, field_validator

from instructor.v2.dsl.partial import Partial
from instructor.v2.providers.anthropic.handlers import (
    AnthropicHandlerBase,
    AnthropicJSONHandler,
    AnthropicToolsHandler,
    AnthropicStructuredOutputsHandler,
)
from instructor.v2.providers.mistral.handlers import (
    MistralHandlerBase,
    MistralJSONSchemaHandler,
    MistralToolsHandler,
    MistralMDJSONHandler,
)
from instructor.v2.providers.openai.handlers import (
    OpenAIJSONHandler,
    OpenAIToolsHandler,
    OpenAIMDJSONHandler,
    OpenAIResponsesToolsHandler,
    OpenAIJSONSchemaHandler,
)
from instructor.v2.providers.xai.handlers import (
    XAIHandlerBase,
    XAIJSONSchemaHandler,
    XAIToolsHandler,
    XAIMDJSONHandler,
)
from instructor.v2.core.mode import Mode


class Person(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def check_context(cls, value: str, info: ValidationInfo) -> str:
        if info.context and info.context.get("reject"):
            raise ValueError("Rejected by current request context")
        return value


HANDLERS = [
    OpenAIJSONHandler,
    OpenAIJSONSchemaHandler,
    AnthropicJSONHandler,
    MistralJSONSchemaHandler,
    XAIJSONSchemaHandler,
    OpenAIToolsHandler,
    OpenAIMDJSONHandler,
    OpenAIResponsesToolsHandler,
    AnthropicToolsHandler,
    AnthropicStructuredOutputsHandler,
    MistralToolsHandler,
    MistralMDJSONHandler,
    XAIToolsHandler,
    XAIMDJSONHandler,
]


def payloads(handler: Any, name: str = "Ada") -> tuple[Any, Any]:
    text = Person(name=name).model_dump_json()
    completion: dict[str, Any] = dict(
        id="local",
        model="local",
        created=0,
        object="chat.completion",
        choices=[
            dict(
                index=0,
                finish_reason="stop",
                message=dict(role="assistant", content=text),
            )
        ],
    )
    chunk: dict[str, Any] = dict(
        id="local",
        model="local",
        created=0,
        object="chat.completion.chunk",
        choices=[dict(index=0, finish_reason=None, delta=dict(content=text))],
    )
    tool = dict(
        id="call_local",
        type="function",
        function=dict(name="PartialPerson", arguments=text),
    )
    if handler.mode == Mode.TOOLS:
        completion["choices"][0]["message"] = dict(role="assistant", tool_calls=[tool])
        chunk["choices"][0]["delta"] = dict(tool_calls=[dict(index=0, **tool)])
    if isinstance(handler, OpenAIResponsesToolsHandler):
        from openai.types.responses import (
            Response,
            ResponseFunctionToolCall,
            ResponseFunctionCallArgumentsDeltaEvent,
        )

        return (
            Response.model_construct(
                output=[
                    ResponseFunctionToolCall(
                        type="function_call",
                        call_id="call_local",
                        name="PartialPerson",
                        arguments=text,
                    )
                ]
            ),
            ResponseFunctionCallArgumentsDeltaEvent(
                type="response.function_call_arguments.delta",
                item_id="call_local",
                output_index=0,
                sequence_number=0,
                delta=text,
            ),
        )
    if isinstance(handler, AnthropicHandlerBase):
        return (
            Message(
                id="local",
                model="local",
                role="assistant",
                type="message",
                content=(
                    [
                        ToolUseBlock(
                            type="tool_use",
                            id="call_local",
                            name="PartialPerson",
                            input=dict(name=name),
                        )
                    ]
                    if handler.mode == Mode.TOOLS
                    else [TextBlock(type="text", text=text)]
                ),
                stop_reason="end_turn",
                usage=Usage(input_tokens=1, output_tokens=1),
            ),
            RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=0,
                delta=(
                    InputJSONDelta(type="input_json_delta", partial_json=text)
                    if handler.mode == Mode.TOOLS
                    else TextDelta(type="text_delta", text=text)
                ),
            ),
        )
    if isinstance(handler, MistralHandlerBase):
        pytest.importorskip("mistralai")
        try:
            from mistralai.client.models import ChatCompletionResponse, CompletionEvent
        except ImportError:
            # Mistral 1.x (Python 3.9) predates the client.models namespace.
            models = importlib.import_module("mistralai.models")
            ChatCompletionResponse = models.ChatCompletionResponse
            CompletionEvent = models.CompletionEvent

        completion["usage"] = dict(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        return ChatCompletionResponse.model_validate(
            completion
        ), CompletionEvent.model_validate(dict(data=chunk))
    if isinstance(handler, XAIHandlerBase):
        pytest.importorskip("xai_sdk")
        from xai_sdk.chat import Response
        from xai_sdk.proto import chat_pb2

        message = dict(content=text)
        delta = dict(content=text)
        if handler.mode == Mode.TOOLS:
            native_tool = dict(
                id="call_local", function=dict(name="PartialPerson", arguments=text)
            )
            message = dict(tool_calls=[native_tool])
            delta = dict(tool_calls=[native_tool])
        return (
            Response(
                chat_pb2.GetChatCompletionResponse(
                    choices=[dict(index=0, message=message)]
                ),
                0,
            ),
            chat_pb2.GetChatCompletionChunk(choices=[dict(index=0, delta=delta)]),
        )
    return ChatCompletion.model_validate(
        completion
    ), ChatCompletionChunk.model_validate(chunk)


@pytest.mark.parametrize("handler_type", HANDLERS)
def test_explicit_nonstream_does_not_consume_another_request_as_stream(
    handler_type: Any,
) -> None:
    handler = handler_type()
    model = Partial[Person]
    handler.mark_streaming_model(model, True)  # request A prepares, then awaits I/O
    completion, chunk = payloads(handler)
    result = handler.parse_response(completion, model, stream=False)  # B finishes first
    assert isinstance(result, BaseModel)
    assert result.name == "Ada"
    assert (
        list(handler.parse_response(iter([chunk]), model, stream=True))[-1].name
        == "Ada"
    )


@pytest.mark.parametrize("handler_type", HANDLERS)
def test_two_explicit_streams_and_legacy_sequential_call(handler_type: Any) -> None:
    handler = handler_type()
    model = Partial[Person]
    _, chunk = payloads(handler)
    handler.mark_streaming_model(model, True)
    handler.mark_streaming_model(model, True)
    for _ in range(2):
        assert (
            list(handler.parse_response(iter([chunk]), model, stream=True))[-1].name
            == "Ada"
        )
    handler.mark_streaming_model(model, True)
    assert list(handler.parse_response(iter([chunk]), model))[-1].name == "Ada"


@pytest.mark.asyncio
@pytest.mark.parametrize("handler_type", HANDLERS)
async def test_async_overlap_cancellation_and_early_close(handler_type: Any) -> None:
    handler = handler_type()
    model = Partial[Person]
    entered = asyncio.Event()
    release = asyncio.Event()
    closed = asyncio.Event()
    completion, chunk = payloads(handler)

    async def source():
        try:
            entered.set()
            await release.wait()
            yield chunk
        finally:
            closed.set()

    raw = source()
    handler.mark_streaming_model(model, True)
    parsed = handler.parse_response(raw, model, stream=True, is_async=True)
    pending = asyncio.create_task(parsed.__anext__())
    await asyncio.wait_for(entered.wait(), 2)
    try:
        handler.mark_streaming_model(model, True)  # another request fails/cancels
        assert handler.parse_response(completion, model, stream=False).name == "Ada"
    finally:
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    assert closed.is_set()  # cancellation reaches the source currently being awaited
    await parsed.aclose()
    await raw.aclose()

    release.set()
    raw = source()
    parsed = handler.parse_response(raw, model, stream=True, is_async=True)
    assert isinstance(await parsed.__anext__(), BaseModel)
    # Caller owns source lifetime on early termination; no new cleanup contract.
    await parsed.aclose()
    await raw.aclose()
    assert handler.parse_response(completion, model, stream=False).name == "Ada"


@pytest.mark.asyncio
@pytest.mark.parametrize("handler_type", HANDLERS)
async def test_async_streams_overlap_and_keep_payloads_separate(
    handler_type: Any,
) -> None:
    handler = handler_type()
    model = Partial[Person]
    gates = [asyncio.Event(), asyncio.Event()]

    async def source(index: int):
        await gates[index].wait()
        yield payloads(handler, ["Ada", "Grace"][index])[1]

    async def collect(index: int):
        raw = source(index)
        parsed = handler.parse_response(raw, model, stream=True, is_async=True)
        try:
            return [item async for item in parsed]
        finally:
            await parsed.aclose()
            await raw.aclose()

    handler.mark_streaming_model(model, True)
    handler.mark_streaming_model(model, True)
    first = asyncio.create_task(collect(0))
    second = asyncio.create_task(collect(1))
    try:
        gates[1].set()
        assert (await asyncio.wait_for(second, 2))[-1].name == "Grace"
        gates[0].set()
        assert (await asyncio.wait_for(first, 2))[-1].name == "Ada"
    finally:
        for task in [first, second]:
            if not task.done():
                task.cancel()
        await asyncio.gather(first, second, return_exceptions=True)


@pytest.mark.parametrize("handler_type", HANDLERS)
def test_legacy_prepare_parse_still_infers_stream(handler_type: Any) -> None:
    handler = handler_type()
    _, chunk = payloads(handler)
    model, _ = handler.prepare_request(
        Partial[Person], {"stream": True, "messages": []}
    )
    assert list(handler.parse_response(iter([chunk]), model))[-1].name == "Ada"
    assert not handler._consume_streaming_flag(model)


@pytest.mark.asyncio
@pytest.mark.parametrize("handler_type", HANDLERS)
async def test_validation_timing_and_retry_remain_provider_owned(
    handler_type: Any,
) -> None:
    handler = handler_type()
    model = Partial[Person]
    _, chunk = payloads(handler)
    # Partial sync parsing materializes a list and raises during parse_response.
    with pytest.raises(ValidationError):
        handler.parse_response(
            iter([chunk]), model, stream=True, validation_context={"reject": True}
        )

    async def source():
        yield chunk

    raw = source()
    # Async parsing returns a generator; validation remains deferred to iteration.
    parsed = handler.parse_response(
        raw, model, stream=True, validation_context={"reject": True}
    )
    try:
        with pytest.raises(ValidationError):
            [item async for item in parsed]
    finally:
        await parsed.aclose()
        await raw.aclose()
    assert (
        list(handler.parse_response(iter([chunk]), model, stream=True))[-1].name
        == "Ada"
    )


@pytest.mark.asyncio
async def test_same_model_across_providers_and_modes() -> None:
    model = Partial[Person]

    async def run(handler_type: Any, index: int):
        handler = handler_type()
        name = f"Person {index}"
        completion, chunk = payloads(handler, name)
        handler.mark_streaming_model(model, True)

        async def source():
            await asyncio.sleep(0)
            yield chunk

        raw = source()
        parsed = handler.parse_response(raw, model, stream=True)
        try:
            result = [item async for item in parsed]
            assert result[-1].name == name
            assert handler.parse_response(completion, model, stream=False).name == name
        finally:
            await parsed.aclose()
            await raw.aclose()

    await asyncio.gather(
        *(run(handler, index) for index, handler in enumerate(HANDLERS))
    )
