from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any, Callable

import pytest
from openai.types.responses import (
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemAddedEvent,
    ResponseOutputMessage,
    ResponseTextDeltaEvent,
)
from pydantic import BaseModel, ValidationError, field_validator

from instructor.v2.dsl.partial import Partial
from instructor.v2.providers.openai.handlers import (
    OpenAIHandlerBase,
    OpenAIJSONHandler,
    OpenAIJSONSchemaHandler,
    OpenAIMDJSONHandler,
    OpenAIToolsHandler,
    OpenAIResponsesToolsHandler,
)


class User(BaseModel):
    name: str


def _responses_events(
    tool_name: str,
    payload: str,
    *,
    text_answer: bool,
    commentary: bool,
    final_phase: str | None,
) -> list[Any]:
    events: list[Any] = (
        [
            ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                output_index=0,
                sequence_number=0,
                item=ResponseOutputMessage(
                    id="msg_commentary",
                    type="message",
                    role="assistant",
                    status="in_progress",
                    content=[],
                    phase="commentary",
                ),
            ),
            ResponseTextDeltaEvent(
                type="response.output_text.delta",
                item_id="msg_commentary",
                output_index=0,
                content_index=0,
                sequence_number=1,
                delta='{"name":"Commentary","tasks":[{"name":"Commentary"}]}',
                logprobs=[],
            ),
        ]
        if commentary
        else []
    )
    item = (
        ResponseOutputMessage(
            id="result",
            type="message",
            role="assistant",
            status="in_progress",
            content=[],
            phase=final_phase,
        )
        if text_answer
        else ResponseFunctionToolCall(
            id="result",
            type="function_call",
            call_id="call_user",
            name=tool_name,
            arguments="",
        )
    )
    events.append(
        ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            output_index=1,
            sequence_number=2,
            item=item,
        )
    )
    for index, delta in enumerate([payload[:8], payload[8:]]):
        values = dict(
            item_id="result", output_index=1, sequence_number=index + 3, delta=delta
        )
        events.append(
            ResponseTextDeltaEvent(
                type="response.output_text.delta",
                content_index=0,
                logprobs=[],
                **values,
            )
            if text_answer
            else ResponseFunctionCallArgumentsDeltaEvent(
                type="response.function_call_arguments.delta", **values
            )
        )
    return events


@pytest.mark.parametrize("iterable", [False, True], ids=["partial", "iterable"])
@pytest.mark.parametrize("text_answer", [False, True], ids=["tool", "text"])
@pytest.mark.parametrize(
    ("commentary", "final_phase"),
    [(True, "final_answer"), (True, None), (False, None)],
)
def test_responses_stream_ignores_commentary(
    iterable: bool, text_answer: bool, commentary: bool, final_phase: str | None
) -> None:
    handler = OpenAIResponsesToolsHandler()
    model, _ = handler.prepare_request(
        Iterable[User] if iterable else Partial[User], {"stream": True}
    )
    payload = '{"tasks":[{"name":"Ada"}]}' if iterable else '{"name":"Ada"}'
    events = _responses_events(
        model.__name__,
        payload,
        text_answer=text_answer,
        commentary=commentary,
        final_phase=final_phase,
    )

    result = list(handler.parse_response(iter(events), model, stream=True))

    assert result[-1].name == "Ada"
    if iterable:
        assert result == [User(name="Ada")]


@pytest.mark.asyncio
@pytest.mark.parametrize("iterable", [False, True], ids=["partial", "iterable"])
@pytest.mark.parametrize("text_answer", [False, True], ids=["tool", "text"])
@pytest.mark.parametrize(
    ("commentary", "final_phase"),
    [(True, "final_answer"), (True, None), (False, None)],
)
async def test_async_responses_stream_ignores_commentary(
    iterable: bool, text_answer: bool, commentary: bool, final_phase: str | None
) -> None:
    handler = OpenAIResponsesToolsHandler()
    model, _ = handler.prepare_request(
        Iterable[User] if iterable else Partial[User], {"stream": True}
    )
    payload = '{"tasks":[{"name":"Ada"}]}' if iterable else '{"name":"Ada"}'
    events = _responses_events(
        model.__name__,
        payload,
        text_answer=text_answer,
        commentary=commentary,
        final_phase=final_phase,
    )

    async def source():
        for event in events:
            yield event

    result = [
        item async for item in handler.parse_response(source(), model, stream=True)
    ]

    assert result[-1].name == "Ada"
    if iterable:
        assert result == [User(name="Ada")]


def test_openai_tools_streaming_iterable_not_parallel():
    handler = OpenAIToolsHandler()
    response_model, kwargs = handler.prepare_request(
        Iterable[User],
        {"stream": True},
    )

    assert kwargs["tool_choice"] != "auto"
    assert handler._consume_streaming_flag(response_model)


class _Foo(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def _must_have_space(cls, value: str) -> str:
        if " " not in value:
            raise ValueError("must contain a space")
        return value


def _tool_stream(payload: str) -> Any:
    delta = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                index=0,
                id="call-0",
                function=SimpleNamespace(name="_Foo", arguments=payload),
            )
        ],
    )
    chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(delta=delta, finish_reason="stop"),
        ]
    )
    return iter([chunk])


def _text_stream(payload: str) -> Any:
    delta = SimpleNamespace(content=payload, tool_calls=None)
    return iter([SimpleNamespace(choices=[SimpleNamespace(delta=delta)])])


def _markdown_stream(payload: str) -> Any:
    return _text_stream(f"```json\n{payload}\n```")


@pytest.mark.parametrize(
    ("handler", "stream_factory"),
    [
        pytest.param(OpenAIToolsHandler(), _tool_stream, id="tools"),
        pytest.param(OpenAIJSONSchemaHandler(), _text_stream, id="json-schema"),
        pytest.param(OpenAIJSONHandler(), _text_stream, id="json"),
        pytest.param(OpenAIMDJSONHandler(), _markdown_stream, id="md-json"),
    ],
)
def test_streaming_retry_recovers_after_flag_consumed(
    handler: OpenAIHandlerBase,
    stream_factory: Callable[[str], Any],
) -> None:
    response_model = Partial[_Foo]
    handler.mark_streaming_model(response_model, True)

    with pytest.raises(ValidationError):
        handler.parse_response(
            stream_factory('{"name": "Jane"}'), response_model, stream=True
        )
    assert handler._consume_streaming_flag(response_model) is False

    result = handler.parse_response(
        stream_factory('{"name": "Jane Doe"}'), response_model, stream=True
    )
    assert isinstance(result, list)
    assert result[-1].name == "Jane Doe"
