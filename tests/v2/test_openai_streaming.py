from __future__ import annotations

from collections.abc import Generator, Iterable
from types import SimpleNamespace
from typing import Any, Callable

import pytest
from pydantic import BaseModel, ValidationError, field_validator

from instructor.v2.dsl.partial import Partial
from instructor.v2.providers.openai.handlers import (
    OpenAIHandlerBase,
    OpenAIJSONHandler,
    OpenAIJSONSchemaHandler,
    OpenAIMDJSONHandler,
    OpenAIToolsHandler,
)


class User(BaseModel):
    name: str


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
        list(
            handler.parse_response(
                stream_factory('{"name": "Jane"}'), response_model, stream=True
            )
        )
    assert handler._consume_streaming_flag(response_model) is False

    result = handler.parse_response(
        stream_factory('{"name": "Jane Doe"}'), response_model, stream=True
    )
    assert isinstance(result, Generator)
    final = list(result)[-1]
    assert isinstance(final, _Foo)
    assert final.name == "Jane Doe"


def test_sync_partial_yields_before_reading_next_chunk_and_closes() -> None:
    closed = False

    def source():
        nonlocal closed
        try:
            yield from _text_stream('{"name":"Jane')
            pytest.fail("The stream read the next chunk before the first yield")
        finally:
            closed = True

    result = OpenAIJSONSchemaHandler().parse_response(
        source(), Partial[User], stream=True
    )
    assert next(result).name == "Jane"
    result.close()
    assert closed


def _responses_message_events() -> list[Any]:
    from openai.types.responses import (
        ResponseOutputItemAddedEvent,
        ResponseOutputMessage,
        ResponseTextDeltaEvent,
    )

    def added(index: int, phase: Any) -> Any:
        return ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            output_index=index,
            sequence_number=index * 2,
            item=ResponseOutputMessage(
                id=f"msg-{index}",
                type="message",
                role="assistant",
                status="in_progress",
                content=[],
                phase=phase,
            ),
        )

    def text(index: int, delta: str) -> Any:
        return ResponseTextDeltaEvent(
            type="response.output_text.delta",
            item_id=f"msg-{index}",
            output_index=index,
            content_index=0,
            sequence_number=index * 2 + 1,
            delta=delta,
            logprobs=[],
        )

    return [
        added(0, "commentary"),
        text(0, "Checking the input first."),
        added(1, "final_answer"),
        text(1, '{"name": "Ada"}'),
    ]


def test_responses_stream_skips_commentary_message() -> None:
    from instructor.v2.providers.openai.handlers import OpenAIResponsesToolsHandler

    handler = OpenAIResponsesToolsHandler()
    results = list(
        handler.parse_response(
            iter(_responses_message_events()), Partial[User], stream=True
        )
    )

    assert results[-1].name == "Ada"
