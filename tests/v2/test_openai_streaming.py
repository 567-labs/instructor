from __future__ import annotations

from collections.abc import Iterable
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
        handler.parse_response(
            stream_factory('{"name": "Jane"}'), response_model, stream=True
        )
    assert handler._consume_streaming_flag(response_model) is False

    result = handler.parse_response(
        stream_factory('{"name": "Jane Doe"}'), response_model, stream=True
    )
    assert isinstance(result, list)
    assert result[-1].name == "Jane Doe"
