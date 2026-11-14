"""Writer-specific handler behavior not covered by shared compatibility tests."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from instructor import Mode, Provider
from instructor.v2.core.errors import IncompleteOutputException
from instructor.v2.core.registry import mode_registry
from tests.coverage._openai import chat_completion, tool_call


class Answer(BaseModel):
    answer: float


class User(BaseModel):
    name: str
    age: int


def test_tools_request_uses_auto_tool_choice() -> None:
    _, kwargs = mode_registry.get_handlers(Provider.WRITER, Mode.TOOLS).request_handler(
        Answer,
        {"messages": [{"role": "user", "content": "What is 2+2?"}]},
    )

    assert kwargs["tool_choice"] == "auto"


def test_tools_parse_user_model() -> None:
    response = chat_completion(
        tool_calls=[tool_call("User", {"name": "Alice", "age": 30})]
    )

    result = mode_registry.get_handlers(Provider.WRITER, Mode.TOOLS).response_parser(
        response,
        User,
    )

    assert result == User(name="Alice", age=30)


def test_md_json_parses_nested_codeblock() -> None:
    response = chat_completion(content='```json\n{"name": "Bob", "age": 25}\n```')

    result = mode_registry.get_handlers(Provider.WRITER, Mode.MD_JSON).response_parser(
        response,
        User,
    )

    assert result == User(name="Bob", age=25)


@pytest.mark.parametrize(
    ("mode", "content"),
    [
        (Mode.MD_JSON, '```json\n{"name": "Bob", "age": 25}\n```'),
        (Mode.JSON_SCHEMA, '{"name": "Bob", "age": 25}'),
    ],
)
def test_json_modes_raise_incomplete_on_length(mode: Mode, content: str) -> None:
    response = chat_completion(content=content, finish_reason="length")

    with pytest.raises(IncompleteOutputException) as exc_info:
        mode_registry.get_handlers(Provider.WRITER, mode).response_parser(
            response, User
        )

    assert exc_info.value.last_completion is response
