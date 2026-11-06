"""Isolated tests for Anthropic JSON parsing helpers."""

from anthropic.types import Message, TextBlock, Usage
import pytest
from pydantic import BaseModel, ValidationError
from typing import cast

from instructor.mode import Mode
from instructor.processing.response import process_response
from instructor.utils.providers import Provider


CONTROL_CHAR_JSON = """{
"data": "Claude likes
control
characters"
}"""


class _AnthropicTestModel(BaseModel):
    data: str


def _build_message(data_content: str) -> Message:
    return Message(
        id="test_id",
        content=[TextBlock(type="text", text=data_content)],
        model="claude-3-5-haiku-20241022",
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=10, output_tokens=10),
    )


def test_parse_anthropic_json_strict_control_characters() -> None:
    message = _build_message(CONTROL_CHAR_JSON)

    with pytest.raises(ValidationError):
        process_response(
            response=message,
            response_model=_AnthropicTestModel,
            stream=False,
            strict=True,
            mode=Mode.JSON,
            provider=Provider.ANTHROPIC,
        )


def test_parse_anthropic_json_non_strict_preserves_control_characters() -> None:
    message = _build_message(CONTROL_CHAR_JSON)

    model = cast(
        _AnthropicTestModel,
        process_response(
            response=message,
            response_model=_AnthropicTestModel,
            stream=False,
            strict=False,
            mode=Mode.JSON,
            provider=Provider.ANTHROPIC,
        ),
    )

    assert model.data == "Claude likes\ncontrol\ncharacters"
