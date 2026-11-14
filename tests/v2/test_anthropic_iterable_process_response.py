"""Non-streaming Iterable[...] results from Anthropic go through the Anthropic parser."""

from __future__ import annotations

from typing import Any

import pytest
from anthropic.types import Message
from pydantic import BaseModel

from instructor import Mode, Provider
from instructor.v2.core.response import process_response, process_response_async
from instructor.v2.dsl.iterable import IterableModel


class U(BaseModel):
    name: str


def _tool_message() -> Message:
    return Message.model_validate(
        {
            "id": "msg_local",
            "type": "message",
            "role": "assistant",
            "model": "claude-local",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_local",
                    "name": "IterableU",
                    "input": {"tasks": [{"name": "a"}]},
                }
            ],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )


def _assert_result(result: Any, response: Message) -> None:
    assert [item.model_dump() for item in result] == [{"name": "a"}]
    assert isinstance(result[0], U)
    assert result._raw_response is response


def test_process_response_anthropic_iterable() -> None:
    response = _tool_message()
    result = process_response(
        response,
        response_model=IterableModel(U),
        stream=False,
        mode=Mode.TOOLS,
        provider=Provider.ANTHROPIC,
    )
    _assert_result(result, response)


@pytest.mark.asyncio
async def test_process_response_async_anthropic_iterable() -> None:
    response = _tool_message()
    result = await process_response_async(
        response,  # type: ignore[arg-type]
        response_model=IterableModel(U),
        stream=False,
        mode=Mode.TOOLS,
        provider=Provider.ANTHROPIC,
    )
    _assert_result(result, response)
