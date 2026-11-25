"""Converse parsers must preserve all generated answer text."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from pydantic import BaseModel

from instructor import Mode, Provider
from instructor.v2.core.errors import ResponseParsingError
from instructor.v2.core.registry import mode_registry
import instructor.v2.providers.bedrock.handlers  # noqa: F401


class Booking(BaseModel):
    duration: int


@pytest.mark.parametrize("mode", [Mode.MD_JSON, Mode.JSON_SCHEMA])
@pytest.mark.parametrize("following_text", [False, True])
def test_bedrock_parses_cited_response_text(mode: Mode, following_text: bool) -> None:
    blocks: list[dict[str, Any]] = [
        {"reasoningContent": {"reasoningText": {"text": "internal", "signature": "s"}}},
        {
            "citationsContent": {
                "content": [{"text": '{"duration":'}, {"text": "45}"}],
                "citations": [
                    {
                        "title": "Booking record",
                        "sourceContent": [{"text": "45 minutes"}],
                    }
                ],
            }
        },
    ]
    if following_text:
        blocks.append({"text": '{"duration":99}'})
    response = {"output": {"message": {"role": "assistant", "content": blocks}}}
    original = deepcopy(response)
    handler = mode_registry.get_handlers(Provider.BEDROCK, mode)
    result = handler.response_parser(response, Booking)
    assert result.duration == 45
    assert response == original


@pytest.mark.parametrize("mode", [Mode.MD_JSON, Mode.JSON_SCHEMA])
def test_bedrock_single_text_block(mode: Mode) -> None:
    response = {"output": {"message": {"content": [{"text": '{"duration":45}'}]}}}
    handler = mode_registry.get_handlers(Provider.BEDROCK, mode)
    assert handler.response_parser(response, Booking).duration == 45


@pytest.mark.parametrize("mode", [Mode.MD_JSON, Mode.JSON_SCHEMA])
def test_bedrock_missing_text_is_retryable(mode: Mode) -> None:
    response = {"output": {"message": {"content": [{"reasoningContent": {}}]}}}
    handler = mode_registry.get_handlers(Provider.BEDROCK, mode)
    with pytest.raises(ResponseParsingError):
        handler.response_parser(response, Booking)


@pytest.mark.parametrize("mode", [Mode.MD_JSON, Mode.JSON_SCHEMA])
def test_bedrock_keeps_first_plain_text_answer(mode: Mode) -> None:
    response = {
        "output": {
            "message": {
                "content": [{"text": '{"duration":45}'}, {"text": '{"duration":99}'}]
            }
        }
    }
    handler = mode_registry.get_handlers(Provider.BEDROCK, mode)
    assert handler.response_parser(response, Booking).duration == 45
