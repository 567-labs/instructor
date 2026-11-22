"""Offline parsing tests for the Writer v2 JSON handlers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from instructor.v2.core.errors import ResponseParsingError
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.retry import retry_sync_v2
from instructor.v2.providers.writer.handlers import (
    WriterJSONSchemaHandler,
    WriterMDJSONHandler,
)


class Answer(BaseModel):
    answer: int


def _completion(*choices: dict[str, Any]) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-writer",
            "object": "chat.completion",
            "created": 1,
            "model": "palmyra-test",
            "choices": [{"index": i, **choice} for i, choice in enumerate(choices)],
        }
    )


def _answer(text: str) -> ChatCompletion:
    return _completion(
        {
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": text},
        }
    )


@pytest.mark.parametrize("mode", [Mode.MD_JSON, Mode.JSON_SCHEMA])
@pytest.mark.parametrize(
    "response",
    [
        pytest.param(_completion(), id="empty-choices"),
        pytest.param(SimpleNamespace(choices=None), id="choices-none"),
        pytest.param(SimpleNamespace(), id="no-choices-attribute"),
    ],
)
def test_json_modes_raise_response_parsing_error_without_choices(
    mode: Mode, response: Any
) -> None:
    handler = (
        WriterMDJSONHandler() if mode == Mode.MD_JSON else WriterJSONSchemaHandler()
    )

    with pytest.raises(ResponseParsingError, match="No choices in Writer") as exc_info:
        handler.parse_response(response, Answer)

    assert exc_info.value.raw_response is response
    # Stays a ValueError so existing `except ValueError` callers keep working.
    assert isinstance(exc_info.value, ValueError)


@pytest.mark.parametrize(
    ("mode", "good_text"),
    [
        (Mode.MD_JSON, '```json\n{"answer": 7}\n```'),
        (Mode.JSON_SCHEMA, '{"answer": 7}'),
    ],
)
def test_json_modes_reask_after_a_response_without_choices(
    mode: Mode, good_text: str
) -> None:
    responses = iter([_completion(), _answer(good_text)])
    seen_messages: list[int] = []

    def create(**kwargs: Any) -> ChatCompletion:
        seen_messages.append(len(kwargs["messages"]))
        return next(responses)

    result = retry_sync_v2(
        func=create,
        response_model=Answer,
        provider=Provider.WRITER,
        mode=mode,
        context=None,
        max_retries=1,
        args=(),
        kwargs={"messages": [{"role": "user", "content": "hi"}]},
        strict=False,
    )

    assert result == Answer(answer=7)
    # The failed attempt adds a correction message to the second request.
    assert seen_messages[0] == 1
    assert seen_messages[1] > seen_messages[0]


@pytest.mark.parametrize("handler_cls", [WriterMDJSONHandler, WriterJSONSchemaHandler])
def test_json_modes_still_report_truncated_output(handler_cls: type) -> None:
    from instructor.v2.core.errors import IncompleteOutputException

    truncated = _completion(
        {
            "finish_reason": "length",
            "message": {"role": "assistant", "content": '{"answer":'},
        }
    )

    with pytest.raises(IncompleteOutputException):
        handler_cls().parse_response(truncated, Answer)
