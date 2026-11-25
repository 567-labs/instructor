"""Regression tests for blocked/empty GenAI structured-output responses."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from instructor import Mode, Provider
from instructor.v2.core.errors import ResponseParsingError
from instructor.v2.core.retry import retry_sync_v2
from instructor.v2.providers.genai.handlers import parse_genai_structured_outputs


class Answer(BaseModel):
    value: int


@pytest.mark.parametrize("text", [None, ""])
def test_parse_genai_structured_outputs_empty_text_raises_response_parsing_error(
    text: str | None,
) -> None:
    completion = SimpleNamespace(text=text)

    with pytest.raises(ResponseParsingError) as exc_info:
        parse_genai_structured_outputs(Answer, completion)

    error = exc_info.value
    assert "No text content" in str(error)
    assert error.mode == "GENAI_JSON"
    assert error.raw_response is completion


def test_parse_genai_structured_outputs_empty_candidates_response() -> None:
    types = pytest.importorskip("google.genai.types")
    completion = types.GenerateContentResponse(candidates=[])
    assert completion.text is None

    with pytest.raises(ResponseParsingError) as exc_info:
        parse_genai_structured_outputs(Answer, completion)

    assert exc_info.value.raw_response is completion


def test_parse_genai_structured_outputs_valid_json_control() -> None:
    completion = SimpleNamespace(text='{"value": 42}')

    assert parse_genai_structured_outputs(Answer, completion) == Answer(value=42)


def test_retry_reasks_after_empty_genai_structured_output() -> None:
    call_count = 0

    def fake_completion(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return SimpleNamespace(text=None)
        return SimpleNamespace(text='{"value": 7}')

    result = retry_sync_v2(
        func=fake_completion,
        response_model=Answer,
        provider=Provider.GENAI,
        mode=Mode.JSON,
        context=None,
        max_retries=1,
        args=(),
        kwargs={},
        strict=False,
        hooks=None,
    )

    assert result == Answer(value=7)
    assert call_count == 2
