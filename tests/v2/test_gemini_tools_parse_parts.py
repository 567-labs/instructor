"""Regression tests for Gemini TOOLS parsing with leading text/thought parts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from instructor.v2.core.errors import ResponseParsingError
from instructor.v2.providers.gemini.handlers import (
    GeminiToolsHandler,
    parse_gemini_tools,
)


class Answer(BaseModel):
    value: int


def _part(*, text: str | None = None, function_call: Any = None) -> SimpleNamespace:
    """Build a part; protobuf exposes an empty function_call on text-only parts."""
    return SimpleNamespace(text=text, function_call=function_call)


def _completion(*parts: Any) -> SimpleNamespace:
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=list(parts)))],
    )


def _call(*, name: str = "Answer", args: Any = None) -> SimpleNamespace:
    return SimpleNamespace(name=name, args=args)


def test_parse_gemini_tools_skips_leading_text_part() -> None:
    thought = _part(text="Let me think", function_call=SimpleNamespace())
    completion = _completion(thought, _part(function_call=_call(args={"value": 7})))

    parsed = parse_gemini_tools(Answer, completion)

    assert parsed == Answer(value=7)


def test_parse_gemini_tools_raises_when_no_named_function_call() -> None:
    completion = _completion(_part(text="only prose", function_call=SimpleNamespace()))

    with pytest.raises(ResponseParsingError) as no_call:
        parse_gemini_tools(Answer, completion)
    assert no_call.value.mode == "GEMINI_TOOLS"

    with pytest.raises(ResponseParsingError, match="No tool call found"):
        parse_gemini_tools(Answer, SimpleNamespace(candidates=[]))


def test_tools_handler_parses_call_after_leading_text_part() -> None:
    thought = _part(text="reasoning", function_call=SimpleNamespace())
    completion = _completion(thought, _part(function_call=_call(args={"value": 9})))

    parsed = GeminiToolsHandler().parse_response(completion, Answer)

    assert parsed.value == 9
    assert parsed._raw_response is completion


def test_parse_gemini_tools_legacy_single_part_without_name() -> None:
    """Legacy single-part calls omit `name`; the parts[0] fallback still parses."""
    completion = _completion(_part(function_call=SimpleNamespace(args={"value": 11})))

    parsed = parse_gemini_tools(Answer, completion)

    assert parsed == Answer(value=11)


def test_parse_gemini_tools_raises_when_part_has_no_function_call() -> None:
    """A part with no function_call leaves the call unresolved and raises."""
    completion = _completion(_part(text="only prose"))

    with pytest.raises(ResponseParsingError, match="No tool call found") as no_call:
        parse_gemini_tools(Answer, completion)
    assert no_call.value.mode == "GEMINI_TOOLS"
