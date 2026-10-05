"""Truncation handling for the Gemini (google.generativeai) and VertexAI handlers.

The GenAI handler has refused a response cut off at max_tokens since #2232. The Gemini and
VertexAI handlers did not, so `from_gemini` / `from_vertexai` returned schema defaults for
every field past the cut, indistinguishable from values the model actually chose.
"""

import json
from types import SimpleNamespace
from typing import Any, Optional

import pytest


pytest.importorskip("google.generativeai")


from google.generativeai import protos
from google.generativeai.types import answer_types
from pydantic import BaseModel

from instructor.v2.core.errors import IncompleteOutputException
from instructor.v2.core.mode import Mode
from instructor.v2.providers.gemini.handlers import (
    GeminiJSONHandler,
    GeminiToolsHandler,
)
from instructor.v2.providers.gemini.utils import is_truncated_at_max_tokens
from instructor.v2.providers.vertexai.handlers import (
    VertexAIJSONHandler,
    VertexAIToolsHandler,
)


class Order(BaseModel):
    """`dose_mg` and `contraindications` have defaults, so truncation is invisible."""

    drug: str
    dose_mg: int = 1
    contraindications: Optional[str] = None


def _candidate(finish_reason: Any, *, tools: bool, args: dict) -> Any:
    """A real protobuf Candidate, so `finish_reason` is the SDK's own enum."""
    if tools:
        part = protos.Part(function_call=protos.FunctionCall(name="Order", args=args))
    else:
        part = protos.Part(text=json.dumps(args))
    return protos.Candidate(
        content=protos.Content(role="model", parts=[part]),
        finish_reason=finish_reason,
    )


def _response(finish_reason: Any, *, tools: bool, args: dict | None = None) -> Any:
    """A response that parses cleanly, but may have been truncated mid-object.

    Only `drug` reaches the wire by default, which is what a max_tokens cut-off looks
    like: the remaining fields are absent from the tool arguments, so Pydantic fills
    them in with the schema defaults.

    The candidates are real protobuf objects. The wrapper only supplies `.text`, which
    the MD_JSON parse path reads and which this installed `google.generativeai` cannot
    derive from a bare protobuf message; nothing under test depends on it.
    """
    return SimpleNamespace(
        candidates=[
            _candidate(finish_reason, tools=tools, args=args or {"drug": "warfarin"})
        ],
        text=json.dumps(args or {"drug": "warfarin"}),
    )


def _full_response(*, tools: bool) -> Any:
    """A complete response, as a control: every field the schema declares is present."""
    return _response(
        answer_types.FinishReason.STOP,
        tools=tools,
        args={"drug": "warfarin", "dose_mg": 10, "contraindications": "pregnancy"},
    )


HANDLERS = [
    (GeminiToolsHandler, Mode.TOOLS, True),
    (GeminiJSONHandler, Mode.MD_JSON, False),
    (VertexAIToolsHandler, Mode.TOOLS, True),
    (VertexAIJSONHandler, Mode.MD_JSON, False),
]


@pytest.mark.parametrize("handler_cls, mode, tools", HANDLERS)
def test_truncated_response_raises(handler_cls, mode, tools):  # noqa: ARG001
    """A response cut off at max_tokens is unevaluable and must not be parsed.

    Without this the caller gets `dose_mg=1` and `contraindications=None` — the schema
    defaults — with nothing marking them as values the model never chose.
    """
    handler = handler_cls()
    response = _response(answer_types.FinishReason.MAX_TOKENS, tools=tools)

    with pytest.raises(IncompleteOutputException):
        handler.parse_response(response, Order, None, None, False, False)


@pytest.mark.parametrize("handler_cls, mode, tools", HANDLERS)  # noqa: ARG001
def test_complete_response_still_parses(handler_cls, mode, tools):  # noqa: ARG001
    """The control: a response the model finished must parse as before."""
    handler = handler_cls()

    parsed = handler.parse_response(
        _full_response(tools=tools), Order, None, None, False, False
    )

    assert parsed == Order(drug="warfarin", dose_mg=10, contraindications="pregnancy")


def test_is_truncated_compares_the_member_name_not_str():
    """`google.generativeai`'s ProtoEnum renders as its integer value through `str()`.

    A `str()` comparison would look for "MAX_TOKENS" in "2" and silently never match, so
    this pins that the member name is what gets used.
    """
    legacy = answer_types.FinishReason.MAX_TOKENS
    assert legacy.name == "MAX_TOKENS"
    assert str(legacy) == "2"

    assert is_truncated_at_max_tokens(_response(legacy, tools=True))


@pytest.mark.parametrize(
    "finish_reason",
    [
        answer_types.FinishReason.STOP,
        answer_types.FinishReason.SAFETY,
        answer_types.FinishReason.RECITATION,
        answer_types.FinishReason.FINISH_REASON_UNSPECIFIED,
    ],
)
def test_only_max_tokens_counts_as_truncated(finish_reason):
    """Other stop reasons are the model's own decisions and must not be rejected."""
    assert not is_truncated_at_max_tokens(_response(finish_reason, tools=True))


def test_missing_or_empty_candidates_is_not_truncated():
    """A response with no candidates cannot be a truncation, and must not raise here."""

    class _Empty:
        candidates: list = []

    class _NoReason:
        class _Cand:
            finish_reason = None

        candidates = [_Cand()]

    assert not is_truncated_at_max_tokens(None)
    assert not is_truncated_at_max_tokens(object())
    assert not is_truncated_at_max_tokens(_Empty())
    assert not is_truncated_at_max_tokens(_NoReason())
