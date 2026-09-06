from typing import Optional

import pytest


pytest.importorskip("google.genai")


from google.genai import types
from pydantic import BaseModel

from instructor.v2.core.errors import IncompleteOutputException
from instructor.v2.core.mode import Mode
from instructor.v2.providers.genai.handlers import (
    GenAIStructuredOutputsHandler,
    GenAIToolsHandler,
)


class Order(BaseModel):
    drug: str
    dose_mg: int
    contraindications: Optional[str] = None


def _response(
    finish_reason: types.FinishReason, *, tools: bool
) -> types.GenerateContentResponse:
    """A response whose payload parses cleanly but may have been truncated."""
    if tools:
        part = types.Part.from_function_call(
            name="Order", args={"drug": "warfarin", "dose_mg": 10}
        )
    else:
        part = types.Part(text='{"drug": "warfarin", "dose_mg": 10}')
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[part]),
                finish_reason=finish_reason,
            )
        ]
    )


@pytest.mark.parametrize(
    "handler_cls, mode, tools",
    [
        (GenAIToolsHandler, Mode.TOOLS, True),
        (GenAIStructuredOutputsHandler, Mode.JSON, False),
    ],
)
def test_truncated_response_raises(handler_cls, mode, tools):
    """A response cut off at max_tokens is unevaluable and must not be parsed.

    The remaining fields were never emitted, so parsing yields schema defaults
    indistinguishable from values the model actually chose.
    """
    handler = handler_cls(mode=mode)
    with pytest.raises(IncompleteOutputException):
        handler.parse_response(
            response_model=Order,
            response=_response(types.FinishReason.MAX_TOKENS, tools=tools),
        )


@pytest.mark.parametrize(
    "handler_cls, mode, tools",
    [
        (GenAIToolsHandler, Mode.TOOLS, True),
        (GenAIStructuredOutputsHandler, Mode.JSON, False),
    ],
)
def test_complete_response_still_parses(handler_cls, mode, tools):
    """A response that finished normally is unaffected."""
    handler = handler_cls(mode=mode)
    result = handler.parse_response(
        response_model=Order,
        response=_response(types.FinishReason.STOP, tools=tools),
    )
    assert result.drug == "warfarin"
    assert result.dose_mg == 10


@pytest.mark.parametrize(
    "handler_cls, mode, tools",
    [
        (GenAIToolsHandler, Mode.TOOLS, True),
        (GenAIStructuredOutputsHandler, Mode.JSON, False),
    ],
)
@pytest.mark.parametrize(
    "finish_reason",
    [types.FinishReason.SAFETY, None],
    ids=["safety", "unset"],
)
def test_non_truncation_finish_reasons_are_not_incomplete(
    handler_cls, mode, tools, finish_reason
):
    """Only MAX_TOKENS means truncation.

    A SAFETY block and an unset finish_reason are different conditions and must
    not be reported as incomplete output, or callers lose the ability to tell
    them apart.
    """
    handler = handler_cls(mode=mode)
    result = handler.parse_response(
        response_model=Order,
        response=_response(finish_reason, tools=tools),
    )
    assert result.drug == "warfarin"


@pytest.mark.parametrize(
    "handler_cls, mode",
    [
        (GenAIToolsHandler, Mode.TOOLS),
        (GenAIStructuredOutputsHandler, Mode.JSON),
    ],
)
def test_empty_candidates_does_not_raise_incomplete_output(handler_cls, mode):
    """An empty candidate list is not a truncated response.

    The guard must tolerate it rather than indexing into it; whatever the
    downstream parser already did with this input is unchanged.
    """
    handler = handler_cls(mode=mode)
    empty = types.GenerateContentResponse(candidates=[])
    with pytest.raises(Exception) as excinfo:
        handler.parse_response(response_model=Order, response=empty)
    assert not isinstance(excinfo.value, IncompleteOutputException), (
        f"empty candidates reported as truncation: {excinfo.value!r}"
    )
    assert not isinstance(excinfo.value, IndexError), (
        f"guard indexed into an empty candidate list: {excinfo.value!r}"
    )
