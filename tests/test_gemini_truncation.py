"""Google responses stopped at the token limit must not validate using field defaults."""

import json
from enum import IntEnum
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from instructor.v2.core.errors import IncompleteOutputException
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
    item: str
    quantity: int = 1


class FinishReason(IntEnum):
    STOP = 1
    MAX_TOKENS = 2
    SAFETY = 3


def response(reason, payload):
    part = SimpleNamespace(
        function_call=SimpleNamespace(args=payload), text=json.dumps(payload)
    )
    candidate = SimpleNamespace(
        finish_reason=reason, content=SimpleNamespace(parts=[part])
    )
    return SimpleNamespace(candidates=[candidate], text=part.text)


@pytest.mark.parametrize(
    "handler_cls",
    [GeminiToolsHandler, GeminiJSONHandler, VertexAIToolsHandler, VertexAIJSONHandler],
)
def test_truncation_precedes_parsing_and_complete_results_still_validate(handler_cls):
    handler = handler_cls()
    incomplete = response(FinishReason.MAX_TOKENS, {"item": "notebook"})
    with pytest.raises(IncompleteOutputException) as exc:
        handler.parse_response(incomplete, Order, None, None, False, False)
    assert exc.value.last_completion is incomplete
    complete = response(FinishReason.STOP, {"item": "notebook", "quantity": 3})
    assert handler.parse_response(complete, Order, None, None, False, False) == Order(
        item="notebook", quantity=3
    )


@pytest.mark.parametrize("reason", [FinishReason.SAFETY, FinishReason.STOP, None])
def test_other_finish_reasons_do_not_indicate_truncation(reason):
    assert not is_truncated_at_max_tokens(response(reason, {}))


def test_empty_candidates_do_not_indicate_truncation():
    assert not is_truncated_at_max_tokens(None)
    assert not is_truncated_at_max_tokens(SimpleNamespace(candidates=[]))


@pytest.mark.parametrize(
    "module",
    [
        "google.generativeai.types.answer_types",
        "vertexai.generative_models",
        "google.genai.types",
    ],
)
def test_google_sdk_finish_reasons(module):
    sdk = pytest.importorskip(module)
    assert is_truncated_at_max_tokens(response(sdk.FinishReason.MAX_TOKENS, {}))
    assert not is_truncated_at_max_tokens(response(sdk.FinishReason.STOP, {}))


@pytest.mark.parametrize(
    ("reason", "truncated"),
    [("MAX_TOKENS", True), ("FinishReason.MAX_TOKENS", True), ("STOP", False)],
)
def test_string_finish_reasons(reason, truncated):
    assert is_truncated_at_max_tokens(response(reason, {})) is truncated
