"""Shared handler finalization preserves result shapes and response identity."""

from typing import Any, Callable, cast

import pytest
from pydantic import BaseModel

from instructor.v2.core.handler import ModeHandler
from instructor.v2.dsl.iterable import IterableModel
from instructor.v2.dsl.parallel import ParallelBase
from instructor.v2.dsl.simple_type import ModelAdapter
from instructor.v2.providers.anthropic.handlers import AnthropicToolsHandler
from instructor.v2.providers.mistral.handlers import MistralToolsHandler
from instructor.v2.providers.openai.handlers import OpenAIToolsHandler
from instructor.v2.providers.xai.handlers import XAIToolsHandler


class Item(BaseModel):
    value: int


@pytest.fixture(
    params=[
        OpenAIToolsHandler,
        AnthropicToolsHandler,
        MistralToolsHandler,
        XAIToolsHandler,
    ],
    ids=["openai", "anthropic", "mistral", "xai"],
)
def handler(request: pytest.FixtureRequest) -> ModeHandler:
    return request.param()


def test_model_keeps_original_raw_response(handler: ModeHandler) -> None:
    raw = {"provider_metadata": {"opaque": "retained"}}
    parsed = Item(value=1)

    result = handler._finalize_parsed_result(Item, raw, parsed)

    assert result is parsed
    assert result._raw_response is raw


def test_iterable_unwraps_tasks_without_changing_items(handler: ModeHandler) -> None:
    model = IterableModel(Item)
    item = Item(value=1)
    parsed = model(tasks=[item])

    result = handler._finalize_parsed_result(model, {}, parsed)

    assert isinstance(result, list)
    assert result == [item]
    assert result[0] is item


@pytest.mark.parametrize("value", [0, False, ""])
def test_adapter_keeps_falsey_contents(handler: ModeHandler, value: Any) -> None:
    adapter_factory = cast(
        Callable[[type[Any]], type[BaseModel]], ModelAdapter.__class_getitem__
    )
    model = adapter_factory(type(value))
    parsed = model(content=value)

    assert handler._finalize_parsed_result(model, {}, parsed) == value


def test_parallel_keeps_iterator_identity_and_laziness(handler: ModeHandler) -> None:
    visited = []

    def results():
        visited.append(True)
        yield Item(value=1)

    parsed = results()
    result = handler._finalize_parsed_result(ParallelBase(Item), {}, parsed)

    assert result is parsed
    assert not visited
    assert list(result) == [Item(value=1)]
    assert visited == [True]


def test_non_model_result_is_unchanged(handler: ModeHandler) -> None:
    parsed = {"unchanged": True}
    assert handler._finalize_parsed_result(Item, {}, parsed) is parsed
