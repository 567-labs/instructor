"""Native Mistral tool arguments may span arbitrary streaming events."""

from __future__ import annotations

import importlib
import json
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest
from pydantic import BaseModel

from instructor.v2.dsl.iterable import IterableModel
from instructor.v2.providers.mistral.handlers import MistralToolsHandler

pytest.importorskip("mistralai")
try:
    models = importlib.import_module("mistralai.client.models")
except ImportError:
    models = importlib.import_module("mistralai.models")


class Person(BaseModel):
    name: str


TEXT = json.dumps({"tasks": [{"name": 'Ada \\"{}[]'}, {"name": "李"}]})
EXPECTED = [Person(**item) for item in json.loads(TEXT)["tasks"]]


def event(arguments: str | dict[str, Any]) -> Any:
    return models.CompletionEvent.model_validate(
        {
            "data": {
                "id": "local-completion",
                "model": "mistral-small-latest",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": None,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "IterablePerson",
                                        "arguments": arguments,
                                    },
                                }
                            ]
                        },
                    }
                ],
            }
        }
    )


@pytest.mark.parametrize("split", range(1, len(TEXT)))
def test_iterable_tool_arguments_split_at_every_boundary(split: int) -> None:
    result = MistralToolsHandler()._parse_streaming_response(
        IterableModel(Person),
        iter([event(TEXT[:split]), event(TEXT[split:])]),
        validation_context=None,
        strict=None,
    )
    assert list(result) == EXPECTED


@pytest.mark.asyncio
@pytest.mark.parametrize("split", range(1, len(TEXT)))
async def test_async_iterable_tool_arguments_split_at_every_boundary(
    split: int,
) -> None:
    async def stream() -> AsyncIterator[Any]:
        yield event(TEXT[:split])
        yield event(TEXT[split:])

    result = MistralToolsHandler()._parse_streaming_response(
        IterableModel(Person),
        stream(),
        validation_context=None,
        strict=None,
    )
    assert [item async for item in result] == EXPECTED


def test_iterable_tool_argument_complete_event_control() -> None:
    result = MistralToolsHandler()._parse_streaming_response(
        IterableModel(Person),
        iter([event(TEXT)]),
        validation_context=None,
        strict=None,
    )
    assert list(result) == EXPECTED


@pytest.mark.parametrize("text", ['{"tasks":[]}', '{ "tasks" : [ ] }'])
def test_empty_task_list(text: str) -> None:
    result = MistralToolsHandler()._parse_streaming_response(
        IterableModel(Person),
        iter(event(char) for char in text),
        validation_context=None,
        strict=None,
    )
    assert list(result) == []


def test_items_wait_for_the_complete_task_list_wrapper() -> None:
    consumed: list[int] = []

    def stream() -> Any:
        consumed.append(1)
        yield event('{"tasks":[{"name":"Ada"},')
        consumed.append(2)
        yield event('{"name":"Grace"}]}')

    result = MistralToolsHandler()._parse_streaming_response(
        IterableModel(Person),
        stream(),
        validation_context=None,
        strict=None,
    )
    assert next(result) == Person(name="Ada")
    assert consumed == [1, 2]
    assert list(result) == [Person(name="Grace")]


def test_nested_tasks_field_is_not_unwrapped_as_outer_tasks() -> None:
    class Group(BaseModel):
        tasks: list[Person]

    text = '{"tasks":[{"tasks":[{"name":"Ada"}]}]}'
    result = MistralToolsHandler()._parse_streaming_response(
        IterableModel(Group),
        iter(event(char) for char in text),
        validation_context=None,
        strict=None,
    )
    assert list(result) == [Group(tasks=[Person(name="Ada")])]


COMPATIBLE_WRAPPERS = [
    '{"metadata":0,"tasks":[{"name":"Ada"}]}',
    '{"tasks":[{"name":"Ada"}],"metadata":{"name":"not a task"}}',
    '{"metadata":{"nested":[{"name":"not a task"}]},"tasks":[{"name":"Ada"}]}',
    '{"\\u0074asks":[{"name":"Ada"}]}',
    ' \t\r\n{"tasks":[{"name":"Ada"}]} \r\n',
]


@pytest.mark.parametrize("text", COMPATIBLE_WRAPPERS)
@pytest.mark.parametrize("split_characters", [False, True])
def test_complete_wrapper_semantics(text: str, split_characters: bool) -> None:
    fragments = list(text) if split_characters else [text]
    result = MistralToolsHandler()._parse_streaming_response(
        IterableModel(Person), iter(event(part) for part in fragments), None, None
    )
    assert list(result) == [Person(name="Ada")]


@pytest.mark.asyncio
@pytest.mark.parametrize("text", COMPATIBLE_WRAPPERS)
@pytest.mark.parametrize("split_characters", [False, True])
async def test_async_complete_wrapper_semantics(
    text: str, split_characters: bool
) -> None:
    fragments = list(text) if split_characters else [text]

    async def stream() -> AsyncIterator[Any]:
        for part in fragments:
            yield event(part)

    result = MistralToolsHandler()._parse_streaming_response(
        IterableModel(Person), stream(), None, None
    )
    assert [item async for item in result] == [Person(name="Ada")]


INVALID_ARGUMENTS = [
    'garbage {"tasks":[{"name":"Ada"}]}',
    '{"tasks":[{"name":"Ada"}]} garbage',
    '{"tasks":[{"name":"Ada"}]} {"tasks":[]}',
    '\u00a0{"tasks":[{"name":"Ada"}]}',
    '{"tasks":[{"name":"Ada"}]}\v',
    '[{"name":"Ada"}]',
    "null",
    "42",
    '"not an object"',
    " \t\r\n",
]


@pytest.mark.parametrize("arguments", INVALID_ARGUMENTS)
def test_invalid_complete_arguments_do_not_yield_items(arguments: str) -> None:
    result = MistralToolsHandler()._parse_streaming_response(
        IterableModel(Person), iter([event(arguments)]), None, None
    )
    items = []
    with pytest.raises((json.JSONDecodeError, TypeError)):
        for item in result:
            items.append(item)
    assert items == []


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", INVALID_ARGUMENTS)
async def test_async_invalid_complete_arguments_do_not_yield_items(
    arguments: str,
) -> None:
    async def stream() -> AsyncIterator[Any]:
        yield event(arguments)

    result = MistralToolsHandler()._parse_streaming_response(
        IterableModel(Person), stream(), None, None
    )
    items = []
    with pytest.raises((json.JSONDecodeError, TypeError)):
        async for item in result:
            items.append(item)
    assert items == []


@pytest.mark.parametrize("arguments", ['{"tasks":[7]}', '{"tasks":[null]}'])
def test_scalar_task_items_are_not_silently_discarded(arguments: str) -> None:
    from pydantic import ValidationError

    result = MistralToolsHandler()._parse_streaming_response(
        IterableModel(Person), iter([event(arguments)]), None, None
    )
    with pytest.raises(ValidationError):
        list(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", ['{"tasks":[7]}', '{"tasks":[null]}'])
async def test_async_scalar_task_items_are_not_silently_discarded(
    arguments: str,
) -> None:
    from pydantic import ValidationError

    async def stream() -> AsyncIterator[Any]:
        yield event(arguments)

    result = MistralToolsHandler()._parse_streaming_response(
        IterableModel(Person), stream(), None, None
    )
    with pytest.raises(ValidationError):
        [item async for item in result]


def test_multiple_complete_wrappers_remain_separate() -> None:
    fragments = ["", '{"tasks":[]}', '{"tasks":[{"name":"Ada"}]}', TEXT]
    result = MistralToolsHandler()._parse_streaming_response(
        IterableModel(Person), iter(event(part) for part in fragments), None, None
    )
    assert list(result) == [Person(name="Ada"), *EXPECTED]


@pytest.mark.asyncio
async def test_async_multiple_complete_wrappers_remain_separate() -> None:
    async def stream() -> AsyncIterator[Any]:
        for part in ["", '{"tasks":[]}', '{"tasks":[{"name":"Ada"}]}', TEXT]:
            yield event(part)

    result = MistralToolsHandler()._parse_streaming_response(
        IterableModel(Person), stream(), None, None
    )
    assert [item async for item in result] == [Person(name="Ada"), *EXPECTED]


def test_dict_arguments_keep_raising_instead_of_silently_disappearing() -> None:
    # The SDK allows dict arguments, but this streaming path did not support them.
    result = MistralToolsHandler()._parse_streaming_response(
        IterableModel(Person), iter([event({"tasks": [{"name": "Ada"}]})]), None, None
    )
    with pytest.raises(TypeError):
        list(result)


@pytest.mark.asyncio
async def test_async_dict_arguments_keep_raising() -> None:
    async def stream() -> AsyncIterator[Any]:
        yield event({"tasks": [{"name": "Ada"}]})

    result = MistralToolsHandler()._parse_streaming_response(
        IterableModel(Person), stream(), None, None
    )
    with pytest.raises(TypeError):
        [item async for item in result]


def test_close_after_first_item_closes_source_once() -> None:
    closed: list[bool] = []

    def stream() -> Any:
        try:
            yield event(TEXT)
            pytest.fail("closing must not consume another event")
        finally:
            closed.append(True)

    result = MistralToolsHandler()._parse_streaming_response(
        IterableModel(Person), stream(), None, None
    )
    assert next(result) == EXPECTED[0]
    result.close()
    assert closed == [True]


@pytest.mark.asyncio
async def test_async_close_after_first_item_closes_source_once() -> None:
    closed: list[bool] = []

    async def stream() -> AsyncIterator[Any]:
        try:
            yield event(TEXT)
            pytest.fail("closing must not consume another event")
        finally:
            closed.append(True)

    result = MistralToolsHandler()._parse_streaming_response(
        IterableModel(Person), stream(), None, None
    )
    assert await anext(result) == EXPECTED[0]
    await result.aclose()
    assert closed == [True]


def test_iterable_string_adapter_keeps_its_values() -> None:
    from collections.abc import Iterable

    handler = MistralToolsHandler()
    response_model, _ = handler.prepare_request(
        cast(Any, Iterable[str]), {"stream": True, "messages": []}
    )
    assert response_model is not None
    text = '{"tasks":[{"content":"Ada"},{"content":"Grace"}]}'
    result = handler._parse_streaming_response(
        response_model, iter(event(char) for char in text), None, None
    )
    assert list(result) == ["Ada", "Grace"]


@pytest.mark.asyncio
async def test_async_iterable_string_adapter_keeps_its_values() -> None:
    from collections.abc import Iterable

    handler = MistralToolsHandler()
    response_model, _ = handler.prepare_request(
        cast(Any, Iterable[str]), {"stream": True, "messages": []}
    )
    assert response_model is not None

    async def stream() -> AsyncIterator[Any]:
        for char in '{"tasks":[{"content":"Ada"},{"content":"Grace"}]}':
            yield event(char)

    result = handler._parse_streaming_response(response_model, stream(), None, None)
    assert [item async for item in result] == ["Ada", "Grace"]


def test_root_model_scalar_items_are_not_discarded() -> None:
    from pydantic import RootModel

    text = '{"tasks":["Ada","Grace"]}'
    result = MistralToolsHandler()._parse_streaming_response(
        IterableModel(RootModel[str]), iter(event(char) for char in text), None, None
    )
    assert [item.root for item in result] == ["Ada", "Grace"]


@pytest.mark.asyncio
async def test_async_root_model_scalar_items_are_not_discarded() -> None:
    from pydantic import RootModel

    async def stream() -> AsyncIterator[Any]:
        for char in '{"tasks":["Ada","Grace"]}':
            yield event(char)

    result = MistralToolsHandler()._parse_streaming_response(
        IterableModel(RootModel[str]), stream(), None, None
    )
    assert [item.root async for item in result] == ["Ada", "Grace"]


@pytest.mark.parametrize("strict", [False, True])
def test_buffering_preserves_context_and_strict_validation(strict: bool) -> None:
    from pydantic import ValidationError, ValidationInfo, field_validator

    class ContextItem(BaseModel):
        age: int

        @field_validator("age")
        @classmethod
        def add_context(cls, value: int, info: ValidationInfo) -> int:
            assert info.context == {"increment": 1}
            return value + info.context["increment"]

    text = '{"tasks":[{"age":"42"}]}'
    result = MistralToolsHandler()._parse_streaming_response(
        IterableModel(ContextItem),
        iter(event(char) for char in text),
        {"increment": 1},
        strict,
    )
    if strict:
        with pytest.raises(ValidationError):
            list(result)
    else:
        assert [item.age for item in result] == [43]


@pytest.mark.asyncio
@pytest.mark.parametrize("strict", [False, True])
async def test_async_buffering_preserves_context_and_strict_validation(
    strict: bool,
) -> None:
    from pydantic import ValidationError, ValidationInfo, field_validator

    class ContextItem(BaseModel):
        age: int

        @field_validator("age")
        @classmethod
        def add_context(cls, value: int, info: ValidationInfo) -> int:
            assert info.context == {"increment": 1}
            return value + info.context["increment"]

    async def stream() -> AsyncIterator[Any]:
        for char in '{"tasks":[{"age":"42"}]}':
            yield event(char)

    result = MistralToolsHandler()._parse_streaming_response(
        IterableModel(ContextItem), stream(), {"increment": 1}, strict
    )
    if strict:
        with pytest.raises(ValidationError):
            [item async for item in result]
    else:
        assert [item.age async for item in result] == [43]
