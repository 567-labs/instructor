"""A streaming chunk may carry deltas for several parallel tool calls at once.

extract_streaming_json previously read only ``tool_calls[0]`` of every chunk, so
the fragments of every call after the first were silently discarded.
"""

from __future__ import annotations

import pytest

from types import SimpleNamespace
from typing import Any

from instructor.v2.providers.openai.handlers import OpenAIToolsHandler


def _delta_chunk(*tool_calls: SimpleNamespace) -> Any:
    delta = SimpleNamespace(content=None, tool_calls=list(tool_calls))
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=None)])


def _call(index: int, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        index=index,
        id=f"call-{index}",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def test_streaming_extractor_keeps_every_tool_call_in_a_chunk() -> None:
    handler = OpenAIToolsHandler()
    chunks = [
        _delta_chunk(_call(0, "_Foo", '{"name": "A'), _call(1, "_Foo", '{"name": "B')),
        _delta_chunk(_call(0, "_Foo", 'lice"}'), _call(1, "_Foo", 'ob"}')),
    ]

    parts = list(handler.extract_streaming_json(iter(chunks)))

    assert parts == ['{"name": "A', 'lice"}', '{"name": "B', 'ob"}']


def _sdk_event(parts):
    from openai.types.chat import ChatCompletionChunk

    calls = [
        {"index": index, "type": "function", "function": {"arguments": args}}
        for index, args in parts
    ]
    return ChatCompletionChunk.model_validate(
        {
            "id": "test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test",
            "choices": [{"index": 0, "delta": {"tool_calls": calls}}],
        }
    )


@pytest.mark.asyncio
async def test_interleaved_sdk_calls_parse_as_separate_items_sync_and_async():
    from pydantic import BaseModel
    from instructor.v2.dsl.iterable import IterableModel
    from instructor.v2.providers.openai.handlers import OpenAIToolsHandler

    class Item(BaseModel):
        name: str
        tags: list[str] = []

    chunks = [
        _sdk_event([(1, None), (0, '{"name":"A'), (1, '{"name":"B')]),
        _sdk_event([(1, 'ob","tags":["b"]}'), (0, 'da","tags":["a"]}')]),
    ]
    handler = OpenAIToolsHandler()
    model = IterableModel(Item)
    expected = [Item(name="Ada", tags=["a"]), Item(name="Bob", tags=["b"])]
    assert list(handler.parse_response(iter(chunks), model, stream=True)) == expected

    async def events():
        for chunk in chunks:
            yield chunk

    assert [
        item async for item in handler.parse_response(events(), model, stream=True)
    ] == expected
    assert (
        "".join(handler.extract_streaming_json(iter(chunks), single_model=True))
        == '{"name":"Ada","tags":["a"]}'
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("indices", [(0, 1), (1, 2)])
async def test_incomplete_call_cannot_consume_another_calls_arguments(indices):
    from pydantic import BaseModel
    from instructor.v2.core.errors import IncompleteOutputException
    from instructor.v2.dsl.iterable import IterableModel

    class Item(BaseModel):
        name: str

    first, second = indices
    chunk = _sdk_event([(first, '{"name":'), (second, '"cross-call"}')])
    model = IterableModel(Item)
    handler = OpenAIToolsHandler()
    with pytest.raises(IncompleteOutputException):
        list(handler.parse_response(iter([chunk]), model, stream=True))

    async def events():
        yield chunk

    with pytest.raises(IncompleteOutputException):
        _ = [
            item async for item in handler.parse_response(events(), model, stream=True)
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_iterable_reports_length_limit_between_complete_items(
    is_async: bool,
) -> None:
    from openai.types.chat import ChatCompletionChunk
    from pydantic import BaseModel
    from instructor.v2.core.errors import IncompleteOutputException
    from instructor.v2.dsl.iterable import IterableModel

    class Item(BaseModel):
        name: str

    chunks = [
        _sdk_event([(0, '{"tasks":[{"name":"Ada"},')]),
        ChatCompletionChunk.model_validate(
            {
                "id": "test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "test",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
            }
        ),
    ]
    handler = OpenAIToolsHandler()
    model = IterableModel(Item)
    if is_async:

        async def events():
            for chunk in chunks:
                yield chunk

        items = handler.parse_response(events(), model, stream=True)
        assert await items.__anext__() == Item(name="Ada")
        with pytest.raises(IncompleteOutputException):
            _ = [item async for item in items]
    else:
        items = handler.parse_response(iter(chunks), model, stream=True)
        assert next(items) == Item(name="Ada")
        with pytest.raises(IncompleteOutputException):
            list(items)
