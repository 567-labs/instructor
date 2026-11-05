from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Union, cast

import pytest
from pydantic import BaseModel

from instructor import Mode, Provider
from instructor.v2.core.patch import patch
from instructor.v2.core.response_model import prepare_response_model
from instructor.v2.dsl.response_list import ListResponse
from tests.coverage._openai import chat_completion, tool_call


@pytest.mark.parametrize(
    ("annotation", "wire", "expected", "json_type"),
    [
        (dict[str, int], {"key": 7}, {"key": 7}, "object"),
        (set[int], [1, 1, 2], {1, 2}, "array"),
        (tuple[int, str], [3, "x"], (3, "x"), "array"),
        (dict, {"nested": [1]}, {"nested": [1]}, "object"),
        (tuple, ["a", 1], ("a", 1), "array"),
    ],
)
def test_create_returns_native_collection_values(annotation, wire, expected, json_type):
    prepared = cast(Any, prepare_response_model(annotation))
    prop = prepared.model_json_schema()["properties"]["content"]
    assert prop["type"] == json_type
    if annotation == set[int]:
        assert prop["uniqueItems"] is True

    def create(**kwargs):
        name = kwargs["tools"][0]["function"]["name"]
        return chat_completion(tool_calls=[tool_call(name, {"content": wire})])

    wrapped = cast(Any, patch(create=create, mode=Mode.TOOLS, provider=Provider.OPENAI))
    result = wrapped(
        messages=[{"role": "user", "content": "example"}], response_model=annotation
    )
    assert type(result) is type(expected)
    assert result == expected


@pytest.mark.parametrize(
    ("element", "items"),
    [(str, ["hello", "world"]), (dict[str, int], [{"k": 1}, {"k": 2}])],
)
@pytest.mark.asyncio
async def test_iterable_items_are_native_in_sync_async_and_nonstreaming_results(
    element, items
):
    import json

    annotation = Iterable[element]
    prepared = cast(Any, prepare_response_model(annotation))
    payload = {"tasks": [{"content": item} for item in items]}
    chunks = list(json.dumps(payload))
    assert list(prepared.tasks_from_chunks(chunks)) == items

    async def stream():
        for chunk in chunks:
            yield chunk

    assert [item async for item in prepared.tasks_from_chunks_async(stream())] == items
    completions = []

    def create(**kwargs):
        name = kwargs["tools"][0]["function"]["name"]
        completion = chat_completion(tool_calls=[tool_call(name, payload)])
        completions.append(completion)
        return completion

    wrapped = cast(Any, patch(create=create, mode=Mode.TOOLS, provider=Provider.OPENAI))
    result = wrapped(
        messages=[{"role": "user", "content": "example"}], response_model=annotation
    )
    assert isinstance(result, ListResponse)
    assert result == items
    assert result._raw_response is completions[0]


def test_mixed_union_iterable_preserves_models_and_scalars():
    class Named(BaseModel):
        name: str

    prepared = cast(Any, prepare_response_model(Iterable[Union[Named, int]]))
    assert list(
        prepared.tasks_from_chunks(
            ['{"tasks":[{"content":{"name":"Ada"}},{"content":9}]}']
        )
    ) == [Named(name="Ada"), 9]
