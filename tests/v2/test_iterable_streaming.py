from __future__ import annotations

from collections.abc import Iterable
import sys
from typing import Any, Union, cast

import pytest
from pydantic import BaseModel

from instructor.v2.core.response_model import prepare_response_model
from instructor.v2.dsl.iterable import IterableBase, IterableModel


class User(BaseModel):
    name: str
    bio: str


class Weather(BaseModel):
    location: str


class GoogleSearch(BaseModel):
    query: str


def test_iterable_get_object_ignores_braces_inside_strings() -> None:
    obj, rest = IterableBase.get_object('{"bio": "a } b"},{"bio": "next"}', 0)

    assert obj == '{"bio": "a } b"}'
    assert rest == '{"bio": "next"}'


def test_iterable_get_object_handles_escaped_quotes_before_brace() -> None:
    obj, rest = IterableBase.get_object(
        '{"bio": "quote \\" } still string"},{"bio": "next"}',
        0,
    )

    assert obj == '{"bio": "quote \\" } still string"}'
    assert rest == '{"bio": "next"}'


def test_iterable_get_object_handles_even_backslashes_before_quote() -> None:
    obj, rest = IterableBase.get_object(
        r'{"bio": "ends \\"},{"bio": "next"}',
        0,
    )

    assert obj == r'{"bio": "ends \\"}'
    assert rest == '{"bio": "next"}'


def test_iterable_tasks_from_chunks_handles_braces_inside_strings() -> None:
    chunks = [
        '{"tasks": [',
        '{"name": "Alice", "bio": "happy :}"}',
        ', {"name": "Bob", "bio": "plain"}',
        "]}",
    ]

    iterable_model = cast(Any, IterableModel(User))
    users = list(iterable_model.tasks_from_chunks(chunks))

    assert users == [
        User(name="Alice", bio="happy :}"),
        User(name="Bob", bio="plain"),
    ]


UNION_CHUNKS = [
    '{"tasks": [',
    '{"location": "Toronto"}',
    ', {"query": "super bowl winner"}',
    "]}",
]

EXPECTED_UNION_TASKS = [
    Weather(location="Toronto"),
    GoogleSearch(query="super bowl winner"),
]

TASK_TYPE_CASES = [pytest.param(Union[Weather, GoogleSearch], id="typing-union")]
RESPONSE_MODEL_CASES = [
    pytest.param(Iterable[Union[Weather, GoogleSearch]], id="typing-union")
]
if sys.version_info >= (3, 10):
    TASK_TYPE_CASES.append(pytest.param(Weather | GoogleSearch, id="pep604-union"))
    RESPONSE_MODEL_CASES.append(
        pytest.param(Iterable[Weather | GoogleSearch], id="pep604-union")
    )


@pytest.mark.parametrize(
    "task_type",
    TASK_TYPE_CASES,
)
def test_iterable_streaming_parses_both_union_spellings(task_type: Any) -> None:
    """`A | B` must stream like `Union[A, B]`."""
    iterable_model = cast(Any, IterableModel(cast(type[BaseModel], task_type)))

    assert list(iterable_model.tasks_from_chunks(UNION_CHUNKS)) == EXPECTED_UNION_TASKS


@pytest.mark.parametrize(
    "response_model",
    RESPONSE_MODEL_CASES,
)
def test_create_iterable_response_model_streams_union_members(
    response_model: Any,
) -> None:
    prepared = cast(Any, prepare_response_model(response_model))

    assert prepared.__name__ == "IterableWeatherOrGoogleSearch"
    assert list(prepared.tasks_from_chunks(UNION_CHUNKS)) == EXPECTED_UNION_TASKS


@pytest.mark.skipif(sys.version_info < (3, 10), reason="PEP 604 requires Python 3.10")
def test_iterable_pep604_union_reports_unmatched_payload_as_value_error() -> None:
    iterable_model = cast(
        Any, IterableModel(cast(type[BaseModel], Weather | GoogleSearch))
    )

    with pytest.raises(ValueError, match="Failed to extract task type"):
        iterable_model.extract_cls_task_type('{"unrelated": true}')


@pytest.mark.asyncio
async def test_streaming_is_independent_of_chunk_boundaries() -> None:
    """Exercise every split, including escapes, strings and object delimiters."""
    import json

    expected = [
        User(name="Alice", bio='braces } {, brackets ][, quote " and slash \\'),
        User(name="Bob", bio="雪"),
    ]
    payload = json.dumps({"tasks": [user.model_dump() for user in expected]})
    model = cast(Any, IterableModel(User))
    partitions = [[payload[:i], payload[i:]] for i in range(len(payload) + 1)]
    partitions.append(list(payload))
    for chunks in partitions:
        assert list(model.tasks_from_chunks(chunks)) == expected, chunks

        async def source(parts=chunks):
            for chunk in parts:
                yield chunk

        actual = [item async for item in model.tasks_from_chunks_async(source())]
        assert actual == expected, chunks


# A stream cut off by the provider (e.g. at max_tokens) ends inside an item. The
# non-streaming path raises IncompleteOutputException for this; streaming must too,
# and must still yield every item that did arrive complete.
TRUNCATED_CHUNKS = [
    '{"tasks": [',
    '{"name": "Alice", "bio": "first"}',
    ', {"name": "Bob", "bio": "quote \\" and } inside, then cut',
]


def test_tasks_from_chunks_raises_when_stream_ends_inside_an_item() -> None:
    from instructor.v2.core.errors import IncompleteOutputException

    model = cast(Any, IterableModel(User))
    received = []
    with pytest.raises(IncompleteOutputException):
        for user in model.tasks_from_chunks(TRUNCATED_CHUNKS):
            received.append(user)

    assert received == [User(name="Alice", bio="first")]


@pytest.mark.asyncio
async def test_tasks_from_chunks_async_raises_when_stream_ends_inside_an_item() -> None:
    from instructor.v2.core.errors import IncompleteOutputException

    async def source():
        for chunk in TRUNCATED_CHUNKS:
            yield chunk

    model = cast(Any, IterableModel(User))
    received = []
    with pytest.raises(IncompleteOutputException):
        async for user in model.tasks_from_chunks_async(source()):
            received.append(user)

    assert received == [User(name="Alice", bio="first")]


@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        (['{"tasks": []}'], []),
        (
            ['{"tasks": [{"name": "Alice", "bio": "a { b"}]}'],
            [User(name="Alice", bio="a { b")],
        ),
        (
            ['{"tasks": [{"name": "Alice", "bio": "x"}], "note": "has { inside"}'],
            [User(name="Alice", bio="x")],
        ),
    ],
)
def test_tasks_from_chunks_complete_stream_does_not_raise(
    chunks: list[str], expected: list[User]
) -> None:
    model = cast(Any, IterableModel(User))
    assert list(model.tasks_from_chunks(chunks)) == expected
