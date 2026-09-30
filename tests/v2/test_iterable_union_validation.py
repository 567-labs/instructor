from __future__ import annotations

import json
import operator
import sys
from collections.abc import AsyncGenerator
from typing import Any, Literal, Union, cast

import pytest
from pydantic import BaseModel, ValidationInfo, field_validator

from instructor.v2.dsl.iterable import IterableBase, IterableModel


class BasicContact(BaseModel):
    name: str


class DetailedContact(BaseModel):
    name: str
    email: str


class IntegerValue(BaseModel):
    value: int


class StringValue(BaseModel):
    value: str


class EmailTask(BaseModel):
    kind: Literal["email"]
    value: str

    @field_validator("value")
    @classmethod
    def allowed_domain(cls, value: str, info: ValidationInfo) -> str:
        if info.context and not value.endswith(info.context["domain"]):
            raise ValueError("Email domain is not allowed")
        return value


class CountTask(BaseModel):
    kind: Literal["count"]
    value: int


def make_union(first: type[BaseModel], second: type[BaseModel], style: str) -> Any:
    if style == "pipe":
        return operator.or_(first, second)
    return cast(Any, Union)[first, second]


async def async_chunks(chunks: list[str]) -> AsyncGenerator[str, None]:
    for chunk in chunks:
        yield chunk


async def parse_stream(
    model: Any, payload: str, parser: str, **kwargs: Any
) -> list[BaseModel]:
    if parser == "task-list":
        return list(model.tasks_from_task_list_chunks([payload], **kwargs))
    if parser == "async-task-list":
        return [
            item
            async for item in model.tasks_from_task_list_chunks_async(
                async_chunks([payload]), **kwargs
            )
        ]
    # Split every character so object boundaries cannot hide the regression.
    if parser == "chunks":
        return list(model.tasks_from_chunks(list(payload), **kwargs))
    return [
        item
        async for item in model.tasks_from_chunks_async(
            async_chunks(list(payload)), **kwargs
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "style",
    [
        "typing",
        pytest.param(
            "pipe",
            marks=pytest.mark.skipif(
                sys.version_info < (3, 10), reason="PEP 604 requires Python 3.10"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "parser", ["chunks", "async-chunks", "task-list", "async-task-list"]
)
@pytest.mark.parametrize(
    "first,second,data,expected",
    [
        (
            BasicContact,
            DetailedContact,
            {"name": "Ada", "email": "ada@example.com"},
            DetailedContact(name="Ada", email="ada@example.com"),
        ),
        (IntegerValue, StringValue, {"value": "123"}, StringValue(value="123")),
    ],
    ids=["preserve-fields", "preserve-exact-type"],
)
async def test_streaming_union_matches_full_validation(
    first: type[BaseModel],
    second: type[BaseModel],
    data: dict[str, Any],
    expected: BaseModel,
    style: str,
    parser: str,
) -> None:
    model = cast(Any, IterableModel(make_union(first, second, style)))
    payload = json.dumps({"tasks": [data]})
    assert model.model_validate_json(payload).tasks == [expected]
    assert await parse_stream(model, payload, parser) == [expected]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "parser", ["chunks", "async-chunks", "task-list", "async-task-list"]
)
async def test_union_stream_preserves_tags_context_and_strictness(parser: str) -> None:
    model = cast(Any, IterableModel(cast(Any, Union[EmailTask, CountTask])))
    payload = '{"tasks":[{"kind":"count","value":2},{"kind":"email","value":"ada@example.com"}]}'
    kwargs = {"context": {"domain": "example.com"}, "strict": True}
    assert (
        await parse_stream(model, payload, parser, **kwargs)
        == model.model_validate_json(payload, **kwargs).tasks
    )
    for invalid in [
        '{"tasks":[{"kind":"count","value":"2"}]}',
        '{"tasks":[{"kind":"email","value":"ada@other.com"}]}',
        '{"tasks":[{"kind":"unknown","value":2}]}',
    ]:
        with pytest.raises(ValueError, match="Failed to extract task type"):
            await parse_stream(model, invalid, parser, **kwargs)


def test_union_parser_does_not_share_the_wrong_validator_between_subclasses() -> None:
    class Contacts(IterableBase):
        task_type = cast(Any, Union[BasicContact, DetailedContact])

    class Values(Contacts):
        task_type = cast(Any, Union[IntegerValue, StringValue])

    assert Contacts.extract_cls_task_type(
        '{"name":"Ada","email":"ada@example.com"}'
    ) == DetailedContact(name="Ada", email="ada@example.com")
    assert Values.extract_cls_task_type('{"value":"123"}') == StringValue(value="123")


def test_union_stream_preserves_first_member_when_matches_tie() -> None:
    class OtherContact(BaseModel):
        name: str

    forward = cast(Any, IterableModel(cast(Any, Union[BasicContact, OtherContact])))
    reverse = cast(Any, IterableModel(cast(Any, Union[OtherContact, BasicContact])))
    payload = '{"name":"Ada"}'
    assert type(forward.extract_cls_task_type(payload)) is BasicContact
    assert type(reverse.extract_cls_task_type(payload)) is OtherContact


def test_union_validator_is_reused_and_tracks_task_type_changes() -> None:
    class Tasks(IterableBase):
        task_type = cast(Any, Union[IntegerValue, StringValue])

    assert Tasks.extract_cls_task_type('{"value":"123"}') == StringValue(value="123")
    adapter = Tasks._task_type_adapter
    assert Tasks.extract_cls_task_type('{"value":123}') == IntegerValue(value=123)
    assert Tasks._task_type_adapter is adapter

    Tasks.task_type = cast(Any, Union[BasicContact, DetailedContact])
    assert Tasks.extract_cls_task_type('{"name":"Ada"}') == BasicContact(name="Ada")
    assert Tasks._task_type_adapter is not adapter


def test_union_validator_does_not_swallow_programming_errors() -> None:
    class BrokenValue(BaseModel):
        value: str

        @field_validator("value")
        @classmethod
        def broken_validator(cls, _value: str) -> str:
            raise TypeError("validator bug")

    model = cast(Any, IterableModel(cast(Any, Union[BrokenValue, StringValue])))
    with pytest.raises(TypeError, match="validator bug"):
        model.model_validate_json('{"tasks":[{"value":"hello"}]}')
    with pytest.raises(TypeError, match="validator bug"):
        model.extract_cls_task_type('{"value":"hello"}')
