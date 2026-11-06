from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, List, Optional, Union, cast  # noqa: UP035

import pytest
from pydantic import BaseModel, Field, ValidationError, ValidationInfo, field_validator

from instructor.v2.dsl.partial import Partial


class Item(BaseModel):
    name: str
    age: int

    @field_validator("name")
    @classmethod
    def uppercase_name(cls, value: str, info: ValidationInfo) -> str:
        return (
            value.upper() if info.context and info.context.get("uppercase") else value
        )


class PlainEnvelope(BaseModel):
    items: list[Item]
    note: str


class OptionalEnvelope(BaseModel):
    items: Optional[list[Item]] = None  # noqa: UP007, UP045
    note: str


class LegacyEnvelope(BaseModel):
    items: Optional[List[Item]] = None  # noqa: UP006, UP007, UP045
    note: str


class UnionEnvelope(BaseModel):
    items: list[Item] | None = None
    note: str


ENVELOPES = [PlainEnvelope, OptionalEnvelope, LegacyEnvelope, UnionEnvelope]
NULLABLE_ENVELOPES = ENVELOPES[1:]
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"]),
]


async def stream(
    schema: type[BaseModel],
    chunks: list[str],
    asynchronous: bool,
    **kwargs: Any,
) -> AsyncGenerator[Any, None]:
    # Partial adds these streaming methods dynamically.
    api = cast(Any, Partial[schema])
    if asynchronous:

        async def source() -> AsyncGenerator[str, None]:
            for chunk in chunks:
                yield chunk

        async for obj in api.model_from_chunks_async(source(), **kwargs):
            yield obj
    else:
        for obj in api.model_from_chunks(iter(chunks), **kwargs):
            yield obj


@pytest.mark.parametrize("schema", ENVELOPES)
async def test_incremental_items_remain_models(
    schema: type[BaseModel], asynchronous: bool
) -> None:
    chunks = [
        '{"items":[{"name":"Alice","age":30},',
        '{"name":"Bo',
        'b","age":40}]',
        ',"note":"done"}',
    ]
    outputs = [obj async for obj in stream(schema, chunks, asynchronous)]
    assert len(outputs) == len(chunks)
    assert isinstance(outputs[0].items[0], Item)
    assert outputs[0].items[0].name == "Alice"
    assert outputs[0].note is None
    assert isinstance(outputs[1].items[1], Item)
    assert outputs[1].items[1].name == "Bo"
    assert outputs[1].items[1].age is None
    assert isinstance(outputs[2].items[1], Item)
    assert outputs[2].items[1].name == "Bob"
    assert outputs[2].note is None
    assert outputs[-1] == schema.model_validate_json("".join(chunks))


@pytest.mark.parametrize("schema", NULLABLE_ENVELOPES)
@pytest.mark.parametrize(
    ("start", "expected"),
    [('{"items":null,', None), ('{"items":[],', []), ("{", None)],
)
async def test_null_empty_and_missing(
    schema: type[BaseModel], start: str, expected: Any, asynchronous: bool
) -> None:
    outputs = [
        obj async for obj in stream(schema, [start, '"note":"done"}'], asynchronous)
    ]
    assert outputs[0].items == expected
    assert outputs[-1].items == expected


async def test_missing_default_factory(asynchronous: bool) -> None:
    class Envelope(BaseModel):
        items: Optional[list[Item]] = Field(default_factory=list)  # noqa: UP007, UP045
        note: str

    outputs = [
        obj async for obj in stream(Envelope, ["{", '"note":"done"}'], asynchronous)
    ]
    assert outputs[0].items == []
    assert outputs[-1].items == []


@pytest.mark.parametrize("schema", ENVELOPES)
async def test_validation_context_reaches_complete_item(
    schema: type[BaseModel], asynchronous: bool
) -> None:
    # The tracker marks items complete once the next field starts.
    chunks = ['{"items":[{"name":"Alice","age":30}],"note":"d', 'one"}']
    outputs = [
        obj
        async for obj in stream(
            schema, chunks, asynchronous, context={"uppercase": True}
        )
    ]
    assert outputs[0].items[0].name == "ALICE"
    assert outputs[0].note == "d"
    assert outputs[-1].items[0].name == "ALICE"


@pytest.mark.parametrize("schema", ENVELOPES)
async def test_complete_item_is_validated_before_root(
    schema: type[BaseModel], asynchronous: bool
) -> None:
    with pytest.raises(ValidationError, match="age"):
        async for _ in stream(
            schema, ['{"items":[{"name":"Alice"}],"note":"d'], asynchronous
        ):
            pass


@pytest.mark.parametrize("schema", ENVELOPES)
async def test_final_validation_still_requires_note(
    schema: type[BaseModel], asynchronous: bool
) -> None:
    with pytest.raises(ValidationError, match="note"):
        async for _ in stream(
            schema, ['{"items":[{"name":"Alice","age":30}]}'], asynchronous
        ):
            pass


async def test_optional_items_and_ambiguous_collections(asynchronous: bool) -> None:
    class OptionalItems(BaseModel):
        items: Optional[list[Optional[Item]]] = None  # noqa: UP007, UP045
        note: str

    chunks = ['{"items":[null,{"name":"Alice","age":30}],', '"note":"done"}']
    outputs = [obj async for obj in stream(OptionalItems, chunks, asynchronous)]
    assert outputs[0].items[0] is None
    assert isinstance(outputs[0].items[1], Item)
    assert outputs[-1].items[0] is None

    class OtherItem(BaseModel):
        label: str

    class AmbiguousEnvelope(BaseModel):
        items: Union[list[Item], list[OtherItem], None] = None  # noqa: UP007
        note: str

    chunks = ['{"items":[{"label":"other"}],', '"note":"done"}']
    outputs = [obj async for obj in stream(AmbiguousEnvelope, chunks, asynchronous)]
    assert outputs[0].items == [{"label": "other"}]
    assert isinstance(outputs[-1].items[0], OtherItem)
