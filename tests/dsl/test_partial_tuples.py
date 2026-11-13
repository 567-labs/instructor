from collections.abc import AsyncGenerator
from typing import Any, cast

import pytest
from pydantic import BaseModel, ValidationError, ValidationInfo, field_validator

from instructor import Partial


class Item(BaseModel):
    name: str
    age: int

    @field_validator("name")
    @classmethod
    def uppercase_name(cls, value: str, info: ValidationInfo) -> str:
        return (
            value.upper() if info.context and info.context.get("uppercase") else value
        )


class Envelope(BaseModel):
    items: tuple[Item, Item]


class VariadicEnvelope(BaseModel):
    items: tuple[Item, ...]


class OptionalEnvelope(BaseModel):
    items: tuple[Item, ...] | None = None


async def snapshots(
    schema: type[BaseModel], chunks: list[str], asynchronous: bool, **kwargs: Any
) -> list[Any]:
    api = cast(Any, Partial[schema])
    if asynchronous:

        async def source() -> AsyncGenerator[str, None]:
            for chunk in chunks:
                yield chunk

        return [
            value async for value in api.model_from_chunks_async(source(), **kwargs)
        ]
    return list(api.model_from_chunks(iter(chunks), **kwargs))


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("schema", [Envelope, VariadicEnvelope, OptionalEnvelope])
async def test_tuple_items_remain_models_while_streaming(
    schema: type[BaseModel], asynchronous: bool
) -> None:
    chunks = [
        '{"items":[{"name":"Alice","age":30},{"name":"Bo',
        'b","age":40}]}',
    ]
    outputs = await snapshots(schema, chunks, asynchronous)

    assert isinstance(outputs[0].items, tuple)
    assert isinstance(outputs[0].items[0], Item)
    assert outputs[0].items[0].name == "Alice"
    assert isinstance(outputs[0].items[1], Item)
    assert outputs[0].items[1].name == "Bo"
    assert outputs[0].items[1].age is None
    assert outputs[-1] == schema.model_validate_json("".join(chunks))


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_heterogeneous_tuple_uses_each_position_type(asynchronous: bool) -> None:
    class Label(BaseModel):
        text: str

    class MixedEnvelope(BaseModel):
        items: tuple[Item, Label]

    outputs = await snapshots(
        MixedEnvelope,
        ['{"items":[{"name":"Alice","age":30},{"text":"Bo', 'b"}]}'],
        asynchronous,
    )
    assert isinstance(outputs[0].items[0], Item)
    assert isinstance(outputs[0].items[1], Label)
    assert outputs[0].items[1].text == "Bo"
    assert outputs[-1].items[1].text == "Bob"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_tuple_inside_mapping_keeps_nested_models(asynchronous: bool) -> None:
    class Groups(BaseModel):
        groups: dict[str, tuple[Item, ...]]

    outputs = await snapshots(
        Groups,
        ['{"groups":{"team":[{"name":"Al', 'ice","age":30}]}}'],
        asynchronous,
    )
    assert isinstance(outputs[0].groups["team"], tuple)
    assert outputs[0].groups["team"][0].name == "Al"
    assert outputs[0].groups["team"][0].age is None
    assert outputs[-1].groups["team"][0].age == 30


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("prefix", "expected"),
    [('{"items":[],', ()), ('{"items":null,', None), ("{", None)],
)
async def test_empty_null_and_missing_optional_tuples(
    prefix: str, expected: Any, asynchronous: bool
) -> None:
    outputs = await snapshots(
        OptionalEnvelope, [prefix, '"note":"done"}'], asynchronous
    )
    assert outputs[0].items == expected
    assert outputs[-1].items == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_closed_tuple_item_is_validated_before_parent_completes(
    asynchronous: bool,
) -> None:
    with pytest.raises(ValidationError, match="age"):
        await snapshots(
            VariadicEnvelope,
            ['{"items":[{"name":"Alice","age":"invalid"}],"note":"pending'],
            asynchronous,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_completed_tuple_preserves_pydantic_length_validation(
    asynchronous: bool,
) -> None:
    with pytest.raises(ValidationError):
        await snapshots(
            Envelope,
            ['{"items":[{"name":"Alice","age":30}]}'],
            asynchronous,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_explicitly_empty_tuple_keeps_its_type(asynchronous: bool) -> None:
    class EmptyEnvelope(BaseModel):
        items: tuple[()]

    outputs = await snapshots(
        EmptyEnvelope, ['{"items":[],', '"note":"done"}'], asynchronous
    )
    assert outputs[0].items == ()
    assert outputs[-1].items == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_validation_context_reaches_closed_tuple_item(asynchronous: bool) -> None:
    outputs = await snapshots(
        VariadicEnvelope,
        ['{"items":[{"name":"Alice","age":30}],"note":"pending', '"}'],
        asynchronous,
        context={"uppercase": True},
    )
    assert outputs[0].items[0].name == "ALICE"
    assert outputs[-1].items[0].name == "ALICE"
