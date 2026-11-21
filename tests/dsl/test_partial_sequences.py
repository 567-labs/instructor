from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from typing import Any, cast

import pytest
from pydantic import BaseModel, ValidationError, ValidationInfo, field_validator

from instructor.v2.dsl.partial import Partial


class Item(BaseModel):
    name: str
    age: int

    @field_validator("name")
    @classmethod
    def uppercase_name(cls, value: str, info: ValidationInfo) -> str:
        return value.upper() if (info.context or {}).get("uppercase") else value


class Envelope(BaseModel):
    items: Sequence[Item]
    note: str


class NullableEnvelope(BaseModel):
    items: Sequence[Item] | None = None
    note: str


async def snapshots(
    schema: type[BaseModel], chunks: list[str], asynchronous: bool, **kwargs: Any
) -> list[Any]:
    partial = cast(Any, Partial[schema])
    if asynchronous:

        async def source() -> AsyncGenerator[str, None]:
            for chunk in chunks:
                yield chunk

        return [
            item async for item in partial.model_from_chunks_async(source(), **kwargs)
        ]
    return list(partial.model_from_chunks(chunks, **kwargs))


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("schema", [Envelope, NullableEnvelope])
async def test_sequence_items_stay_models_while_streaming(
    schema: type[BaseModel], asynchronous: bool
) -> None:
    chunks = [
        '{"items":[{"name":"Alice","age":30},',
        '{"name":"Bo',
        'b","age":40}],"note":"d',
        'one"}',
    ]
    outputs = await snapshots(schema, chunks, asynchronous, context={"uppercase": True})

    assert isinstance(outputs[0].items[0], Item)
    assert outputs[0].items[0].age == 30
    assert outputs[0].note is None
    assert isinstance(outputs[1].items[1], Item)
    assert outputs[1].items[0].name == "ALICE"
    assert outputs[1].items[1].name == "Bo"
    assert outputs[1].items[1].age is None
    assert [item.name for item in outputs[2].items] == ["ALICE", "BOB"]
    assert outputs[-1] == schema.model_validate_json(
        "".join(chunks), context={"uppercase": True}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("schema", [Envelope, NullableEnvelope])
async def test_completed_sequence_item_is_validated_before_root(
    schema: type[BaseModel], asynchronous: bool
) -> None:
    with pytest.raises(ValidationError, match="age"):
        await snapshots(
            schema,
            ['{"items":[{"name":"Alice","age":"invalid"}],"note":"d'],
            asynchronous,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_nested_sequences_build_partial_children(asynchronous: bool) -> None:
    class NestedEnvelope(BaseModel):
        groups: Sequence[Sequence[Item]]
        note: str

    outputs = await snapshots(NestedEnvelope, ['{"groups":[[{"name":"Al'], asynchronous)
    item = outputs[0].groups[0][0]
    assert isinstance(item, Item)
    assert item.name == "Al"
    assert item.age is None
