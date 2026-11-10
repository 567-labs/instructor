from collections.abc import AsyncGenerator, Generator
from typing import Any, Protocol, TypeVar, cast

import pytest
from pydantic import BaseModel, RootModel, ValidationError

from instructor import Partial

T_Model = TypeVar("T_Model", bound=BaseModel, covariant=True)


class _PartialModelApi(Protocol[T_Model]):
    @classmethod
    def model_from_chunks(
        cls, chunks: list[str], **kwargs: Any
    ) -> Generator[T_Model, None, None]: ...

    @classmethod
    def model_from_chunks_async(
        cls, chunks: AsyncGenerator[str, None], **kwargs: Any
    ) -> AsyncGenerator[T_Model, None]: ...


async def snapshots(
    model: type[T_Model], chunks: list[str], asynchronous: bool, **kwargs: Any
) -> list[T_Model]:
    partial = cast(type[_PartialModelApi[T_Model]], Partial[model])

    async def source() -> AsyncGenerator[str, None]:
        for chunk in chunks:
            yield chunk

    return (
        [item async for item in partial.model_from_chunks_async(source(), **kwargs)]
        if asynchronous
        else list(partial.model_from_chunks(chunks, **kwargs))
    )


class Person(BaseModel):
    name: str
    age: int


class People(RootModel[dict[str, Person]]):
    pass


class Team(RootModel[list[Person]]):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_partial_root_mapping_preserves_models(asynchronous: bool) -> None:
    chunks = ['{"alice":{"name":"Al', 'ice","age":30}', "}"]
    results = await snapshots(People, chunks, asynchronous)

    assert all(isinstance(snapshot, People) for snapshot in results)
    assert isinstance(results[0].root["alice"], Person)
    assert results[0].root["alice"].name == "Al"
    assert results[0].root["alice"].age is None
    assert results[1].root["alice"] == Person(name="Alice", age=30)
    assert results[-1] == People.model_validate_json("".join(chunks))


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_partial_root_list_preserves_models(asynchronous: bool) -> None:
    chunks = ['[{"name":"Al', 'ice","age":30}', "]"]
    results = await snapshots(Team, chunks, asynchronous)

    assert all(isinstance(snapshot, Team) for snapshot in results)
    assert isinstance(results[0].root[0], Person)
    assert results[0].root[0].name == "Al"
    assert results[0].root[0].age is None
    assert results[1].root[0] == Person(name="Alice", age=30)
    assert results[-1] == Team.model_validate_json("".join(chunks))


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_nested_root_model_keeps_root_named_input(asynchronous: bool) -> None:
    class Envelope(BaseModel):
        people: People
        tail: str

    chunks = ['{"people":{"root":{"name":"Al', 'ice","age":30}},"tail":"done"}']
    results = await snapshots(Envelope, chunks, asynchronous)

    assert isinstance(results[0].people, People)
    assert isinstance(results[0].people.root["root"], Person)
    assert results[0].people.root["root"].name == "Al"
    assert results[-1] == Envelope.model_validate_json("".join(chunks))


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_root_mapping_validates_item_with_next_sibling(
    asynchronous: bool,
) -> None:
    with pytest.raises(ValidationError, match="age"):
        await snapshots(
            People,
            ['{"alice":{"name":"Alice","age":"invalid"},"bob":{'],
            asynchronous,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_partial_root_string_keeps_wrapper(asynchronous: bool) -> None:
    class Title(RootModel[str]):
        pass

    results = await snapshots(Title, ['"Al', 'ice"'], asynchronous)

    assert all(isinstance(result, Title) for result in results)
    assert [result.root for result in results] == ["Al", "Alice"]
