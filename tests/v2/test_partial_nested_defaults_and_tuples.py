from __future__ import annotations

from typing import Any, Optional, cast

from pydantic import BaseModel

from instructor.v2.dsl.partial import Partial


class Child(BaseModel):
    value: Optional[int] = None  # noqa: UP007


class Parent(BaseModel):
    name: str
    child: Child = Child(value=7)


class Item(BaseModel):
    name: str


class FixedTuple(BaseModel):
    pair: tuple[Item, int]


class VariadicTuple(BaseModel):
    items: tuple[Item, ...]


def _last(model: type[BaseModel], chunks: list[str]) -> Any:
    partial = cast(Any, Partial[model])  # type: ignore[valid-type]
    return list(partial.model_from_chunks(chunks))[-1]


def test_missing_nested_model_uses_declared_default_mid_stream() -> None:
    result = _last(Parent, ['{"name":"a'])

    assert result.child == Child(value=7)


def test_fixed_tuple_snapshot_builds_models() -> None:
    result = _last(FixedTuple, ['{"pair":[{"name":"a"},', "3"])

    assert isinstance(result.pair, tuple)
    assert isinstance(result.pair[0], Item)
    assert result.pair[0].name == "a"


def test_variadic_tuple_snapshot_builds_models() -> None:
    result = _last(VariadicTuple, ['{"items":[{"name":"a"},{"name":"b'])

    assert isinstance(result.items, tuple)
    assert all(isinstance(item, Item) for item in result.items)
    assert [item.name for item in result.items] == ["a", "b"]
