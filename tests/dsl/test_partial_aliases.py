"""Partial streaming must resolve fields that arrive under a validation alias.

The streamed JSON is keyed by the alias ``model_json_schema()`` advertises, while
``model.model_fields`` is keyed by the Python attribute name, so a name-only
lookup silently misses every aliased field.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from instructor.v2.dsl.partial import Partial


class Location(BaseModel):
    city: str
    lat: float


class Tag(BaseModel):
    name: str
    score: float


class PlainEnvelope(BaseModel):
    answer: str
    location: Location


class AliasedEnvelope(BaseModel):
    answer: str
    location: Location = Field(alias="loc")


class AliasedListEnvelope(BaseModel):
    title: str
    tags: list[Tag] = Field(alias="t")
    meta: Optional[Tag] = Field(default=None, alias="m")  # noqa: UP007, UP045


def _last(model: type[BaseModel], chunk: str):
    partial_model = Partial[model]
    return list(partial_model.model_from_chunks(iter([chunk])))[-1]


# Each snapshot leaves the root open with a trailing unterminated key, so the
# nested value itself is complete while the enclosing object is not.
ALIAS_CHUNK = '{"answer": "hi", "loc": {"city": "NYC", "lat": 1.5}, "tail": '
PLAIN_CHUNK = '{"answer": "hi", "location": {"city": "NYC", "lat": 1.5}, "tail": '
LIST_CHUNK = (
    '{"title": "x", "t": [{"name": "a", "score": 1.0}], '
    '"m": {"name": "b", "score": 2.0}, "z": '
)


def test_complete_nested_object_without_alias_is_validated() -> None:
    obj = _last(PlainEnvelope, PLAIN_CHUNK)
    assert isinstance(obj.location, Location)


def test_complete_nested_object_under_alias_is_validated() -> None:
    obj = _last(AliasedEnvelope, ALIAS_CHUNK)
    assert isinstance(obj.location, Location), f"got {type(obj.location).__name__}"
    assert obj.location.city == "NYC"
    assert obj.location.lat == 1.5


def test_alias_list_and_optional_nested_are_models() -> None:
    obj = _last(AliasedListEnvelope, LIST_CHUNK)
    assert all(isinstance(tag, Tag) for tag in obj.tags), f"got {obj.tags!r}"
    assert isinstance(obj.meta, Tag), f"got {obj.meta!r}"
