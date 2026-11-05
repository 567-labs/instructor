"""Cached responses must reload into the model that produced them."""

from __future__ import annotations


import json

import pytest
from pydantic import BaseModel, Field, ValidationError

from instructor.cache import AutoCache, load_cached_response, store_cached_response


class Address(BaseModel):
    postal_code: str = Field(alias="postalCode")


class Person(BaseModel):
    user_id: int = Field(alias="userId")
    full_name: str = Field(alias="fullName")
    address: Address


def person() -> Person:
    return Person(userId=7, fullName="Ada", address=Address(postalCode="EC1A"))


def test_aliased_model_round_trips_through_cache() -> None:
    cache = AutoCache()
    store_cached_response(cache, "person", person())
    restored = load_cached_response(cache, "person", Person)
    assert restored.user_id == 7
    assert restored.full_name == "Ada"
    assert restored.address.postal_code == "EC1A"


def test_aliased_model_round_trips_with_context_and_strict() -> None:
    cache = AutoCache()
    store_cached_response(cache, "person", person())
    restored = load_cached_response(
        cache, "person", Person, context={"unused": True}, strict=True
    )
    assert restored == person()


def test_alias_keyed_entries_still_load() -> None:
    """Accepting field names must not stop aliases from validating."""
    cache = AutoCache()
    cache.set(
        "alias-keyed",
        '{"userId": 7, "fullName": "Ada", "address": {"postalCode": "EC1A"}}',
    )
    assert load_cached_response(cache, "alias-keyed", Person) == person()


def test_separate_validation_and_serialization_aliases_round_trip():
    from pydantic import AliasPath, ValidationError

    class Child(BaseModel):
        code: int = Field(
            validation_alias=AliasPath("ids", 0), serialization_alias="outputCode"
        )

    class Envelope(BaseModel):
        children: list[Child] = Field(alias="items")

    model = Envelope.model_validate({"items": [{"ids": [9]}]})
    cache = AutoCache()
    store_cached_response(cache, "entry", model)
    assert load_cached_response(cache, "entry", Envelope, strict=True) == model
    cache.set("entry", '{"children":[{"code":"9"}]}')
    with pytest.raises(ValidationError):
        load_cached_response(cache, "entry", Envelope, strict=True)
    assert load_cached_response(cache, "entry", Envelope, strict=False) == model


def test_schema_fallback_preserves_field_names_and_original_configuration(monkeypatch):
    from instructor.v2.core import cache_response

    class OldEntry(BaseModel):
        code: int = Field(alias="wireCode")
        values: list[int] = Field(default_factory=list)

    monkeypatch.setattr(cache_response, "_SUPPORTS_BY_NAME", False)
    cache = AutoCache()
    cache.set("legacy", '{"code":1,"values":[1,2]}')
    assert load_cached_response(cache, "legacy", OldEntry) == OldEntry(
        wireCode=1, values=[1, 2]
    )
    with pytest.raises(ValidationError):
        OldEntry.model_validate({"code": 1})


def test_cache_keeps_overlapping_field_names_distinct() -> None:
    class Pair(BaseModel):
        a: int = Field(validation_alias="b")
        b: int = Field(validation_alias="c")

    class Nested(BaseModel):
        pair: Pair

    original = Nested.model_validate({"pair": {"b": 1, "c": 2}})
    cache = AutoCache()
    store_cached_response(cache, "pair", original)
    assert load_cached_response(cache, "pair", Nested, strict=True) == original

    # Previous Instructor versions wrote the same field-name JSON without a tag.
    cache.set("pair", json.dumps({"model": original.model_dump_json(), "raw": None}))
    assert load_cached_response(cache, "pair", Nested, strict=True) == original

    # Raw, alias-keyed records use the original wire representation.
    cache.set("pair", '{"pair":{"b":1,"c":2}}')
    assert load_cached_response(cache, "pair", Nested, strict=True) == original
