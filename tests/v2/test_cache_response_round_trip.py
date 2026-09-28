"""Cached responses must reload into the model that produced them."""

from __future__ import annotations

import inspect

import pytest
from pydantic import BaseModel, Field

from instructor.cache import AutoCache, load_cached_response, store_cached_response

requires_by_name = pytest.mark.skipif(
    "by_name" not in inspect.signature(BaseModel.model_validate_json).parameters,
    reason="Pydantic < 2.11 cannot validate by field name per call",
)


class Address(BaseModel):
    postal_code: str = Field(alias="postalCode")


class Person(BaseModel):
    user_id: int = Field(alias="userId")
    full_name: str = Field(alias="fullName")
    address: Address


def person() -> Person:
    return Person(userId=7, fullName="Ada", address=Address(postalCode="EC1A"))


@requires_by_name
def test_aliased_model_round_trips_through_cache() -> None:
    cache = AutoCache()
    store_cached_response(cache, "person", person())
    restored = load_cached_response(cache, "person", Person)
    assert restored.user_id == 7
    assert restored.full_name == "Ada"
    assert restored.address.postal_code == "EC1A"


@requires_by_name
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
