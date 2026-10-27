"""Legacy cache entries whose business fields mimic the envelope must still load."""

import pytest
from pydantic import BaseModel

from instructor.cache import load_cached_response


class Gadget(BaseModel):
    model: str
    price: int


class Pair(BaseModel):
    model: str
    raw: str


class FakeCache:
    """Minimal in-memory stand-in for a single cached key."""

    def __init__(self, value: str | None):
        self.value = value

    def get(self, key: str) -> str | None:
        return self.value


def test_legacy_entry_with_a_business_model_field_loads() -> None:
    # A model whose business field is named "model" used to be misread as the
    # response envelope, selecting "ThinkPad" as the model JSON.
    legacy = Gadget(model="ThinkPad", price=3).model_dump_json()

    obj = load_cached_response(FakeCache(legacy), "key", Gadget)

    assert obj == Gadget(model="ThinkPad", price=3)


def test_envelope_entries_still_load() -> None:
    import json

    from instructor.cache import store_cached_response

    cache = FakeCache(None)
    cache.set = lambda key, value, ttl=None: setattr(cache, "value", value)
    gadget = Gadget(model="X1", price=5)
    store_cached_response(cache, "key", gadget)

    obj = load_cached_response(cache, "key", Gadget)

    assert obj == gadget


def test_envelope_with_raw_response_still_restores_raw() -> None:
    import json

    payload = json.dumps(
        {
            "model": Gadget(model="X2", price=7).model_dump_json(),
            "raw": json.dumps(
                {"id": "resp-1", "object": "response", "model": "gpt", "choices": []}
            ),
        }
    )

    obj = load_cached_response(FakeCache(payload), "key", Gadget)

    assert obj == Gadget(model="X2", price=7)
    assert getattr(obj, "_raw_response", None) is not None


def test_legacy_entry_with_both_model_and_raw_fields_loads() -> None:
    # A legacy model can carry business fields named model and raw. Its model
    # value is not an object-shaped JSON string, so it is not mistaken for the
    # envelope and loads through the legacy path.
    pathological = Pair(model="m-1", raw="r-1").model_dump_json()

    obj = load_cached_response(FakeCache(pathological), "key", Pair)

    assert obj == Pair(model="m-1", raw="r-1")
