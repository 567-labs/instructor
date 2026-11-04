"""Legacy cache entries whose business fields collide with envelope keys.

Regression tests for https://github.com/567-labs/instructor/issues/2830:
load_cached_response must recognize the response envelope by shape (all keys in
{"model", "raw"} and "model" holding a JSON document), not merely by the
presence of a "model" key, so legacy entries whose models have business fields
named "model"/"raw" keep restoring as legacy entries.
"""

import json

import pytest
from pydantic import BaseModel, RootModel, ValidationError, field_validator

from instructor.cache import AutoCache, load_cached_response, store_cached_response


class Device(BaseModel):
    model: str
    year: int


class Solo(BaseModel):
    model: str


class Widget(BaseModel):
    model: str
    raw: str


@pytest.mark.parametrize("strict", [False, True])
def test_legacy_entry_with_model_field_restores(strict: bool) -> None:
    """The exact report from #2830: legacy entry must not be misread."""
    cache = AutoCache()
    original = Device(model="ThinkPad", year=2026)
    cache.set("device", original.model_dump_json())

    restored = load_cached_response(cache, "device", Device, strict=strict)

    assert restored == original
    assert restored.model_dump() == original.model_dump()


@pytest.mark.parametrize("value", ["ThinkPad", "123", "null", '{"sku": 1}'])
def test_legacy_entry_with_only_model_field_restores(value: str) -> None:
    cache = AutoCache()
    original = Solo(model=value)
    cache.set("solo", original.model_dump_json())

    restored = load_cached_response(cache, "solo", Solo)

    assert restored == original


def test_legacy_entry_with_model_and_raw_fields_restores() -> None:
    """Legacy business fields named both "model" and "raw" must not trigger
    envelope handling when "model" is not a JSON document."""
    cache = AutoCache()
    original = Widget(model="abc", raw="def")
    cache.set("widget", original.model_dump_json())

    restored = load_cached_response(cache, "widget", Widget)

    assert restored == original


def test_envelope_without_raw_key_restores() -> None:
    """Older envelope versions may omit "raw"; they must still restore."""
    cache = AutoCache()
    original = Device(model="ThinkPad", year=2026)
    envelope = {"model": original.model_dump_json()}
    cache.set("legacy-envelope", json.dumps(envelope))

    restored = load_cached_response(cache, "legacy-envelope", Device)

    assert restored == original
    assert not hasattr(restored, "_raw_response")


def test_envelope_with_raw_restores_raw_response() -> None:
    """The raw side of a current envelope keeps its restore behavior."""
    from types import SimpleNamespace

    cache = AutoCache()
    original = Device(model="ThinkPad", year=2026)
    object.__setattr__(original, "_raw_response", {"id": "resp-1"})
    store_cached_response(cache, "device", original)

    restored = load_cached_response(cache, "device", Device)

    assert restored == original
    # Completion-like raw mappings regain attribute access (existing behavior).
    assert restored._raw_response == SimpleNamespace(id="resp-1")


@pytest.mark.parametrize(
    "original", [RootModel[int](123), RootModel[list[int]]([1, 2])]
)
def test_root_model_envelope_round_trip(original: BaseModel) -> None:
    cache = AutoCache()
    store_cached_response(cache, "root", original)
    assert load_cached_response(cache, "root", type(original), strict=True) == original


def test_rejected_envelope_cannot_be_reinterpreted_as_business_fields() -> None:
    class Restricted(BaseModel):
        model: str

        @field_validator("model")
        @classmethod
        def check_model(cls, value: str) -> str:
            if value == "blocked":
                raise ValueError("blocked model")
            return value

    cache = AutoCache()
    cache.set("restricted", json.dumps({"model": '{"model":"blocked"}', "raw": None}))
    with pytest.raises(ValidationError, match="blocked model"):
        load_cached_response(cache, "restricted", Restricted)


def test_invalid_cache_document_reports_json_error() -> None:
    cache = AutoCache()
    cache.set("broken", "incomplete{")
    with pytest.raises(ValidationError) as caught:
        load_cached_response(cache, "broken", Device)
    assert caught.value.errors()[0]["type"] == "json_invalid"


def test_invalid_inner_and_outer_documents_report_inner_error() -> None:
    cache = AutoCache()
    cache.set("broken", json.dumps({"model": '{"year":2026}', "raw": None}))
    with pytest.raises(ValidationError) as caught:
        load_cached_response(cache, "broken", Device)
    assert [error["loc"] for error in caught.value.errors()] == [("model",)]
