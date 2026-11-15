"""Cached response envelope: round trips, legacy entries, raw responses, errors.

Current entries are `{"model": <model JSON>, "raw": <raw JSON | null>}`.
Older Instructor versions wrote bare model JSON, envelopes without `raw`, or
alias-keyed records. `load_cached_response` recognizes the envelope by shape
(issue #2830), so business fields named `model`/`raw` still load as legacy.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import (
    AliasPath,
    BaseModel,
    ConfigDict,
    Field,
    Json,
    RootModel,
    ValidationError,
    computed_field,
    field_validator,
)
from typing_extensions import NotRequired, TypedDict

from instructor.cache import AutoCache, load_cached_response, store_cached_response
from instructor.processing.response import handle_response_model


class JsonAnswer(BaseModel):
    values: Json[list[int]]


class NestedAnswer(BaseModel):
    answers: list[JsonAnswer]


class ComputedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: int

    @computed_field
    @property
    def doubled(self) -> int:
        return self.value * 2


class Address(BaseModel):
    postal_code: str = Field(alias="postalCode")


class Person(BaseModel):
    user_id: int = Field(alias="userId")
    full_name: str = Field(alias="fullName")
    address: Address


class Child(BaseModel):
    code: int = Field(
        validation_alias=AliasPath("ids", 0), serialization_alias="outputCode"
    )


class Children(BaseModel):
    children: list[Child] = Field(alias="items")


class Pair(BaseModel):
    a: int = Field(validation_alias="b")
    b: int = Field(validation_alias="c")


class NestedPair(BaseModel):
    pair: Pair


class Device(BaseModel):
    model: str
    year: int


class Solo(BaseModel):
    model: str


class Widget(BaseModel):
    model: str
    raw: str


class Restricted(BaseModel):
    model: str

    @field_validator("model")
    @classmethod
    def check_model(cls, value: str) -> str:
        if value == "blocked":
            raise ValueError("blocked model")
        return value


def _legacy_typed_dict_list() -> BaseModel:
    class Record(TypedDict):
        _id: str

    model, _ = handle_response_model(list[Record])
    return cast(type[BaseModel], model).model_validate({"tasks": [{"_id": "doc"}]})


PERSON = Person(userId=7, fullName="Ada", address=Address(postalCode="EC1A"))
CHILDREN = Children.model_validate({"items": [{"ids": [9]}]})
PAIR = NestedPair.model_validate({"pair": {"b": 1, "c": 2}})
DEVICE = Device(model="ThinkPad", year=2026)
TYPED_DICT_LIST = _legacy_typed_dict_list()

ROUND_TRIPS = {
    "json-field": JsonAnswer.model_validate({"values": "[1, 2, 3]"}),
    "nested-json-field": NestedAnswer.model_validate(
        {"answers": [{"values": "[1, 2, 3]"}]}
    ),
    "computed-field-forbid-extra": ComputedAnswer(value=3),
    "aliases": PERSON,
    "separate-validation-serialization-aliases": CHILDREN,
    "overlapping-field-aliases": PAIR,
    "business-model-field": DEVICE,
    "root-int": RootModel[int](123),
    "root-list": RootModel[list[int]]([1, 2]),
}


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("original", ROUND_TRIPS.values(), ids=ROUND_TRIPS.keys())
def test_stored_response_round_trips(original: BaseModel, strict: bool) -> None:
    cache = AutoCache()
    store_cached_response(cache, "entry", original)

    restored = load_cached_response(
        cache, "entry", type(original), context={"unused": True}, strict=strict
    )

    assert type(restored) is type(original)
    assert restored == original
    assert restored.model_dump() == original.model_dump()


def _envelope(model: BaseModel, **extra: Any) -> str:
    return json.dumps({"model": model.model_dump_json(round_trip=True), **extra})


# (model type, stored payload, strict, expected model or first validation error)
STORED: dict[str, tuple[Any, Any, bool, Any]] = {
    "legacy-model-field": (Device, DEVICE.model_dump_json(), False, DEVICE),
    "legacy-model-field-strict": (Device, DEVICE.model_dump_json(), True, DEVICE),
    **{
        f"legacy-only-model-field-{value}": (
            Solo,
            Solo(model=value).model_dump_json(),
            False,
            Solo(model=value),
        )
        for value in ["ThinkPad", "123", "null", '{"sku": 1}']
    },
    "legacy-model-and-raw-fields": (
        Widget,
        Widget(model="abc", raw="def").model_dump_json(),
        False,
        Widget(model="abc", raw="def"),
    ),
    "envelope-without-raw": (Device, _envelope(DEVICE), False, DEVICE),
    "untagged-field-name-envelope": (
        NestedPair,
        json.dumps({"model": PAIR.model_dump_json(), "raw": None}),
        True,
        PAIR,
    ),
    "legacy-typed-dict-envelope": (
        type(TYPED_DICT_LIST),
        _envelope(TYPED_DICT_LIST, raw=None),
        False,
        TYPED_DICT_LIST,
    ),
    "alias-keyed-record": (
        Person,
        '{"userId": 7, "fullName": "Ada", "address": {"postalCode": "EC1A"}}',
        False,
        PERSON,
    ),
    "alias-keyed-overlapping-record": (
        NestedPair,
        '{"pair":{"b":1,"c":2}}',
        True,
        PAIR,
    ),
    "field-name-record-lax": (
        Children,
        '{"children":[{"code":"9"}]}',
        False,
        CHILDREN,
    ),
    "field-name-record-strict": (
        Children,
        '{"children":[{"code":"9"}]}',
        True,
        {"type": "int_type", "loc": ("children", 0, "code")},
    ),
    "rejected-envelope-is-not-business-fields": (
        Restricted,
        json.dumps({"model": '{"model":"blocked"}', "raw": None}),
        False,
        {"type": "value_error", "loc": ("model",)},
    ),
    "invalid-document": (Device, "incomplete{", False, {"type": "json_invalid"}),
    "invalid-inner-and-outer-reports-inner": (
        Device,
        json.dumps({"model": '{"year":2026}', "raw": None}),
        False,
        {"type": "missing", "loc": ("model",)},
    ),
}


@pytest.mark.parametrize(
    ("model", "payload", "strict", "expected"), STORED.values(), ids=STORED.keys()
)
def test_stored_payload_loads(
    model: type[BaseModel], payload: str, strict: bool, expected: Any
) -> None:
    cache = AutoCache()
    cache.set("entry", payload)

    if isinstance(expected, dict):
        with pytest.raises(ValidationError) as caught:
            load_cached_response(cache, "entry", model, strict=strict)
        errors = caught.value.errors()
        assert [{key: error[key] for key in expected} for error in errors] == [expected]
        return
    restored = load_cached_response(cache, "entry", model, strict=strict)
    assert restored == expected
    assert not hasattr(restored, "_raw_response")


class RawCompletion(BaseModel):
    id: str
    choices: list[dict[str, Any]]


def _recursive() -> list[Any]:
    raw: list[Any] = []
    raw.append(raw)
    return raw


RAW_CASES = {
    # (raw response, stored "raw", restored _raw_response). Completion-like
    # mappings regain attribute access; unserializable raws fall back to str.
    "mapping": (
        {"id": "resp-1"},
        {"id": "resp-1"},
        SimpleNamespace(id="resp-1"),
    ),
    "pydantic": (
        RawCompletion(id="c-1", choices=[{"message": {"content": "ok"}}]),
        {"id": "c-1", "choices": [{"message": {"content": "ok"}}]},
        SimpleNamespace(
            id="c-1", choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        ),
    ),
    "unserializable": (_recursive(), "[[...]]", "[[...]]"),
}


@pytest.mark.parametrize(
    ("raw", "stored", "restored_raw"), RAW_CASES.values(), ids=RAW_CASES.keys()
)
def test_raw_response_is_stored_and_restored(
    raw: Any, stored: Any, restored_raw: Any, caplog: pytest.LogCaptureFixture
) -> None:
    cache = AutoCache()
    original = Device(model="ThinkPad", year=2026)
    object.__setattr__(original, "_raw_response", raw)

    store_cached_response(cache, "entry", original, ttl=10)
    restored = load_cached_response(cache, "entry", Device)

    stored_raw = json.loads(cast(str, cache.get("entry")))["raw"]
    assert (stored_raw if isinstance(stored, str) else json.loads(stored_raw)) == stored
    assert restored == original
    assert restored._raw_response == restored_raw
    if isinstance(stored, str):
        assert "Raw response could not be serialized as JSON" in caplog.text


class TtlCache(AutoCache):
    def __init__(self) -> None:
        super().__init__()
        self.ttls: list[int | None] = []

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        self.ttls.append(ttl)
        super().set(key, value, ttl)


def test_response_without_raw_stores_null_raw_with_ttl() -> None:
    cache = TtlCache()
    store_cached_response(cache, "entry", DEVICE, ttl=15)

    assert cache.ttls == [15]
    assert json.loads(cast(str, cache.get("entry")))["raw"] is None
    restored = load_cached_response(cache, "entry", Device)
    assert restored == DEVICE
    assert not hasattr(restored, "_raw_response")


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("optional_present", [False, True])
def test_cached_typed_dict_preserves_underscore_keys_and_omission(
    nested: bool, optional_present: bool
) -> None:
    class Record(TypedDict):
        _id: str
        _note: NotRequired[str]

    prepared, _ = handle_response_model(list[Record] if nested else Record)
    model = cast(type[BaseModel], prepared)
    payload = {"_id": "document-1"}
    if optional_present:
        payload["_note"] = "saved"
    data = {"tasks": [payload]} if nested else payload
    original = model.model_validate(data)
    cache = AutoCache()

    store_cached_response(cache, "record", original)
    restored = load_cached_response(cache, "record", model)

    assert restored == original
    assert restored.model_dump(by_alias=True, exclude_unset=True) == data
