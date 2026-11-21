import json

from typing import cast

import pytest
from pydantic import BaseModel, ConfigDict, Json, computed_field
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


class PlainAnswer(BaseModel):
    value: int


class DefaultedAnswer(BaseModel):
    title: str
    summary: str | None = None
    word_count: int = 0


class NestedAuthor(BaseModel):
    name: str
    country: str = "unknown"


class NestedDefaultedAnswer(BaseModel):
    title: str
    author: NestedAuthor
    related: list[NestedAuthor] = []


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    "original",
    [
        pytest.param(
            JsonAnswer.model_validate({"values": "[1, 2, 3]"}), id="json-field"
        ),
        pytest.param(
            NestedAnswer.model_validate({"answers": [{"values": "[1, 2, 3]"}]}),
            id="nested-json-field",
        ),
        pytest.param(ComputedAnswer(value=3), id="computed-field-forbid-extra"),
        pytest.param(PlainAnswer(value=3), id="plain-model"),
    ],
)
def test_cached_response_round_trips(original: BaseModel, strict: bool) -> None:
    cache = AutoCache()

    store_cached_response(cache, "answer", original)
    restored = load_cached_response(cache, "answer", type(original), strict=strict)

    assert type(restored) is type(original)
    assert restored == original
    assert restored.model_dump() == original.model_dump()


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("optional_present", [False, True])
def test_cached_typed_dict_preserves_underscore_keys_and_omission(
    nested: bool, optional_present: bool
) -> None:
    class Record(TypedDict):
        _id: str
        _note: NotRequired[str]

    response_type = list[Record] if nested else Record
    model, _ = handle_response_model(response_type)
    assert isinstance(model, type) and issubclass(model, BaseModel)
    payload = {"_id": "document-1"}
    if optional_present:
        payload["_note"] = "saved"
    data = {"tasks": [payload]} if nested else payload
    original = cast(type[BaseModel], model).model_validate(data)
    cache = AutoCache()

    store_cached_response(cache, "record", original)
    restored = load_cached_response(cache, "record", model)

    assert restored == original
    assert restored.model_dump(by_alias=True, exclude_unset=True) == data


def test_cached_response_keeps_explicit_fields_and_omits_defaults() -> None:
    original = DefaultedAnswer.model_validate({"title": "Hello"})
    assert original.model_dump(exclude_unset=True) == {"title": "Hello"}

    cache = AutoCache()
    store_cached_response(cache, "defaulted", original)
    restored = load_cached_response(cache, "defaulted", DefaultedAnswer)

    assert restored == original
    assert restored.model_fields_set == {"title"}
    assert restored.model_dump(exclude_unset=True) == {"title": "Hello"}


def test_cached_response_keeps_explicit_fields_of_nested_models() -> None:
    original = NestedDefaultedAnswer.model_validate(
        {"title": "Hello", "author": {"name": "Ada"}, "related": [{"name": "Bob"}]}
    )
    expected = {
        "title": "Hello",
        "author": {"name": "Ada"},
        "related": [{"name": "Bob"}],
    }
    assert original.model_dump(exclude_unset=True) == expected

    cache = AutoCache()
    store_cached_response(cache, "nested-defaulted", original)
    restored = load_cached_response(cache, "nested-defaulted", NestedDefaultedAnswer)

    assert restored == original
    assert restored.author.model_fields_set == {"name"}
    assert restored.related[0].model_fields_set == {"name"}
    assert restored.model_dump(exclude_unset=True) == expected


def test_tagged_cache_entry_without_field_set_still_loads() -> None:
    original = DefaultedAnswer(title="Hello", summary="s")
    cache = AutoCache()
    cache.set(
        "tagged-without-set",
        json.dumps(
            {
                "$instructor_cache": "field-names-v2",
                "model": original.model_dump_json(round_trip=True, by_alias=False),
                "raw": None,
            }
        ),
    )

    restored = load_cached_response(cache, "tagged-without-set", DefaultedAnswer)

    assert restored == original


def test_legacy_nested_typed_dict_envelope_retains_wire_keys() -> None:
    class Record(TypedDict):
        _id: str

    model, _ = handle_response_model(list[Record])
    response_type = cast(type[BaseModel], model)
    original = response_type.model_validate({"tasks": [{"_id": "document-1"}]})
    cache = AutoCache()
    cache.set(
        "prior-typed-dict",
        json.dumps({"model": original.model_dump_json(round_trip=True), "raw": None}),
    )
    assert load_cached_response(cache, "prior-typed-dict", response_type) == original
