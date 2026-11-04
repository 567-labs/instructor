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
    original = model.model_validate(data)
    cache = AutoCache()

    store_cached_response(cache, "record", original)
    restored = load_cached_response(cache, "record", model)

    assert restored == original
    assert restored.model_dump(by_alias=True, exclude_unset=True) == data
