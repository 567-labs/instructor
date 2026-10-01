import pytest
from pydantic import BaseModel, ConfigDict, Json, computed_field

from instructor.cache import AutoCache, load_cached_response, store_cached_response


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
