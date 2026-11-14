import sys
from typing import Any, cast

import pytest
from pydantic import GetCoreSchemaHandler, ValidationError
from pydantic_core import CoreSchema, core_schema
from typing_extensions import TypedDict

from instructor.v2.core.function_calls import ResponseSchema
from instructor.v2.core.response_model import prepare_response_model


class Child(TypedDict):
    count: int


class Replacement(TypedDict):
    count: str


class Record(TypedDict):
    children: list["Child"]
    name: str


def test_typed_dict_preparation_uses_current_annotations_and_fresh_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = cast(type[ResponseSchema], prepare_response_model(Record))
    original.model_fields["name"].description = "Local change"
    original.model_rebuild(force=True)

    monkeypatch.setattr(sys.modules[__name__], "Child", Replacement)
    refreshed = cast(type[ResponseSchema], prepare_response_model(Record))

    assert refreshed is not original
    assert refreshed.model_fields["name"].description is None
    assert refreshed.openai_schema["name"] == "Record"
    payload = {"children": [{"count": "2"}], "name": "Ada"}
    assert original.model_validate(payload).model_dump(round_trip=True) == {
        "children": [{"count": 2}],
        "name": "Ada",
    }
    assert refreshed.model_validate(payload).model_dump(round_trip=True) == payload


def test_custom_annotation_keeps_both_schema_builds() -> None:
    class StatefulType:
        builds = 0

        @classmethod
        def __get_pydantic_core_schema__(
            cls, _source: Any, _handler: GetCoreSchemaHandler
        ) -> CoreSchema:
            cls.builds += 1
            return core_schema.int_schema(ge=cls.builds)

    class CustomRecord(TypedDict):
        count: StatefulType

    model = cast(type[ResponseSchema], prepare_response_model(CustomRecord))

    assert StatefulType.builds == 2
    assert model.model_json_schema()["properties"]["count"]["minimum"] == 2
    assert model.model_validate({"count": 2}).model_dump() == {"count": 2}
    with pytest.raises(ValidationError):
        model.model_validate({"count": 1})
