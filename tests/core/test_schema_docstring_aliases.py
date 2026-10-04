"""Offline contracts for docstring descriptions on aliased schema properties."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import (
    AliasChoices,
    AliasPath,
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    create_model,
)
from pydantic.json_schema import SkipJsonSchema

from instructor.processing.schema import generate_openai_schema
from instructor.v2.providers.openai.handlers import OpenAIToolsHandler


@pytest.mark.parametrize(
    ("field_kwargs", "config", "property_name"),
    [
        ({}, {}, "value"),
        ({"alias": "wire"}, {}, "wire"),
        ({"alias": "wire", "validation_alias": "input"}, {}, "input"),
        ({"serialization_alias": "output"}, {}, "value"),
        ({}, {"alias_generator": str.upper}, "VALUE"),
        ({"validation_alias": AliasChoices("first", "second")}, {}, "first"),
        (
            {"validation_alias": AliasChoices(AliasPath("data", "value"), "flat")},
            {},
            "flat",
        ),
        ({"validation_alias": AliasPath("data", "value")}, {}, "value"),
        ({"validation_alias": AliasPath("single")}, {}, "value"),
        (
            {"validation_alias": AliasChoices(AliasPath("single"), "second")},
            {},
            "single",
        ),
        ({"alias": ""}, {}, ""),
        (
            {"validation_alias": "input", "serialization_alias": "output"},
            {"json_schema_mode_override": "serialization"},
            "output",
        ),
    ],
    ids=[
        "no-alias",
        "alias",
        "validation-alias",
        "serialization-alias-ignored",
        "alias-generator",
        "alias-choices",
        "alias-choices-after-path",
        "nested-alias-path",
        "single-alias-path",
        "alias-choice-single-path",
        "empty-alias",
        "serialization-mode-override",
    ],
)
def test_docstring_follows_schema_alias(
    field_kwargs: dict[str, Any], config: dict[str, Any], property_name: str
) -> None:
    class Record(BaseModel):
        """A record.

        Args:
            value: Description for the Python field.
        """

        model_config = ConfigDict(**config)
        value: str = Field(**field_kwargs)

    original = Record.model_json_schema()
    schema = generate_openai_schema(Record)
    properties = schema["parameters"]["properties"]
    assert set(properties) == {property_name}
    assert (
        properties[property_name]["description"] == "Description for the Python field."
    )
    assert schema["parameters"]["required"] == original["required"]
    assert Record.model_json_schema() == original


def test_alias_collision_describes_the_correct_field_in_tools_request() -> None:
    class Record(BaseModel):
        """A record.

        Args:
            count: Number of items.
            label: Human-readable label.
        """

        count: int = Field(alias="label")
        label: str = Field(alias="name")

    prepared, kwargs = OpenAIToolsHandler().prepare_request(Record, {"messages": []})
    properties = kwargs["tools"][0]["function"]["parameters"]["properties"]
    assert properties["label"]["description"] == "Number of items."
    assert properties["name"]["description"] == "Human-readable label."
    parsed = prepared.model_validate_json('{"label": 3, "name": "books"}')
    assert parsed.count == 3
    assert parsed.label == "books"


def test_explicit_description_keeps_precedence() -> None:
    class Record(BaseModel):
        """A record.

        Args:
            value: Docstring description.
        """

        value: str = Field(alias="wire", description="Explicit description.")

    properties = generate_openai_schema(Record)["parameters"]["properties"]
    assert properties["wire"]["description"] == "Explicit description."


def test_docstrings_using_wire_names_remain_supported() -> None:
    class Record(BaseModel):
        """A record.

        Args:
            wire: Wire-name description.
        """

        value: str = Field(alias="wire")

    properties = generate_openai_schema(Record)["parameters"]["properties"]
    assert properties["wire"]["description"] == "Wire-name description."


def test_omitted_field_does_not_describe_a_different_property() -> None:
    class Record(BaseModel):
        """A record.

        Args:
            private: Private field description.
            value: Public field description.
        """

        private: SkipJsonSchema[str] = Field(default="", alias="hidden")
        value: str = Field(alias="private")

    properties = generate_openai_schema(Record)["parameters"]["properties"]
    assert set(properties) == {"private"}
    assert properties["private"]["description"] == "Public field description."


@pytest.mark.parametrize("omission", ["none", "skip-schema", "exclude-serialization"])
def test_shared_alias_does_not_guess_which_field_to_describe(omission: str) -> None:
    record = create_model(
        "Record",
        __doc__="""A record.

        Args:
            hidden: Description for the first field.
            visible: Description for the second field.
        """,
        __config__=ConfigDict(
            json_schema_mode_override=(
                "serialization" if omission == "exclude-serialization" else "validation"
            )
        ),
        hidden=(
            SkipJsonSchema[str] if omission == "skip-schema" else str,
            Field(
                default="", alias="wire", exclude=omission == "exclude-serialization"
            ),
        ),
        visible=(int, Field(alias="wire")),
    )

    original = record.model_json_schema()["properties"]["wire"]
    properties = generate_openai_schema(record)["parameters"]["properties"]
    assert properties["wire"] == original


def test_computed_field_alias_collision_is_not_enriched() -> None:
    class Record(BaseModel):
        """A record.

        Args:
            value: Input value description.
        """

        model_config = ConfigDict(json_schema_mode_override="serialization")
        value: int = Field(serialization_alias="wire")

        @computed_field(alias="wire")
        def doubled(self) -> int:
            return self.value * 2

    original = Record.model_json_schema()["properties"]["wire"]
    properties = generate_openai_schema(Record)["parameters"]["properties"]
    assert properties["wire"] == original


def test_computed_field_docstring_uses_its_own_alias() -> None:
    class Record(BaseModel):
        """A record.

        Args:
            value: Input value description.
            doubled: Computed value description.
        """

        model_config = ConfigDict(json_schema_mode_override="serialization")
        value: int = Field(serialization_alias="doubled")

        @computed_field(alias="")
        def doubled(self) -> int:
            return self.value * 2

    properties = generate_openai_schema(Record)["parameters"]["properties"]
    assert properties["doubled"]["description"] == "Input value description."
    assert properties[""]["description"] == "Computed value description."
