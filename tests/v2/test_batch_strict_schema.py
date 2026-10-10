"""Offline contracts for strict OpenAI batch request schemas."""

from __future__ import annotations

from copy import deepcopy
import io
import json
from typing import Any, ClassVar

import pytest
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel, Field

from instructor.batch import BatchRequest

pytestmark = pytest.mark.unit


class Child(BaseModel):
    label: str = Field(default="child", description="Child label")
    tags: list[str] = Field(default_factory=list)


class Alternative(BaseModel):
    count: int = 0


class Document(BaseModel):
    title: str = Field(default="untitled", description="Document title")
    child: Child | None = None
    entries: list[Child | Alternative] = Field(default_factory=list)


def request_for(model: type[BaseModel]) -> BatchRequest[Any]:
    return BatchRequest(
        custom_id="strict-schema",
        messages=[{"role": "user", "content": "Extract the document."}],
        response_model=model,
        model="offline-model",
    )


def schema_for(model: type[BaseModel]) -> dict[str, Any]:
    return request_for(model).to_openai_format()["body"]["response_format"][
        "json_schema"
    ]["schema"]


def test_defaulted_and_nullable_fields_match_sdk_strict_schema() -> None:
    source = Document.model_json_schema()
    schema = schema_for(Document)

    assert schema == to_strict_json_schema(Document)
    assert schema["required"] == ["title", "child", "entries"]
    assert schema["$defs"]["Child"]["required"] == ["label", "tags"]
    assert schema["$defs"]["Alternative"]["required"] == ["count"]
    assert schema["properties"]["child"]["anyOf"][1] == {"type": "null"}
    assert schema["properties"]["title"]["description"] == "Document title"
    assert schema["properties"]["title"]["default"] == "untitled"
    assert Document.model_json_schema() == source


class SharedSchemaModel(BaseModel):
    schema_source: ClassVar[dict[str, Any]] = {}

    @classmethod
    def model_json_schema(cls, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return cls.schema_source


def custom_model(schema: dict[str, Any]) -> type[SharedSchemaModel]:
    class CustomModel(SharedSchemaModel):
        schema_source: ClassVar[dict[str, Any]] = schema

    return CustomModel


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf", "allOf"])
@pytest.mark.parametrize("definitions_key", ["$defs", "definitions"])
def test_recursive_custom_schema_preserves_booleans_metadata_and_source(
    keyword: str, definitions_key: str
) -> None:
    nested = {
        "type": "object",
        "title": "Nested",
        "description": "Preserve this description",
        "properties": {"value": {"type": "integer", "default": 7}},
        "examples": [{"value": 7}],
    }
    source = {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "forbidden": False,
            "unconstrained": True,
            "choice": {keyword: [deepcopy(nested), {"type": "null"}]},
            "array": {"type": "array", "items": deepcopy(nested)},
        },
        definitions_key: {"Nested": deepcopy(nested)},
        "examples": [{"type": "object", "additionalProperties": True}],
    }
    original = deepcopy(source)
    model = custom_model(source)
    schema = schema_for(model)

    assert schema["required"] == list(source["properties"])
    assert schema["additionalProperties"] is False
    assert schema["properties"]["forbidden"] is False
    assert schema["properties"]["unconstrained"] is True
    for strict_nested in (
        schema["properties"]["choice"][keyword][0],
        schema["properties"]["array"]["items"],
        schema[definitions_key]["Nested"],
    ):
        assert strict_nested["additionalProperties"] is False
        assert strict_nested["required"] == ["value"]
        assert strict_nested["title"] == nested["title"]
        assert strict_nested["description"] == nested["description"]
        assert strict_nested["examples"] == nested["examples"]
        assert strict_nested["properties"]["value"]["default"] == 7
    assert schema["examples"] == original["examples"]
    assert source == original
    schema["properties"]["array"]["items"]["title"] = "Changed"
    assert source == original
    assert schema_for(model)["properties"]["array"]["items"]["title"] == "Nested"


@pytest.mark.parametrize("mapping_type", [dict[str, int], dict[str, Any]])
def test_strict_mapping_fields_are_rejected_without_losing_value_schema(
    mapping_type: Any,
) -> None:
    class MappingDocument(BaseModel):
        values: mapping_type

    original = MappingDocument.model_json_schema()
    with pytest.raises(ValueError, match="mapping.*OpenAI.*strict"):
        schema_for(MappingDocument)
    assert MappingDocument.model_json_schema() == original


def test_schema_valued_additional_properties_with_named_fields_is_rejected() -> None:
    source = {
        "type": "object",
        "properties": {"known": {"type": "string"}},
        "additionalProperties": {"type": "integer"},
    }
    original = deepcopy(source)
    with pytest.raises(ValueError, match="mapping.*OpenAI.*strict"):
        schema_for(custom_model(source))
    assert source == original


def test_anthropic_mapping_schema_is_unchanged() -> None:
    class MappingDocument(BaseModel):
        values: dict[str, int]

    request = request_for(MappingDocument)
    schema = request.to_anthropic_format()["params"]["tools"][0]["input_schema"]
    expected = MappingDocument.model_json_schema()
    expected["additionalProperties"] = False
    assert schema == expected
    assert schema["properties"]["values"]["additionalProperties"] == {"type": "integer"}


def test_aliases_and_ref_sibling_metadata_match_sdk() -> None:
    class AliasedDocument(BaseModel):
        label: str = Field(default="untitled", alias="documentLabel")
        child: Child = Field(description="Described child", title="Child field")

    original = AliasedDocument.model_json_schema()
    child_schema = original["properties"]["child"]
    assert "$ref" in child_schema or "$ref" in child_schema["allOf"][0]
    schema = schema_for(AliasedDocument)

    assert schema == to_strict_json_schema(AliasedDocument)
    assert schema["required"] == ["documentLabel", "child"]
    assert schema["properties"]["child"]["description"] == "Described child"
    assert schema["properties"]["child"]["title"] == "Child field"
    assert schema["properties"]["child"]["required"] == ["label", "tags"]
    assert "$ref" not in schema["properties"]["child"]
    assert AliasedDocument.model_json_schema() == original


def test_single_all_of_ref_matches_sdk_and_preserves_wrapper_metadata() -> None:
    source = {
        "type": "object",
        "properties": {
            "child": {
                "allOf": [{"$ref": "#/$defs/Child"}],
                "description": "Described child",
            }
        },
        "$defs": {"Child": Child.model_json_schema()},
    }
    original = deepcopy(source)
    model = custom_model(source)
    # The SDK mutates custom shared schemas; compare using an independent copy.
    expected = to_strict_json_schema(custom_model(deepcopy(source)))
    schema = schema_for(model)

    assert schema == expected
    assert "allOf" not in schema["properties"]["child"]
    assert "$ref" not in schema["properties"]["child"]
    assert schema["properties"]["child"]["required"] == ["label", "tags"]
    assert schema["properties"]["child"]["description"] == "Described child"
    assert source == original


def test_nullable_object_type_and_tuple_items_are_recursively_strict() -> None:
    nullable_object = {
        "type": ["object", "null"],
        "properties": {"value": {"type": "integer", "default": 1}},
    }
    source = {
        "type": "object",
        "properties": {
            "nullable": deepcopy(nullable_object),
            "tuple": {
                "type": "array",
                "prefixItems": [deepcopy(nullable_object), False],
                "items": False,
                "minItems": 2,
                "maxItems": 2,
            },
        },
    }
    original = deepcopy(source)
    schema = schema_for(custom_model(source))

    for child in (
        schema["properties"]["nullable"],
        schema["properties"]["tuple"]["prefixItems"][0],
    ):
        assert child["type"] == ["object", "null"]
        assert child["required"] == ["value"]
        assert child["additionalProperties"] is False
    array = schema["properties"]["tuple"]
    assert array["prefixItems"][1] is False
    assert array["items"] is False
    assert array["minItems"] == array["maxItems"] == 2
    assert source == original


def test_nested_pattern_mapping_is_rejected_without_mutating_source() -> None:
    source = {
        "type": "object",
        "properties": {
            "choice": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "patternProperties": {"^label": {"type": "string"}},
                    },
                ]
            }
        },
    }
    original = deepcopy(source)
    with pytest.raises(ValueError, match="mapping.*OpenAI.*strict"):
        schema_for(custom_model(source))
    assert source == original


def test_saved_request_preserves_envelope_and_strict_schema() -> None:
    request = request_for(Document)
    buffer = io.BytesIO()
    request.save_to_file(buffer, "openai")
    output = json.loads(buffer.getvalue())

    assert output["custom_id"] == request.custom_id
    assert output["method"] == "POST"
    assert output["url"] == "/v1/chat/completions"
    assert output["body"]["messages"] == request.messages
    assert output["body"]["model"] == request.model
    assert output["body"]["max_tokens"] == request.max_tokens
    assert output["body"]["temperature"] == request.temperature
    assert output["body"]["response_format"]["json_schema"] == {
        "name": "Document",
        "strict": True,
        "schema": to_strict_json_schema(Document),
    }
