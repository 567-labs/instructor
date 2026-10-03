"""Offline contracts for recursive models in the public schema helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from instructor import OpenAISchema
from instructor.processing.function_calls import ResponseSchema
from instructor.processing.schema import (
    generate_anthropic_schema,
    generate_openai_schema,
)
from instructor.v2.providers.openai.handlers import OpenAIToolsHandler


class Node(BaseModel):
    """A recursive tree.

    Args:
        name: Name from the docstring.
        children: Children from the docstring.
        count: Must not replace the field description.
    """

    model_config = ConfigDict(title="extract_tree", extra="forbid")

    name: str
    children: list[Node] = Field(default_factory=list)
    count: int = Field(default=0, description="Explicit field description.")


class ResponseNode(ResponseSchema):
    name: str
    children: list[ResponseNode] = Field(default_factory=list)


class LegacyNode(OpenAISchema):
    name: str
    children: list[LegacyNode] = Field(default_factory=list)


class OptionalNode(BaseModel):
    name: str = "default"
    child: OptionalNode | None = None


class Parent(BaseModel):
    name: str
    child: Child | None = None


class Child(BaseModel):
    name: str
    parent: Parent | None = None


Parent.model_rebuild()


class Plain(BaseModel):
    name: str
    count: int = 0


class Nested(BaseModel):
    child: Plain


@pytest.mark.parametrize("model", [Node, ResponseNode, LegacyNode, Parent, Child])
def test_recursive_public_schema_helpers(model: type[BaseModel]) -> None:
    raw_schema = model.model_json_schema()
    before = deepcopy(raw_schema)
    # Pydantic 2.8 uses single-allOf; later releases use a direct root $ref.
    ref = raw_schema.get("$ref") or raw_schema["allOf"][0]["$ref"]
    definition = raw_schema["$defs"][ref.rsplit("/", 1)[1]]

    schema = generate_openai_schema(model)
    parameters = schema["parameters"]
    assert schema["name"] == definition["title"]
    assert parameters["type"] == "object"
    assert parameters["required"] == ["name"]
    assert parameters["$defs"] == raw_schema["$defs"]
    assert "$ref" not in parameters
    assert "allOf" not in parameters
    assert "title" not in parameters
    assert "description" not in parameters
    assert set(parameters["properties"]) == set(definition["properties"])
    assert generate_openai_schema(model) is schema

    anthropic = generate_anthropic_schema(model)
    assert anthropic["name"] == schema["name"]
    assert anthropic["description"] == schema["description"]
    assert anthropic["input_schema"] == before
    assert model.model_json_schema() == before


def test_recursive_metadata_defaults_and_references_are_preserved() -> None:
    schema = generate_openai_schema(Node)
    assert schema["name"] == "extract_tree"
    assert (
        schema["description"]
        == Node.model_json_schema()["$defs"]["Node"]["description"]
    )
    parameters = schema["parameters"]
    properties = parameters["properties"]
    assert properties["name"]["description"] == "Name from the docstring."
    assert properties["children"]["description"] == "Children from the docstring."
    assert properties["count"]["description"] == "Explicit field description."
    assert properties["count"]["default"] == 0
    assert "default" not in properties["children"]
    assert properties["children"]["items"] == {"$ref": "#/$defs/Node"}
    assert parameters["additionalProperties"] is False
    assert parameters["required"] == ["name"]

    parsed = Node.model_validate({"name": "root", "children": [{"name": "leaf"}]})
    assert parsed.children == [Node(name="leaf")]
    with pytest.raises(ValidationError):
        Node.model_validate({"children": []})


def test_mutually_recursive_validation_and_references_are_preserved() -> None:
    parsed = Parent.model_validate(
        {"name": "root", "child": {"name": "child", "parent": {"name": "leaf"}}}
    )
    assert parsed.child is not None
    assert parsed.child.parent == Parent(name="leaf")
    parameters = generate_openai_schema(Parent)["parameters"]
    assert parameters["properties"]["child"]["anyOf"][0] == {"$ref": "#/$defs/Child"}
    assert parameters["$defs"]["Child"]["properties"]["parent"]["anyOf"][0] == {
        "$ref": "#/$defs/Parent"
    }


@pytest.mark.parametrize("model", [ResponseNode, LegacyNode])
def test_recursive_schema_class_properties(model: type[ResponseSchema]) -> None:
    assert model.openai_schema == generate_openai_schema(model)
    assert model.anthropic_schema == generate_anthropic_schema(model)


@pytest.mark.parametrize("model", [Node, ResponseNode, LegacyNode, Parent, Child])
def test_anthropic_helper_accepts_recursive_models(model: type[BaseModel]) -> None:
    schema = generate_anthropic_schema(model)
    assert schema["input_schema"] == model.model_json_schema()
    assert schema["name"] == model.model_config.get("title", model.__name__)


def test_recursive_model_with_only_defaults_has_no_required_fields() -> None:
    schema = generate_openai_schema(OptionalNode)
    assert schema["parameters"]["required"] == []
    assert schema["parameters"]["properties"]["child"]["default"] is None
    assert schema["parameters"]["properties"]["name"]["default"] == "default"
    assert schema["description"] == (
        "Correctly extracted `OptionalNode` with all "
        "the required parameters with correct types"
    )
    assert OptionalNode.model_validate({}) == OptionalNode()


@pytest.mark.parametrize(
    "model", [Node, ResponseNode, LegacyNode, Parent, Plain, Nested]
)
def test_tools_prepare_request_accepts_recursive_and_plain_models(
    model: type[BaseModel],
) -> None:
    kwargs: dict[str, Any] = {"messages": [{"role": "user", "content": "Extract."}]}
    before = deepcopy(kwargs)
    prepared, request = OpenAIToolsHandler().prepare_request(model, kwargs)
    assert request["tools"][0]["function"] == generate_openai_schema(prepared)
    assert (
        request["tool_choice"]["function"]["name"]
        == request["tools"][0]["function"]["name"]
    )
    assert request["tools"][0]["function"]["parameters"]["type"] == "object"
    assert kwargs == before


@pytest.mark.parametrize("model", [Plain, Nested])
def test_nonrecursive_schema_output_is_unchanged(model: type[BaseModel]) -> None:
    raw_schema = model.model_json_schema()
    assert generate_openai_schema(model) == {
        "name": raw_schema["title"],
        "description": (
            f"Correctly extracted `{model.__name__}` with all "
            "the required parameters with correct types"
        ),
        "parameters": {
            **{key: value for key, value in raw_schema.items() if key != "title"},
            "required": sorted(raw_schema["required"]),
        },
    }
