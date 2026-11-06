from typing import Any

from pydantic import BaseModel, create_model, Field
from instructor import openai_schema


def test_dynamic_model_creation_with_field_description():
    """
    Test that dynamic model creation with Field(description) works correctly.
    This verifies the example in the documentation at docs/concepts/models.md.
    """
    types: dict[str, type[Any]] = {
        "string": str,
        "integer": int,
        "email": str,
    }

    mock_cursor = [
        ("name", "string", "The name of the user."),
        ("age", "integer", "The age of the user."),
        ("email", "email", "The email of the user."),
    ]

    field_definitions: dict[str, Any] = {
        property_name: (types[property_type], Field(description=description))
        for property_name, property_type, description in mock_cursor
    }
    DynamicModel = create_model(
        "User",
        __base__=BaseModel,
        **field_definitions,
    )

    schema = DynamicModel.model_json_schema()

    assert schema["properties"]["name"]["description"] == "The name of the user."
    assert schema["properties"]["age"]["description"] == "The age of the user."
    assert schema["properties"]["email"]["description"] == "The email of the user."

    assert "default" not in schema["properties"]["name"]
    assert "default" not in schema["properties"]["age"]
    assert "default" not in schema["properties"]["email"]

    OpenAISchemaModel = openai_schema(DynamicModel)
    openai_schema_json = OpenAISchemaModel.model_json_schema()

    assert (
        openai_schema_json["properties"]["name"]["description"]
        == "The name of the user."
    )
    assert (
        openai_schema_json["properties"]["age"]["description"] == "The age of the user."
    )
    assert (
        openai_schema_json["properties"]["email"]["description"]
        == "The email of the user."
    )
