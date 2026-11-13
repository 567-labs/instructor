"""Test robust handling of non-object and custom schemas in generate_openai_schema."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, RootModel
from instructor.v2.providers.openai.schema import generate_openai_schema


class TagCollection(RootModel[list[str]]):
    """Collection of string tags.

    :param root: The list of string tags
    """
    pass


class SearchQuery(BaseModel):
    """Search request model.

    :param query: Search query terms
    :param limit: Maximum items to return
    """
    query: str
    limit: int = 10


class CustomSchemaModel(BaseModel):
    """Model with custom schema override without title attribute.

    :param query: search query
    """
    query: str

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        s = super().model_json_schema(*args, **kwargs)
        s.pop("title", None)
        return s


def test_root_model_with_docstring_params():
    """Verify RootModel with docstring parameters does not raise KeyError: 'properties'."""
    schema = generate_openai_schema(TagCollection)
    assert schema["name"] == "TagCollection"
    assert schema["parameters"]["type"] == "array"
    assert "required" not in schema["parameters"]


def test_standard_model_with_docstring_descriptions():
    """Verify standard BaseModel docstring params are correctly mapped to properties."""
    schema = generate_openai_schema(SearchQuery)
    assert schema["name"] == "SearchQuery"
    assert "properties" in schema["parameters"]
    assert schema["parameters"]["properties"]["query"]["description"] == "Search query terms"
    assert schema["parameters"]["required"] == ["query"]


def test_model_without_title_in_json_schema():
    """Verify models whose JSON schema lacks 'title' fall back to model name."""
    schema = generate_openai_schema(CustomSchemaModel)
    assert schema["name"] == "CustomSchemaModel"
    assert "parameters" in schema
