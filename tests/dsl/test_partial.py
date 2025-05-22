# type: ignore[all]
from pydantic import BaseModel, Field
from typing import Optional, Union, Literal
from instructor.dsl.partial import Partial, PartialLiteralMixin
import pytest
import instructor
from openai import OpenAI, AsyncOpenAI

models = ["gpt-4o-mini"]
modes = [
    instructor.Mode.TOOLS,
]


class SampleNestedPartial(BaseModel):
    b: int


class SamplePartial(BaseModel):
    a: int
    b: SampleNestedPartial


class NestedA(BaseModel):
    a: str
    b: Optional[str]


class NestedB(BaseModel):
    c: str
    d: str
    e: list[Union[str, int]]
    f: str


class UnionWithNested(BaseModel):
    a: list[Union[NestedA, NestedB]]
    b: list[NestedA]
    c: NestedB


def test_partial():
    partial = Partial[SamplePartial]
    assert partial.model_json_schema() == {
        "$defs": {
            "PartialSampleNestedPartial": {
                "properties": {"b": {"title": "B", "type": "integer"}},
                "required": ["b"],
                "title": "PartialSampleNestedPartial",
                "type": "object",
            }
        },
        "properties": {
            "a": {"title": "A", "type": "integer"},
            "b": {"$ref": "#/$defs/PartialSampleNestedPartial"},
        },
        "required": ["a", "b"],
        "title": "PartialSamplePartial",
        "type": "object",
    }, "Wrapped model JSON schema has changed"
    assert partial.get_partial_model().model_json_schema() == {
        "$defs": {
            "PartialSampleNestedPartial": {
                "properties": {
                    "b": {
                        "anyOf": [{"type": "integer"}, {"type": "null"}],
                        "default": None,
                        "title": "B",
                    }
                },
                "title": "PartialSampleNestedPartial",
                "type": "object",
            }
        },
        "properties": {
            "a": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "title": "A",
            },
            "b": {
                "anyOf": [
                    {"$ref": "#/$defs/PartialSampleNestedPartial"},
                    {"type": "null"},
                ],
                "default": {},
            },
        },
        "title": "PartialSamplePartial",
        "type": "object",
    }, "Partial model JSON schema has changed"


def test_partial_with_whitespace():
    partial = Partial[SamplePartial]

    # Get the actual models from chunks
    models = list(partial.model_from_chunks(["\n", "\t", " ", '{"b": {"b": 1}}']))

    # Print actual values for debugging
    print(f"Number of models: {len(models)}")
    for i, model in enumerate(models):
        print(f"Model {i}: {model.model_dump()}")

    # Actual behavior: When whitespace chunks are processed, we may get models
    # First model has default values
    assert models[0].model_dump() == {"a": None, "b": {}}

    # Last model has b populated from JSON (from the JSON chunk)
    assert models[-1].model_dump() == {"a": None, "b": {"b": 1}}

    # Check we have the expected number of models (2 instead of 4)
    assert len(models) == 2


@pytest.mark.asyncio
async def test_async_partial_with_whitespace():
    partial = Partial[SamplePartial]

    # Handle any leading whitespace from the model
    async def async_generator():
        for chunk in ["\n", "\t", " ", '{"b": {"b": 1}}']:
            yield chunk

    expected_model_dicts = [
        {"a": None, "b": {}},
        {"a": None, "b": {}},
        {"a": None, "b": {}},
        {"a": None, "b": {"b": 1}},
    ]

    i = 0
    async for model in partial.model_from_chunks_async(async_generator()):
        assert model.model_dump() == expected_model_dicts[i]
        i += 1

    assert model.model_dump() == {"a": None, "b": {"b": 1}}


def test_summary_extraction():
    class Summary(BaseModel, PartialLiteralMixin):
        summary: str = Field(description="A detailed summary")

    client = OpenAI()
    client = instructor.from_openai(client, mode=instructor.Mode.TOOLS)
    extraction_stream = client.chat.completions.create_partial(
        model="gpt-4o",
        response_model=Summary,
        messages=[
            {"role": "system", "content": "You summarize text"},
            {"role": "user", "content": "Summarize: Mary had a little lamb"},
        ],
        stream=True,
    )

    previous_summary = None
    updates = 0
    for extraction in extraction_stream:
        if previous_summary is not None and extraction:
            updates += 1
        previous_summary = extraction.summary

    assert updates == 1


@pytest.mark.asyncio
async def test_summary_extraction_async():
    class Summary(BaseModel, PartialLiteralMixin):
        summary: str = Field(description="A detailed summary")

    client = AsyncOpenAI()
    client = instructor.from_openai(client, mode=instructor.Mode.TOOLS)
    extraction_stream = client.chat.completions.create_partial(
        model="gpt-4o",
        response_model=Summary,
        messages=[
            {"role": "system", "content": "You summarize text"},
            {"role": "user", "content": "Summarize: Mary had a little lamb"},
        ],
        stream=True,
    )

    previous_summary = None
    updates = 0
    async for extraction in extraction_stream:
        if previous_summary is not None and extraction:
            updates += 1
        previous_summary = extraction.summary

    assert updates == 1


def test_union_with_nested():
    partial = Partial[UnionWithNested]
    partial.get_partial_model().model_validate_json(
        '{"a": [{"b": "b"}, {"d": "d"}], "b": [{"b": "b"}], "c": {"d": "d"}, "e": [1, "a"]}'
    )


def test_literal_partial_streaming():
    """Test that Literal types work correctly with Partial during streaming"""

    class UserWithLiteral(BaseModel):
        name: Literal["Bob", "Alice", "John"]
        status: Literal["active", "inactive", "pending"]
        age: int

    partial = Partial[UserWithLiteral]
    partial_model = partial.get_partial_model()

    # Test various streaming scenarios
    test_cases = [
        (
            "{}",
            {"name": "", "status": "", "age": None},
        ),  # Required literals default to ""
        ('{"name": ""}', {"name": "", "status": "", "age": None}),
        ('{"name": "A"}', {"name": "A", "status": "", "age": None}),
        ('{"name": "Al"}', {"name": "Al", "status": "", "age": None}),
        ('{"name": "Ali"}', {"name": "Ali", "status": "", "age": None}),
        ('{"name": "Alice"}', {"name": "Alice", "status": "", "age": None}),
        (
            '{"name": "Alice", "status": "a"}',
            {"name": "Alice", "status": "a", "age": None},
        ),
        (
            '{"name": "Alice", "status": "active"}',
            {"name": "Alice", "status": "active", "age": None},
        ),
        (
            '{"name": "Alice", "status": "active", "age": 25}',
            {"name": "Alice", "status": "active", "age": 25},
        ),
    ]

    for json_str, expected in test_cases:
        result = partial_model.model_validate_json(json_str)
        assert result.model_dump() == expected, f"Failed for {json_str}"


def test_literal_field_schema():
    """Test that Literal fields in Partial models have the correct schema"""

    class ModelWithLiteral(BaseModel):
        color: Literal["red", "green", "blue"]
        size: int

    partial = Partial[ModelWithLiteral]
    partial_model = partial.get_partial_model()
    schema = partial_model.model_json_schema()

    # Check that the color field allows both literal values and strings
    color_schema = schema["properties"]["color"]
    assert "anyOf" in color_schema

    # Should have 2 options for required literal: the literal enum and string type
    any_of_options = color_schema["anyOf"]
    assert len(any_of_options) == 2  # No null for required fields

    # Check for the literal enum option
    has_enum = any(
        opt.get("enum") == ["red", "green", "blue"] for opt in any_of_options
    )
    assert has_enum, "Should have literal enum option"

    # Check for string type option
    has_string = any(opt.get("type") == "string" for opt in any_of_options)
    assert has_string, "Should have string type option"

    # Check that there's NO null type option (because it's required)
    has_null = any(opt.get("type") == "null" for opt in any_of_options)
    assert not has_null, "Should NOT have null type option for required field"
