import os
from collections.abc import AsyncGenerator, Generator, Iterable
from copy import deepcopy
from enum import Enum
from typing import Any, Literal, Optional, Protocol, TypeVar, Union, cast

import pytest
from jiter import from_json
from pydantic import BaseModel, Field, ValidationError

import instructor
from instructor.dsl.partial import Partial, PartialLiteralMixin, _make_field_optional
from openai import AsyncOpenAI, OpenAI


T_Model = TypeVar("T_Model", bound=BaseModel)


class _PartialModelApi(Protocol[T_Model]):
    @classmethod
    def get_partial_model(cls) -> type[T_Model]: ...

    @classmethod
    def model_from_chunks(
        cls, json_chunks: Iterable[Any], **kwargs: Any
    ) -> Generator[T_Model, None, None]: ...

    @classmethod
    def model_from_chunks_async(
        cls, json_chunks: AsyncGenerator[str, None], **kwargs: Any
    ) -> AsyncGenerator[T_Model, None]: ...


def _partial_api(model: type[T_Model]) -> type[_PartialModelApi[T_Model]]:
    """Expose the classmethods dynamically added by ``Partial[T]``."""
    return cast(type[_PartialModelApi[T_Model]], model)


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


class PEP604UnionField(BaseModel):
    value: str | int


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
    assert _partial_api(partial).get_partial_model().model_json_schema() == {
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


partial_chunks = ["\n", "\t", " ", "\x00", '{"a": 42, "b": {"b": 1}}']
expected_sync_models = [
    # First model has default values (nested models show their fields as None)
    {"a": None, "b": {"b": None}},
    {"a": None, "b": {"b": None}},
    {"a": None, "b": {"b": None}},
    {"a": None, "b": {"b": None}},
    # Last model has all fields populated from JSON
    {"a": 42, "b": {"b": 1}},
]
expected_async_models = [
    {"a": None, "b": {"b": None}},
    {"a": None, "b": {"b": None}},
    {"a": None, "b": {"b": None}},
    {"a": None, "b": {"b": None}},
    {"a": 42, "b": {"b": 1}},
]


def test_partial_with_whitespace():
    partial = Partial[SamplePartial]
    # Get the actual models from chunks - must provide complete data for final validation
    models = list(_partial_api(partial).model_from_chunks(partial_chunks))
    assert len(models) == len(expected_sync_models)
    for i, model in enumerate(models):
        assert model.model_dump() == expected_sync_models[i]


@pytest.mark.asyncio
async def test_async_partial_with_whitespace():
    partial = Partial[SamplePartial]

    # Handle any leading whitespace from the model - must provide complete data for final validation
    async def async_generator():
        for chunk in partial_chunks:
            yield chunk

    i = 0
    async for model in _partial_api(partial).model_from_chunks_async(async_generator()):
        # Expected behavior: When whitespace chunks are processed, we should always get a model
        assert model.model_dump() == expected_async_models[i]
        i += 1
    assert i == len(expected_async_models)


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_summary_extraction():
    class Summary(BaseModel):
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

    # Collect all streaming updates and verify final result
    final_summary = None
    chunk_count = 0
    for extraction in extraction_stream:
        final_summary = extraction.summary
        chunk_count += 1

    # Verify we got streaming updates and a valid final summary
    assert chunk_count > 0
    assert final_summary is not None
    assert len(final_summary) > 0


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
@pytest.mark.asyncio
async def test_summary_extraction_async():
    class Summary(BaseModel):
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

    # Collect all streaming updates and verify final result
    final_summary = None
    chunk_count = 0
    async for extraction in extraction_stream:
        final_summary = extraction.summary
        chunk_count += 1

    # Verify we got streaming updates and a valid final summary
    assert chunk_count > 0
    assert final_summary is not None
    assert len(final_summary) > 0


def test_union_with_nested():
    partial = Partial[UnionWithNested]
    _partial_api(partial).get_partial_model().model_validate_json(
        '{"a": [{"b": "b"}, {"d": "d"}], "b": [{"b": "b"}], "c": {"d": "d"}, "e": [1, "a"]}'
    )


def test_partial_streaming_with_pep604_union_field():
    partial = Partial[PEP604UnionField]

    models = list(_partial_api(partial).model_from_chunks(['{"value": ', "1}"]))

    assert models[-1].model_dump() == {"value": 1}


def test_partial_with_default_factory():
    """Test that Partial works with fields that have default_factory.

    This test ensures that when making fields optional, the default_factory
    is properly cleared to avoid Pydantic validation errors about having
    both default and default_factory set.
    """

    class ModelWithDefaultFactory(BaseModel):
        items: list[str] = Field(default_factory=list)
        tags: dict[str, str] = Field(default_factory=dict)
        name: str

    # This should not raise a validation error about both default and default_factory
    partial = Partial[ModelWithDefaultFactory]
    partial_model = _partial_api(partial).get_partial_model()

    # Verify we can instantiate and validate
    # In Partial models, all fields are made Optional with default=None
    instance = partial_model()
    assert instance.items is None
    assert instance.tags is None
    assert instance.name is None

    # Test with partial data
    instance2 = partial_model.model_validate({"items": ["a", "b"]})
    assert instance2.items == ["a", "b"]
    assert instance2.tags is None
    assert instance2.name is None


class TestMakeFieldOptionalWorksWithPydanticV2:
    """Tests proving that _make_field_optional with deepcopy works correctly in Pydantic v2.

    These tests refute the claim that deepcopy + setting default = None doesn't work
    in Pydantic v2. The implementation is correct and fields are properly made optional.

    See: https://github.com/instructor-ai/instructor/issues/XXXX
    """

    def test_deepcopy_approach_makes_field_optional(self):
        """Verify that deepcopy + default = None makes fields optional in Pydantic v2."""

        class Original(BaseModel):
            name: str  # Required field

        field = Original.model_fields["name"]
        assert field.is_required() is True, "Original field should be required"

        # This is what _make_field_optional does
        tmp = deepcopy(field)
        tmp.default = None
        tmp.annotation = Optional[str]

        assert tmp.is_required() is False, "Modified field should not be required"
        assert tmp.default is None, "Default should be None"

    def test_make_field_optional_function_works(self):
        """Verify _make_field_optional correctly transforms required fields."""

        class TestModel(BaseModel):
            name: str
            age: int

        for field_name, field_info in TestModel.model_fields.items():
            assert field_info.is_required() is True, f"{field_name} should be required"

            annotation, new_field = _make_field_optional(field_info)
            assert new_field.is_required() is False, (
                f"{field_name} should be optional after transformation"
            )
            assert new_field.default is None, f"{field_name} should have None default"

    def test_partial_model_validates_empty_dict(self):
        """Verify Partial models can validate empty dicts (all fields None)."""

        class MyModel(BaseModel):
            name: str
            age: int
            status: str

        PartialModel = Partial[MyModel]
        TruePartial = _partial_api(PartialModel).get_partial_model()

        # This should NOT raise ValidationError
        result = TruePartial.model_validate({})

        assert result.name is None
        assert result.age is None
        assert result.status is None

    def test_partial_validates_incremental_streaming_data(self):
        """Verify Partial models correctly handle incremental streaming data."""

        class MyModel(BaseModel):
            name: str
            age: int

        PartialModel = Partial[MyModel]
        TruePartial = _partial_api(PartialModel).get_partial_model()

        # Simulate streaming JSON chunks
        streaming_states = [
            ("{}", None, None),
            ('{"name": "Jo', "Jo", None),  # Partial string
            ('{"name": "John"}', "John", None),
            ('{"name": "John", "age": 25}', "John", 25),
        ]

        for json_str, expected_name, expected_age in streaming_states:
            obj = from_json(json_str.encode(), partial_mode="trailing-strings")
            result = TruePartial.model_validate(obj)
            assert result.name == expected_name, f"Failed for {json_str}"
            assert result.age == expected_age, f"Failed for {json_str}"

    def test_partial_with_all_field_types(self):
        """Verify _make_field_optional works with various field types."""

        class ComplexModel(BaseModel):
            string_field: str
            int_field: int
            float_field: float
            bool_field: bool
            list_field: list[str]
            optional_field: Optional[str]

        PartialModel = Partial[ComplexModel]
        TruePartial = _partial_api(PartialModel).get_partial_model()

        # All fields should validate with empty dict
        result = TruePartial.model_validate({})

        assert result.string_field is None
        assert result.int_field is None
        assert result.float_field is None
        assert result.bool_field is None
        assert result.list_field is None
        assert result.optional_field is None


class TestLiteralTypeStreaming:
    """Tests for Literal type handling during streaming.

    Without PartialLiteralMixin: uses partial_mode='trailing-strings', which keeps
    incomplete strings and causes validation errors for Literal/Enum fields.

    With PartialLiteralMixin: uses partial_mode='on', which drops incomplete strings
    so fields become None.
    """

    def test_literal_without_mixin_fails_on_incomplete_string(self):
        """Without PartialLiteralMixin, incomplete Literal strings cause validation errors."""

        class ModelWithLiteral(BaseModel):
            status: Literal["active", "inactive"]

        PartialModel = Partial[ModelWithLiteral]
        TruePartial = _partial_api(PartialModel).get_partial_model()

        # With partial_mode="trailing-strings", incomplete strings are kept
        partial_json = b'{"status": "act'
        obj = from_json(partial_json, partial_mode="trailing-strings")
        # obj is {"status": "act"} - a partial string that fails Literal validation

        with pytest.raises(ValidationError):
            TruePartial.model_validate(obj)

    def test_literal_with_mixin_incomplete_string_becomes_none(self):
        """With PartialLiteralMixin, incomplete Literal strings are dropped."""

        class ModelWithLiteral(BaseModel, PartialLiteralMixin):
            status: Literal["active", "inactive"]

        PartialModel = Partial[ModelWithLiteral]
        TruePartial = _partial_api(PartialModel).get_partial_model()

        # With partial_mode="on" (enabled by PartialLiteralMixin), incomplete strings are dropped
        partial_json = b'{"status": "act'
        obj = from_json(partial_json, partial_mode="on")
        # obj is {} because the incomplete string was dropped

        result = TruePartial.model_validate(obj)
        assert result.status is None

    def test_literal_accepts_valid_complete_value(self):
        """Literal fields should accept valid complete values."""

        class ModelWithLiteral(BaseModel, PartialLiteralMixin):
            status: Literal["active", "inactive"]

        PartialModel = Partial[ModelWithLiteral]
        TruePartial = _partial_api(PartialModel).get_partial_model()

        result = TruePartial.model_validate({"status": "active"})
        assert result.status == "active"

        result = TruePartial.model_validate({"status": "inactive"})
        assert result.status == "inactive"

    def test_literal_with_missing_field_is_none(self):
        """Literal fields should be None when not present in data."""

        class ModelWithLiteral(BaseModel, PartialLiteralMixin):
            name: str
            status: Literal["active", "inactive"]

        PartialModel = Partial[ModelWithLiteral]
        TruePartial = _partial_api(PartialModel).get_partial_model()

        result = TruePartial.model_validate({"name": "John"})
        assert result.name == "John"
        assert result.status is None

    def test_literal_default_is_available_during_streaming(self):
        """Literal fields with explicit defaults should be present in partial results."""

        class Person(BaseModel):
            type: Literal["Person"] = "Person"
            name: str
            age: int

        PartialModel = Partial[Person]

        results = list(_partial_api(PartialModel).model_from_chunks(['{"name": "Joh']))

        assert len(results) == 1
        assert results[0].type == "Person"
        assert results[0].name == "Joh"
        assert results[0].age is None

    def test_literal_rejects_complete_invalid_value(self):
        """Complete but invalid Literal values should fail validation."""

        class ModelWithLiteral(BaseModel, PartialLiteralMixin):
            status: Literal["active", "inactive"]

        PartialModel = Partial[ModelWithLiteral]
        TruePartial = _partial_api(PartialModel).get_partial_model()

        # "xyz" is a complete string but not a valid Literal value
        with pytest.raises(ValidationError):
            TruePartial.model_validate({"status": "xyz"})


class TestPartialStreamingWithComplexTypes:
    """Tests for streaming with complex Pydantic types using PartialLiteralMixin.

    With PartialLiteralMixin, partial_mode='on' is used, so incomplete values are dropped.
    """

    def test_enum_incomplete_string_becomes_none(self):
        """With PartialLiteralMixin, incomplete Enum strings are dropped."""

        class Status(Enum):
            ACTIVE = "active"
            INACTIVE = "inactive"

        class ModelWithEnum(BaseModel, PartialLiteralMixin):
            status: Status

        PartialModel = Partial[ModelWithEnum]
        TruePartial = _partial_api(PartialModel).get_partial_model()

        # Incomplete string is dropped with partial_mode="on"
        obj = from_json(b'{"status": "act', partial_mode="on")
        result = TruePartial.model_validate(obj)
        assert result.status is None

    def test_enum_accepts_valid_complete_value(self):
        """Enum fields should accept valid complete values."""

        class Status(Enum):
            ACTIVE = "active"
            INACTIVE = "inactive"

        class ModelWithEnum(BaseModel, PartialLiteralMixin):
            status: Status

        PartialModel = Partial[ModelWithEnum]
        TruePartial = _partial_api(PartialModel).get_partial_model()

        result = TruePartial.model_validate({"status": "active"})
        assert result.status == Status.ACTIVE

    def test_optional_literal_incomplete_string_becomes_none(self):
        """With PartialLiteralMixin, incomplete Optional[Literal] strings are dropped."""

        class ModelWithOptionalLiteral(BaseModel, PartialLiteralMixin):
            status: Optional[Literal["on", "off"]] = None

        PartialModel = Partial[ModelWithOptionalLiteral]
        TruePartial = _partial_api(PartialModel).get_partial_model()

        obj = from_json(b'{"status": "o', partial_mode="on")
        result = TruePartial.model_validate(obj)
        assert result.status is None

    def test_optional_literal_accepts_valid_value(self):
        """Optional[Literal] should accept valid complete values."""

        class ModelWithOptionalLiteral(BaseModel, PartialLiteralMixin):
            status: Optional[Literal["on", "off"]] = None

        PartialModel = Partial[ModelWithOptionalLiteral]
        TruePartial = _partial_api(PartialModel).get_partial_model()

        result = TruePartial.model_validate({"status": "on"})
        assert result.status == "on"

    def test_union_literal_incomplete_string_becomes_none(self):
        """With PartialLiteralMixin, incomplete Union[Literal, int] strings are dropped."""

        class ModelWithUnion(BaseModel, PartialLiteralMixin):
            value: Union[Literal["yes", "no"], int]

        PartialModel = Partial[ModelWithUnion]
        TruePartial = _partial_api(PartialModel).get_partial_model()

        # Incomplete string is dropped
        obj = from_json(b'{"value": "ye', partial_mode="on")
        result = TruePartial.model_validate(obj)
        assert result.value is None

    def test_union_literal_accepts_valid_values(self):
        """Union[Literal, int] should accept both valid Literal and int."""

        class ModelWithUnion(BaseModel, PartialLiteralMixin):
            value: Union[Literal["yes", "no"], int]

        PartialModel = Partial[ModelWithUnion]
        TruePartial = _partial_api(PartialModel).get_partial_model()

        result = TruePartial.model_validate({"value": "yes"})
        assert result.value == "yes"

        result = TruePartial.model_validate({"value": 42})
        assert result.value == 42

    def test_partial_model_supports_pep604_union_annotations(self):
        class MyResponse(BaseModel):
            value: str | int

        PartialModel = Partial[MyResponse]
        TruePartial = _partial_api(PartialModel).get_partial_model()

        result = TruePartial.model_validate({"value": "hello"})
        assert result.value == "hello"

        result = TruePartial.model_validate({"value": 42})
        assert result.value == 42

    def test_union_of_literals_matches_all_branches(self):
        """Union[Literal, Literal] should match values from all branches."""

        class ModelWithUnionLiterals(BaseModel, PartialLiteralMixin):
            value: Union[Literal["a", "b"], Literal["x", "y"]]

        PartialModel = Partial[ModelWithUnionLiterals]
        TruePartial = _partial_api(PartialModel).get_partial_model()

        # Both branches should work
        assert TruePartial.model_validate({"value": "a"}).value == "a"
        assert TruePartial.model_validate({"value": "b"}).value == "b"
        assert TruePartial.model_validate({"value": "x"}).value == "x"
        assert TruePartial.model_validate({"value": "y"}).value == "y"

    def test_list_literal_incomplete_item_dropped(self):
        """With PartialLiteralMixin, incomplete list items are dropped."""

        class ModelWithLiteralList(BaseModel, PartialLiteralMixin):
            tags: list[Literal["admin", "user", "guest"]]

        PartialModel = Partial[ModelWithLiteralList]
        TruePartial = _partial_api(PartialModel).get_partial_model()

        # Incomplete list item is dropped
        obj = from_json(b'{"tags": ["admin", "us', partial_mode="on")
        result = TruePartial.model_validate(obj)
        assert result.tags == ["admin"]

    def test_list_literal_accepts_valid_items(self):
        """list[Literal] should accept valid complete items."""

        class ModelWithLiteralList(BaseModel, PartialLiteralMixin):
            tags: list[Literal["admin", "user", "guest"]]

        PartialModel = Partial[ModelWithLiteralList]
        TruePartial = _partial_api(PartialModel).get_partial_model()

        result = TruePartial.model_validate({"tags": ["admin", "user"]})
        assert result.tags == ["admin", "user"]


class TestDiscriminatedUnionPartial:
    """Tests for discriminated unions with Partial streaming.

    KNOWN LIMITATION: Discriminated unions don't work with Partial because:
    - Partial makes all fields Optional
    - Pydantic requires discriminator fields to be strictly Literal, not Optional[Literal]

    Workaround: Use Union without the discriminator parameter.
    """

    def test_discriminated_union_not_compatible_with_partial(self):
        """Discriminated unions fail with Partial (known limitation)."""

        class Cat(BaseModel):
            pet_type: Literal["cat"]
            meows: int

        class Dog(BaseModel):
            pet_type: Literal["dog"]
            barks: int

        class PetContainer(BaseModel):
            pet: Union[Cat, Dog] = Field(discriminator="pet_type")

        # Fails because Partial makes pet_type Optional, but discriminators must be Literal
        from pydantic import PydanticUserError

        PartialModel = Partial[PetContainer]
        with pytest.raises(PydanticUserError):
            _partial_api(PartialModel).get_partial_model()

    def test_union_without_discriminator_works(self):
        """Union without discriminator works with Partial streaming."""

        class Cat(BaseModel):
            pet_type: Literal["cat"]
            meows: int

        class Dog(BaseModel):
            pet_type: Literal["dog"]
            barks: int

        class PetContainerNoDiscriminator(BaseModel):
            pet: Union[Cat, Dog]  # No discriminator - works with Partial

        PartialModel = Partial[PetContainerNoDiscriminator]
        TruePartial = _partial_api(PartialModel).get_partial_model()

        # Complete value works
        result = TruePartial.model_validate({"pet": {"pet_type": "cat", "meows": 5}})
        assert result.pet is not None
        assert result.pet.pet_type == "cat"

    def test_single_value_literal_incomplete_string(self):
        """Single-value Literals with incomplete strings become None."""

        class Cat(BaseModel):
            pet_type: Literal["cat"]

        PartialModel = Partial[Cat]
        TruePartial = _partial_api(PartialModel).get_partial_model()

        # Incomplete string is dropped
        obj = from_json(b'{"pet_type": "ca', partial_mode="on")
        result = TruePartial.model_validate(obj)
        assert result.pet_type is None

        # Complete value works
        result = TruePartial.model_validate({"pet_type": "cat"})
        assert result.pet_type == "cat"


class TestModelValidatorsDuringStreaming:
    """Tests for model validators during partial streaming.

    Model validators are automatically wrapped to skip during streaming
    (when context={"partial_streaming": True} is passed) and only run
    when validating without that context (final validation).
    """

    def test_model_validator_skipped_during_streaming(self):
        """Model validators should be skipped when streaming context is passed."""
        from pydantic import model_validator

        class ModelWithValidator(BaseModel, PartialLiteralMixin):
            status: Literal["active", "inactive"]
            priority: Literal["high", "low"]

            @model_validator(mode="after")
            def validate_relationships(self):
                # This would fail during streaming without wrapping
                if self.status is not None and self.priority is None:
                    raise ValueError("If status is set, priority must also be set!")
                return self

        PartialModel = Partial[ModelWithValidator]

        # With completeness-based validation, incomplete JSON skips all validation
        # by using model_construct() instead of model_validate()
        chunks = ['{"status": "act']  # Incomplete JSON
        results = list(_partial_api(PartialModel).model_from_chunks(chunks))
        # Incomplete JSON - no validation runs, partial value stored
        assert results[0].status == "act"
        assert results[0].priority is None

    def test_model_validator_runs_when_complete(self):
        """Model validators should run when all fields are complete."""
        from pydantic import model_validator

        class ModelWithValidator(BaseModel, PartialLiteralMixin):
            status: Literal["active", "inactive"]
            priority: Literal["high", "low"]

            @model_validator(mode="after")
            def validate_relationships(self):
                if self.status == "active" and self.priority == "low":
                    raise ValueError("Active status requires high priority!")
                return self

        PartialModel = Partial[ModelWithValidator]
        TruePartial = _partial_api(PartialModel).get_partial_model()

        # Valid complete data
        result = TruePartial.model_validate({"status": "active", "priority": "high"})
        assert result.status == "active"
        assert result.priority == "high"

        # Invalid complete data should fail
        with pytest.raises(ValidationError):
            TruePartial.model_validate({"status": "active", "priority": "low"})

    def test_multiple_model_validators(self):
        """Multiple model validators should all be wrapped and run when complete."""
        from pydantic import model_validator

        validator_calls = []

        class ModelWithMultipleValidators(BaseModel, PartialLiteralMixin):
            a: Literal["x", "y"]
            b: Literal["1", "2"]

            @model_validator(mode="after")
            def validator_one(self):
                validator_calls.append("one")
                return self

            @model_validator(mode="after")
            def validator_two(self):
                validator_calls.append("two")
                return self

        PartialModel = Partial[ModelWithMultipleValidators]

        # During streaming with incomplete JSON, validators should be skipped
        # because model_construct() is used instead of model_validate()
        validator_calls.clear()
        chunks = ['{"a": "x']  # Incomplete JSON
        list(_partial_api(PartialModel).model_from_chunks(chunks))
        assert validator_calls == []

        # Complete JSON - validators run during model_validate
        validator_calls.clear()
        chunks = ['{"a": "x", "b": "1"}']  # Complete JSON
        list(_partial_api(PartialModel).model_from_chunks(chunks))
        assert "one" in validator_calls
        assert "two" in validator_calls

    def test_validators_run_without_streaming_context(self):
        """Validators should run when no streaming context is passed (final validation)."""
        from pydantic import model_validator

        class ModelWithValidator(BaseModel, PartialLiteralMixin):
            status: Literal["active", "inactive"]
            priority: Literal["high", "low"]

            @model_validator(mode="after")
            def validate_relationships(self):
                if self.status == "active" and self.priority == "low":
                    raise ValueError("Active requires high priority!")
                return self

        PartialModel = Partial[ModelWithValidator]
        TruePartial = _partial_api(PartialModel).get_partial_model()

        # Without streaming context, validators run even with incomplete data
        # This is the final validation scenario
        with pytest.raises(ValidationError):
            TruePartial.model_validate({"status": "active", "priority": "low"})

        # Valid complete data passes
        result = TruePartial.model_validate({"status": "active", "priority": "high"})
        assert result.status == "active"
        assert result.priority == "high"


class TestFinalValidationAfterStreaming:
    """Tests for final validation after streaming completes.

    When streaming ends, the final object is validated against the original
    model to enforce required fields and run validators without streaming context.
    """

    def test_final_validation_catches_missing_required_fields(self):
        """Final validation should fail if required fields are missing."""

        class ModelWithRequired(BaseModel):
            name: str  # Required
            age: int  # Required
            nickname: Optional[str] = None  # Optional

        PartialModel = Partial[ModelWithRequired]

        # Simulate streaming that doesn't provide all required fields
        chunks = ['{"name": "John"}']  # Missing 'age'

        with pytest.raises(ValidationError) as exc_info:
            list(_partial_api(PartialModel).model_from_chunks(iter(chunks)))

        # Should fail because 'age' is required but missing
        assert "age" in str(exc_info.value)

    def test_final_validation_passes_with_all_required_fields(self):
        """Final validation should pass when all required fields are present."""

        class ModelWithRequired(BaseModel):
            name: str
            age: int

        PartialModel = Partial[ModelWithRequired]

        # Simulate streaming that provides all required fields
        chunks = ['{"name": "John", "age": 30}']

        results = list(_partial_api(PartialModel).model_from_chunks(iter(chunks)))
        assert len(results) > 0
        final = results[-1]
        assert final.name == "John"
        assert final.age == 30

    def test_final_validation_runs_model_validators(self):
        """Final validation should run model validators without streaming context."""
        from pydantic import model_validator

        class ModelWithValidator(BaseModel, PartialLiteralMixin):
            status: Literal["active", "inactive"]
            priority: Literal["high", "low"]

            @model_validator(mode="after")
            def check_consistency(self):
                if self.status == "active" and self.priority == "low":
                    raise ValueError("Active tasks must have high priority")
                return self

        PartialModel = Partial[ModelWithValidator]

        # This should fail final validation due to the model validator
        chunks = ['{"status": "active", "priority": "low"}']

        with pytest.raises(ValidationError) as exc_info:
            list(_partial_api(PartialModel).model_from_chunks(iter(chunks)))

        assert "Active tasks must have high priority" in str(exc_info.value)

    def test_streaming_yields_partial_objects_before_final_validation(self):
        """Streaming should yield partial objects even if final validation will fail."""

        class ModelWithRequired(BaseModel):
            name: str
            age: int

        PartialModel = Partial[ModelWithRequired]

        # Stream with incomplete JSON first, then complete JSON
        # First chunk is incomplete, yields partial object
        # Second chunk completes the JSON with all required fields
        chunks = ['{"name": "Jo', 'hn", "age": 25}']

        partial_objects = []
        for obj in _partial_api(PartialModel).model_from_chunks(iter(chunks)):
            partial_objects.append(obj)

        # Should have yielded partial objects during streaming
        assert len(partial_objects) >= 1
        # First partial object has incomplete name
        assert partial_objects[0].name == "Jo"
        # Final object is fully validated
        assert partial_objects[-1].name == "John"
        assert partial_objects[-1].age == 25

    def test_original_model_reference_is_stored(self):
        """Partial model should store reference to original model."""

        class OriginalModel(BaseModel):
            name: str

        PartialModel = Partial[OriginalModel]

        assert hasattr(PartialModel, "_original_model")
        assert PartialModel._original_model is OriginalModel

    @pytest.mark.asyncio
    async def test_async_final_validation_catches_missing_required_fields(self):
        """Async streaming should also do final validation."""

        class ModelWithRequired(BaseModel):
            name: str
            age: int

        PartialModel = Partial[ModelWithRequired]

        async def async_chunks():
            yield '{"name": "John"}'  # Missing 'age'

        with pytest.raises(ValidationError) as exc_info:
            async for _ in _partial_api(PartialModel).model_from_chunks_async(
                async_chunks()
            ):
                pass

        assert "age" in str(exc_info.value)

    def test_final_validation_accepts_explicit_null_for_required_nullable_field(self):
        """Explicit JSON null for a required-but-nullable field should validate.

        The final validation must not confuse a field the model explicitly
        returned as null with a field that was never streamed at all.
        """

        class ModelWithNullable(BaseModel):
            name: str
            email: Optional[str]  # Required, but nullable

        PartialModel = Partial[ModelWithNullable]

        chunks = ['{"name": "Al', 'ice", "email"', ": null}"]

        results = list(_partial_api(PartialModel).model_from_chunks(iter(chunks)))
        assert len(results) > 0
        final = results[-1]
        assert final.name == "Alice"
        assert final.email is None

    def test_final_validation_still_rejects_absent_required_nullable_field(self):
        """A required nullable field genuinely absent from complete JSON still fails."""

        class ModelWithNullable(BaseModel):
            name: str
            email: Optional[str]  # Required, but nullable

        PartialModel = Partial[ModelWithNullable]

        chunks = ['{"name": "Alice"}']  # 'email' truly missing

        with pytest.raises(ValidationError) as exc_info:
            list(_partial_api(PartialModel).model_from_chunks(iter(chunks)))

        assert "email" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_final_validation_accepts_explicit_null_for_required_nullable_field(
        self,
    ):
        """Async streaming should also accept explicit nulls for nullable fields."""

        class ModelWithNullable(BaseModel):
            name: str
            email: Optional[str]  # Required, but nullable

        PartialModel = Partial[ModelWithNullable]

        async def async_chunks():
            yield '{"name": "Al'
            yield 'ice", "email"'
            yield ": null}"

        results = []
        async for obj in _partial_api(PartialModel).model_from_chunks_async(
            async_chunks()
        ):
            results.append(obj)

        assert len(results) > 0
        final = results[-1]
        assert final.name == "Alice"
        assert final.email is None


class TestRecursiveModels:
    """Test that Partial handles self-referential models without infinite recursion."""

    def test_basic_recursive_model(self):
        """Partial should work with basic recursive models."""

        class TreeNode(BaseModel):
            value: str
            children: Optional[list["TreeNode"]] = None

        TreeNode.model_rebuild()

        # Should not raise RecursionError
        PartialTreeNode = Partial[TreeNode]
        TruePartial = _partial_api(PartialTreeNode).get_partial_model()

        # Can validate partial data
        result = TruePartial.model_validate({"value": "root"})
        assert result.value == "root"
        assert result.children is None

    def test_nested_recursive_model(self):
        """Partial should work with nested children."""

        class TreeNode(BaseModel):
            value: str
            children: Optional[list["TreeNode"]] = None

        TreeNode.model_rebuild()

        PartialTreeNode = Partial[TreeNode]
        TruePartial = _partial_api(PartialTreeNode).get_partial_model()

        # Validate with nested structure
        data = {
            "value": "root",
            "children": [
                {"value": "child1"},
                {"value": "child2", "children": [{"value": "grandchild"}]},
            ],
        }
        result = TruePartial.model_validate(data)
        assert result.value == "root"
        assert len(result.children) == 2
        assert result.children[0].value == "child1"
        assert result.children[1].children[0].value == "grandchild"

    def test_mutually_recursive_models(self):
        """Partial should handle mutually recursive models."""

        class Person(BaseModel):
            name: str
            employer: Optional["Company"] = None

        class Company(BaseModel):
            name: str
            employees: Optional[list[Person]] = None

        Person.model_rebuild()
        Company.model_rebuild()

        # Both should work without RecursionError
        PartialPerson = Partial[Person]
        PartialCompany = Partial[Company]

        assert PartialPerson is not None
        assert PartialCompany is not None

        # Validate partial data
        person_partial = _partial_api(PartialPerson).get_partial_model()
        result = person_partial.model_validate({"name": "Alice"})
        assert result.name == "Alice"

    def test_direct_self_reference(self):
        """Partial should handle direct self-reference (linked list style)."""

        class LinkedNode(BaseModel):
            value: int
            next: Optional["LinkedNode"] = None

        LinkedNode.model_rebuild()

        # Should not raise RecursionError
        PartialLinked = Partial[LinkedNode]
        TruePartial = _partial_api(PartialLinked).get_partial_model()

        # Validate chain
        data = {"value": 1, "next": {"value": 2, "next": {"value": 3}}}
        result = TruePartial.model_validate(data)
        assert result.value == 1
        assert result.next.value == 2
        assert result.next.next.value == 3

    def test_complex_recursive_with_validators(self):
        """Complex recursive model with validators, multiple self-refs, and nested types."""
        from typing import Literal
        from pydantic import model_validator, field_validator
        from enum import Enum

        class NodeType(Enum):
            FOLDER = "folder"
            FILE = "file"
            SYMLINK = "symlink"

        class Permission(BaseModel):
            user: str
            level: Literal["read", "write", "admin"]

        class FileSystemNode(BaseModel):
            name: str
            node_type: NodeType
            size_bytes: Optional[int] = None
            children: Optional[list["FileSystemNode"]] = None
            parent: Optional["FileSystemNode"] = None
            symlink_target: Optional["FileSystemNode"] = None
            permissions: Optional[list[Permission]] = None
            metadata: Optional[dict[str, str]] = None

            @field_validator("name")
            @classmethod
            def validate_name(cls, v):
                if v and "/" in v:
                    raise ValueError("Name cannot contain /")
                return v

            @model_validator(mode="after")
            def validate_node_consistency(self):
                # Folders must have no size, files must have size
                if self.node_type == NodeType.FOLDER and self.size_bytes is not None:
                    raise ValueError("Folders cannot have size_bytes")
                if self.node_type == NodeType.FILE and self.children:
                    raise ValueError("Files cannot have children")
                if self.node_type == NodeType.SYMLINK and not self.symlink_target:
                    raise ValueError("Symlinks must have a target")
                return self

        FileSystemNode.model_rebuild()

        # Should not raise RecursionError
        PartialFS = Partial[FileSystemNode]
        TruePartial = _partial_api(PartialFS).get_partial_model()

        # Complex nested structure
        data = {
            "name": "root",
            "node_type": "folder",
            "permissions": [{"user": "admin", "level": "admin"}],
            "metadata": {"created": "2024-01-01"},
            "children": [
                {
                    "name": "documents",
                    "node_type": "folder",
                    "children": [
                        {
                            "name": "report.pdf",
                            "node_type": "file",
                            "size_bytes": 1024,
                            "permissions": [{"user": "alice", "level": "read"}],
                        },
                        {
                            "name": "data",
                            "node_type": "folder",
                            "children": [
                                {
                                    "name": "archive.zip",
                                    "node_type": "file",
                                    "size_bytes": 2048,
                                }
                            ],
                        },
                    ],
                },
                {
                    "name": "shortcut",
                    "node_type": "symlink",
                    "symlink_target": {
                        "name": "target_file",
                        "node_type": "file",
                        "size_bytes": 512,
                    },
                },
            ],
        }

        result = TruePartial.model_validate(data)
        assert result.name == "root"
        assert result.node_type == NodeType.FOLDER
        assert len(result.children) == 2
        assert result.children[0].name == "documents"
        assert len(result.children[0].children) == 2
        assert result.children[0].children[0].name == "report.pdf"
        assert result.children[0].children[0].size_bytes == 1024
        assert result.children[0].children[1].children[0].name == "archive.zip"
        assert result.children[1].symlink_target.name == "target_file"
        assert result.permissions[0].level == "admin"

    def test_recursive_with_union_types(self):
        """Recursive model with Union types containing self-references."""
        from typing import Union

        class TextBlock(BaseModel):
            text: str

        class Container(BaseModel):
            title: str
            content: list[Union[TextBlock, "Container"]]

        Container.model_rebuild()

        PartialContainer = Partial[Container]
        TruePartial = _partial_api(PartialContainer).get_partial_model()

        data = {
            "title": "Chapter 1",
            "content": [
                {"text": "Introduction paragraph"},
                {
                    "title": "Section 1.1",
                    "content": [
                        {"text": "Section text"},
                        {
                            "title": "Subsection 1.1.1",
                            "content": [{"text": "Deep nested text"}],
                        },
                    ],
                },
                {"text": "Closing paragraph"},
            ],
        }

        result = TruePartial.model_validate(data)
        assert result.title == "Chapter 1"
        assert len(result.content) == 3
        assert result.content[0].text == "Introduction paragraph"
        assert result.content[1].title == "Section 1.1"
        assert result.content[1].content[1].title == "Subsection 1.1.1"


class TestOptionalNestedBaseModelDuringPartialStreaming:
    """Regression tests for Optional[BaseModel] field handling during partial streaming.

    When the root JSON is incomplete (stream still in flight), _build_partial_object()
    is called instead of model_validate().  Fields annotated as ``Optional[NestedModel]``
    (equivalently ``Union[NestedModel, None]``) used to be stored as raw ``dict`` objects
    because the guard ``isinstance(field_type, type)`` returns False for generic aliases
    like ``Optional[NestedModel]``.

    The fix adds ``_unwrap_optional_base_model()`` which unwraps the Optional wrapper, so
    callers get proper model instances with attribute access instead of plain dicts.
    """

    def test_optional_nested_basemodel_is_model_instance_during_incomplete_stream(self):
        """Optional[NestedModel] fields must yield NestedModel instances, not raw dicts."""

        class Inner(BaseModel):
            value: int

        class Outer(BaseModel):
            name: str
            inner: Optional[Inner] = None

        partial = Partial[Outer]

        # Incomplete JSON — root closing brace is missing, so _build_partial_object runs
        chunks = ['{"name": "alice", "inner": {"value": 42}']

        results = list(_partial_api(partial).model_from_chunks(chunks))
        assert len(results) == 1
        obj = results[0]

        # Before the fix: obj.inner was a plain dict {"value": 42}
        assert isinstance(obj.inner, Inner), (
            f"Expected Inner instance, got {type(obj.inner)}: {obj.inner!r}"
        )
        assert obj.inner.value == 42

    def test_optional_nested_basemodel_attribute_access_does_not_raise(self):
        """Attribute access on an Optional[BaseModel] field must not raise AttributeError."""

        class Address(BaseModel):
            city: str
            zip_code: str

        class Person(BaseModel):
            name: str
            address: Optional[Address] = None

        partial = Partial[Person]

        # Root still open — simulates mid-stream state
        chunks = ['{"name": "bob", "address": {"city": "NYC", "zip_code": "10001"}']

        results = list(_partial_api(partial).model_from_chunks(chunks))
        obj = results[-1]

        # Would raise AttributeError on a raw dict before the fix
        assert obj.address.city == "NYC"
        assert obj.address.zip_code == "10001"

    def test_optional_nested_basemodel_absent_field_stays_none(self):
        """When the Optional[BaseModel] field is absent, its default (None) must not change."""

        class Inner(BaseModel):
            value: int

        class Outer(BaseModel):
            name: str
            inner: Optional[Inner] = None

        partial = Partial[Outer]

        # inner field completely absent; root also incomplete
        chunks = ['{"name": "carol"']

        results = list(_partial_api(partial).model_from_chunks(chunks))
        obj = results[-1]

        # Must remain None — not an empty Inner() instance
        assert obj.inner is None

    def test_partial_list_returns_model_instance_for_incomplete_trailing_item(self):
        """The trailing (still-streaming) item in a list[BaseModel] field must be
        a partial model instance, not a raw dict -- consistent with how a
        singular open nested BaseModel field is already handled."""

        class Item(BaseModel):
            name: str
            qty: int

        class Order(BaseModel):
            items: list[Item]

        partial = Partial[Order]

        # First item complete, second item mid-stream (incomplete).
        chunks = ['{"items": [{"name": "apple", "qty": 3}, {"name": "banana", "qty"']

        results = list(_partial_api(partial).model_from_chunks(chunks))
        obj = results[-1]

        assert isinstance(obj.items[0], Item)
        # Would be a raw dict before the fix, raising AttributeError on .name
        assert isinstance(obj.items[-1], Item)
        assert obj.items[-1].name == "banana"


def test_partial_model_builds_models_inside_mapping_values():
    """Model values held in a mapping must become models, not stay plain dicts.

    Regression test: a dict[str, Model] field, and a mapping nested inside a sequence,
    only resolved a model when the field's own annotation was a model, so their values
    stayed plain dicts while streaming and attribute access on them raised
    AttributeError.
    """

    class Item(BaseModel):
        name: str
        qty: int

    class Cart(BaseModel):
        mapping: dict[str, Item]
        nested: list[dict[str, Item]]

    partial = Partial[Cart].get_partial_model()
    chunks = [
        '{"mapping": {"a": {"name": "apple", "qty": 3}, "b": {"name": "ban',
        'ana", "qty": 4}}, "nested": [{"c": {"name": "cherry", "qty": 5}, "d": {"name": "dam',
    ]

    results = list(_partial_api(partial).model_from_chunks(chunks))
    obj = results[-1]

    # A completed value is validated, and one still streaming is a partial model.
    assert isinstance(obj.mapping["a"], Item)
    assert obj.mapping["a"].name == "apple"
    assert obj.mapping["a"].qty == 3
    assert isinstance(obj.mapping["b"], Item)
    assert obj.mapping["b"].name == "banana"

    # A mapping inside a list item behaves the same way.
    assert isinstance(obj.nested[0]["c"], Item)
    assert obj.nested[0]["c"].name == "cherry"
    assert isinstance(obj.nested[0]["d"], Item)
    assert obj.nested[0]["d"].name == "dam"
