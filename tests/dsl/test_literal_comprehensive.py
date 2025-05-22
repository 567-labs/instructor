"""Comprehensive tests for Literal types with Partial"""

from pydantic import BaseModel, ValidationError
from typing import Literal, Optional, Union
from instructor.dsl.partial import Partial
import pytest


class TestLiteralPartialHandling:
    """Test suite for Literal type handling in Partial models"""

    def test_required_literal_basic(self):
        """Test basic required literal field behavior"""

        class Model(BaseModel):
            status: Literal["active", "inactive", "pending"]

        partial_model = Partial[Model].get_partial_model()

        # Should accept partial strings during streaming
        assert partial_model.model_validate({"status": "a"}).status == "a"
        assert partial_model.model_validate({"status": "ac"}).status == "ac"
        assert partial_model.model_validate({"status": "active"}).status == "active"

        # Should NOT accept None for required field
        with pytest.raises(ValidationError):
            partial_model.model_validate({"status": None})

        # Empty object should default to empty string
        assert partial_model.model_validate({}).status == ""

    def test_optional_literal_basic(self):
        """Test basic optional literal field behavior"""

        class Model(BaseModel):
            status: Optional[Literal["active", "inactive", "pending"]]

        partial_model = Partial[Model].get_partial_model()

        # Should accept partial strings during streaming
        assert partial_model.model_validate({"status": "a"}).status == "a"
        assert partial_model.model_validate({"status": "ac"}).status == "ac"
        assert partial_model.model_validate({"status": "active"}).status == "active"

        # SHOULD accept None for optional field
        assert partial_model.model_validate({"status": None}).status is None

        # Empty object should default to None for optional
        assert partial_model.model_validate({}).status is None

    def test_literal_union_with_none(self):
        """Test Literal in Union with None (another way to express Optional)"""

        class Model(BaseModel):
            status: Union[Literal["active", "inactive"], None]

        partial_model = Partial[Model].get_partial_model()

        # Should behave like Optional
        assert partial_model.model_validate({"status": None}).status is None
        assert partial_model.model_validate({}).status is None
        assert partial_model.model_validate({"status": "a"}).status == "a"

    def test_final_validation_required(self):
        """Test that final validation works correctly for required literals"""

        class Model(BaseModel):
            color: Literal["red", "green", "blue"]

        partial_model = Partial[Model].get_partial_model()

        # Partial accepts invalid literal
        partial_result = partial_model.model_validate({"color": "yellow"})
        assert partial_result.color == "yellow"

        # But final validation should fail
        with pytest.raises(ValidationError) as exc_info:
            Model.model_validate(partial_result.model_dump())
        assert "literal_error" in str(exc_info.value)

        # Valid literal should work end-to-end
        partial_valid = partial_model.model_validate({"color": "red"})
        final_valid = Model.model_validate(partial_valid.model_dump())
        assert final_valid.color == "red"

    def test_final_validation_optional(self):
        """Test that final validation works correctly for optional literals"""

        class Model(BaseModel):
            color: Optional[Literal["red", "green", "blue"]]

        partial_model = Partial[Model].get_partial_model()

        # Test None preservation
        partial_none = partial_model.model_validate({"color": None})
        final_none = Model.model_validate(partial_none.model_dump())
        assert final_none.color is None

        # Test invalid literal
        partial_invalid = partial_model.model_validate({"color": "yellow"})
        with pytest.raises(ValidationError):
            Model.model_validate(partial_invalid.model_dump())

        # Test valid literal
        partial_valid = partial_model.model_validate({"color": "blue"})
        final_valid = Model.model_validate(partial_valid.model_dump())
        assert final_valid.color == "blue"

    def test_streaming_simulation(self):
        """Simulate realistic streaming scenarios"""

        class User(BaseModel):
            name: str
            role: Literal["admin", "user", "guest"]
            status: Optional[Literal["active", "inactive"]]

        partial_model = Partial[User].get_partial_model()

        # Simulate streaming "admin" for role
        streaming_chunks = [
            '{"name": "John", "role": ""}',
            '{"name": "John", "role": "a"}',
            '{"name": "John", "role": "ad"}',
            '{"name": "John", "role": "adm"}',
            '{"name": "John", "role": "admi"}',
            '{"name": "John", "role": "admin"}',
            '{"name": "John", "role": "admin", "status": "active"}',
        ]

        for chunk in streaming_chunks:
            result = partial_model.model_validate_json(chunk)
            # All chunks should validate successfully
            assert result.name == "John"
            assert isinstance(result.role, str)  # Can be partial string

        # Final validation
        final_data = partial_model.model_validate_json(streaming_chunks[-1])
        final_user = User.model_validate(final_data.model_dump())
        assert final_user.role == "admin"
        assert final_user.status == "active"

    def test_complex_nested_literals(self):
        """Test complex models with nested literal fields"""

        class Address(BaseModel):
            country: Literal["US", "UK", "CA"]
            state: Optional[str]

        class User(BaseModel):
            name: str
            role: Literal["admin", "user"]
            address: Address
            preferred_contact: Optional[Literal["email", "phone", "mail"]]

        partial_model = Partial[User].get_partial_model()

        # Test partial data
        data = {
            "name": "Alice",
            "role": "adm",  # Partial
            "address": {
                "country": "U",  # Partial
                "state": "NY",
            },
            "preferred_contact": "em",  # Partial
        }

        result = partial_model.model_validate(data)
        assert result.role == "adm"
        assert result.address.country == "U"  # type: ignore
        assert result.preferred_contact == "em"

    def test_literal_with_single_value(self):
        """Test Literal with single value"""

        class Model(BaseModel):
            version: Literal["1.0"]
            optional_flag: Optional[Literal[True]]

        partial_model = Partial[Model].get_partial_model()

        # Even single literals should accept partial strings
        assert partial_model.model_validate({"version": "1"}).version == "1"
        assert partial_model.model_validate({"version": "1."}).version == "1."
        assert partial_model.model_validate({"version": "1.0"}).version == "1.0"

        # Optional literal with boolean
        assert partial_model.model_validate({}).optional_flag is None
        assert (
            partial_model.model_validate({"optional_flag": True}).optional_flag is True
        )

    def test_schema_generation(self):
        """Test that schemas are generated correctly"""

        class Model(BaseModel):
            required_lit: Literal["a", "b", "c"]
            optional_lit: Optional[Literal["x", "y", "z"]]

        partial_model = Partial[Model].get_partial_model()
        schema = partial_model.model_json_schema()

        # Check required literal schema
        required_schema = schema["properties"]["required_lit"]
        assert "anyOf" in required_schema
        assert any(
            opt.get("enum") == ["a", "b", "c"] for opt in required_schema["anyOf"]
        )
        assert any(opt.get("type") == "string" for opt in required_schema["anyOf"])
        assert not any(opt.get("type") == "null" for opt in required_schema["anyOf"])
        assert required_schema.get("default") == ""

        # Check optional literal schema
        optional_schema = schema["properties"]["optional_lit"]
        assert "anyOf" in optional_schema
        assert any(
            opt.get("enum") == ["x", "y", "z"] for opt in optional_schema["anyOf"]
        )
        assert any(opt.get("type") == "string" for opt in optional_schema["anyOf"])
        assert any(opt.get("type") == "null" for opt in optional_schema["anyOf"])
        assert optional_schema.get("default") is None
