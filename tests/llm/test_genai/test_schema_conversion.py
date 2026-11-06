"""Test schema conversion functions for Gemini."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel

from instructor.v2.providers.gemini.utils import (
    map_to_gemini_function_schema,
    verify_no_unions,
)


class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SimpleModel(BaseModel):
    name: str
    age: int
    is_active: bool


class OptionalModel(BaseModel):
    name: str
    age: Optional[int] = None
    description: Optional[str] = None


class EnumModel(BaseModel):
    name: str
    priority: Priority


class NestedModel(BaseModel):
    name: str
    items: list[str]
    details: SimpleModel


def test_simple_schema_conversion():
    """Test conversion strips extra pydantic fields like 'title'."""
    schema = SimpleModel.model_json_schema()
    result = map_to_gemini_function_schema(schema)

    # Input has 'title' fields that should be stripped out
    assert "title" in schema
    assert "title" in schema["properties"]["name"]

    # Output should be clean without title fields
    expected = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "is_active": {"type": "boolean"},
        },
        "required": ["name", "age", "is_active"],
    }

    assert result == expected


def test_optional_schema_conversion():
    """Test conversion transforms anyOf[T, null] to nullable fields."""
    schema = OptionalModel.model_json_schema()
    result = map_to_gemini_function_schema(schema)

    # Input should have anyOf with null type for optional fields
    assert schema["properties"]["age"]["anyOf"] == [
        {"type": "integer"},
        {"type": "null"},
    ]
    assert schema["properties"]["description"]["anyOf"] == [
        {"type": "string"},
        {"type": "null"},
    ]

    # Output should convert to nullable: true
    expected = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "nullable": True},
            "description": {"type": "string", "nullable": True},
        },
        "required": ["name"],
    }

    assert result == expected


def test_enum_schema_conversion():
    """Test conversion resolves $refs and adds format: enum."""
    schema = EnumModel.model_json_schema()
    result = map_to_gemini_function_schema(schema)

    # Input should have $ref and $defs
    assert schema["properties"]["priority"]["$ref"] == "#/$defs/Priority"
    assert "$defs" in schema
    assert schema["$defs"]["Priority"]["enum"] == ["low", "medium", "high"]

    # Output should resolve the ref and add format: enum
    expected = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "priority": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "format": "enum",
            },
        },
        "required": ["name", "priority"],
    }

    assert result == expected


def test_nested_schema_conversion():
    """Test conversion of schema with nested objects."""
    schema = NestedModel.model_json_schema()
    result = map_to_gemini_function_schema(schema)

    expected = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "items": {"type": "array", "items": {"type": "string"}},
            "details": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "is_active": {"type": "boolean"},
                },
                "required": ["name", "age", "is_active"],
            },
        },
        "required": ["name", "items", "details"],
    }

    assert result == expected


def test_verify_no_unions_valid():
    """Test verify_no_unions with valid schemas."""
    # Simple schema should pass
    simple_schema = SimpleModel.model_json_schema()
    assert verify_no_unions(simple_schema) is True

    # Optional schema should pass (Optional[T] is Union[T, None])
    optional_schema = OptionalModel.model_json_schema()
    assert verify_no_unions(optional_schema) is True


def test_verify_no_unions_invalid():
    """Test verify_no_unions with union schemas (now allowed)."""
    # Create a schema with a true union (not just Optional)
    invalid_schema = {
        "type": "object",
        "properties": {"value": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
    }
    assert verify_no_unions(invalid_schema) is True


def test_schema_without_refs():
    """Test schema conversion without $refs."""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "count": {"type": "integer"}},
        "required": ["name"],
    }

    result = map_to_gemini_function_schema(schema)

    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "count": {"type": "integer"}},
        "required": ["name"],
    }

    assert result == expected


def test_schema_with_description():
    """Test schema conversion preserves descriptions."""
    schema = {
        "type": "object",
        "description": "A test object",
        "properties": {"name": {"type": "string", "description": "The name field"}},
    }

    result = map_to_gemini_function_schema(schema)

    expected = {
        "type": "object",
        "description": "A test object",
        "properties": {"name": {"type": "string", "description": "The name field"}},
    }

    assert result == expected


def test_union_type_raises_error():
    """Test that union types are allowed in schema conversion."""
    # Create a model with a true union type (not Optional or Decimal)
    union_schema = {
        "type": "object",
        "properties": {"value": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
    }

    result = map_to_gemini_function_schema(union_schema)
    assert result["properties"]["value"]["anyOf"] == [
        {"type": "string"},
        {"type": "integer"},
    ]


def test_verify_no_unions_allows_optional():
    """Test that verify_no_unions allows Optional types."""
    # Schema with Optional field (Union with null)
    optional_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        },
    }

    assert verify_no_unions(optional_schema) is True


def test_verify_no_unions_allows_decimal():
    """Test that verify_no_unions allows Decimal types (string | number)."""
    # Schema with Decimal field (Union of string and number)
    decimal_schema = {
        "type": "object",
        "properties": {
            "total": {"anyOf": [{"type": "number"}, {"type": "string"}]},
            "price": {
                "anyOf": [{"type": "string"}, {"type": "number"}]
            },  # Order shouldn't matter
        },
    }

    assert verify_no_unions(decimal_schema) is True


def test_verify_no_unions_rejects_other_unions():
    """Test that verify_no_unions allows non-Optional unions."""
    # Schema with unsupported union type (string | integer)
    union_schema = {
        "type": "object",
        "properties": {"value": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
    }

    assert verify_no_unions(union_schema) is True


def test_verify_no_unions_rejects_complex_unions():
    """Test that verify_no_unions allows complex union types."""
    # Schema with more than 2 types in union
    complex_union_schema = {
        "type": "object",
        "properties": {
            "value": {
                "anyOf": [{"type": "string"}, {"type": "integer"}, {"type": "boolean"}]
            }
        },
    }

    assert verify_no_unions(complex_union_schema) is True


def test_verify_no_unions_nested_schemas():
    """Test that verify_no_unions allows unions in nested schemas."""
    # Schema with nested object containing Decimal and Optional fields
    nested_schema = {
        "type": "object",
        "properties": {
            "receipt": {
                "type": "object",
                "properties": {
                    "total": {
                        "anyOf": [{"type": "number"}, {"type": "string"}]
                    },  # Decimal - should pass
                    "notes": {
                        "anyOf": [{"type": "string"}, {"type": "null"}]
                    },  # Optional - should pass
                },
            }
        },
    }

    assert verify_no_unions(nested_schema) is True

    # Schema with nested object containing unsupported union
    bad_nested_schema = {
        "type": "object",
        "properties": {
            "receipt": {
                "type": "object",
                "properties": {
                    "total": {
                        "anyOf": [{"type": "number"}, {"type": "string"}]
                    },  # Decimal - should pass
                    "status": {
                        "anyOf": [{"type": "string"}, {"type": "integer"}]
                    },  # Bad union - should fail
                },
            }
        },
    }

    assert verify_no_unions(bad_nested_schema) is True


def test_decimal_schema_conversion_succeeds():
    """Test that Decimal types (string | number) are successfully converted."""
    # Schema representing a Receipt with Decimal total field
    decimal_schema = {
        "type": "object",
        "title": "Receipt",
        "properties": {
            "total": {
                "anyOf": [{"type": "number"}, {"type": "string"}],
                "title": "Total",
            }
        },
        "required": ["total"],
    }

    # This should not raise an error now
    result = map_to_gemini_function_schema(decimal_schema)

    # The conversion should succeed and preserve the anyOf structure
    assert result["type"] == "object"
    assert result["properties"]["total"]["anyOf"] == [
        {"type": "number"},
        {"type": "string"},
    ]
    assert result["required"] == ["total"]
    # Title should be stripped out
    assert "title" not in result
    assert "title" not in result["properties"]["total"]
