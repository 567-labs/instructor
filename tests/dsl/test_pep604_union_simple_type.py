"""Comprehensive regression test suite for PEP 604 union syntax in simple types.

Verifies that Python 3.10+ pipe-union syntax (types.UnionType, e.g. `int | str`,
`str | None`, `User | Org`) is recognized as a simple type by `is_simple_type`,
instantiable via `ModelAdapter`, and correctly normalized by `prepare_response_model`.
"""

from __future__ import annotations

import sys
from typing import Annotated, Literal, Union

import pytest
from pydantic import BaseModel, Field, ValidationError

from instructor.v2.dsl.simple_type import AdapterBase, ModelAdapter, is_simple_type
from instructor.v2.core.function_calls import ResponseSchema
from instructor.v2.core.response_model import prepare_response_model


class User(BaseModel):
    name: str


class Org(BaseModel):
    title: str


class Guest(BaseModel):
    guest_id: int


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="PEP 604 pipe union syntax requires Python 3.10+",
)
class TestPEP604SimpleTypeIdentification:
    """Test suite verifying is_simple_type correctly classifies PEP 604 unions."""

    @pytest.mark.parametrize(
        ("pipe_type", "typing_union_equiv"),
        [
            (int | str, Union[int, str]),
            (str | float, Union[str, float]),
            (bool | int, Union[bool, int]),
            (float | bool, Union[float, bool]),
        ],
    )
    def test_primitive_pipe_unions(self, pipe_type, typing_union_equiv) -> None:
        """Primitive pipe unions must be identified as simple types identically to typing.Union."""
        assert is_simple_type(pipe_type) is True
        assert is_simple_type(typing_union_equiv) is True
        assert is_simple_type(pipe_type) == is_simple_type(typing_union_equiv)

    @pytest.mark.parametrize(
        ("pipe_type", "typing_union_equiv"),
        [
            (str | None, Union[str, None]),
            (int | None, Union[int, None]),
            (float | None, Union[float, None]),
            (bool | None, Union[bool, None]),
        ],
    )
    def test_optional_pipe_unions(self, pipe_type, typing_union_equiv) -> None:
        """Optional pipe unions (T | None) must be identified as simple types."""
        assert is_simple_type(pipe_type) is True
        assert is_simple_type(typing_union_equiv) is True
        assert is_simple_type(pipe_type) == is_simple_type(typing_union_equiv)

    @pytest.mark.parametrize(
        ("pipe_type", "typing_union_equiv"),
        [
            (int | str | float, Union[int, str, float]),
            (int | str | bool | float, Union[int, str, bool, float]),
            (str | int | None, Union[str, int, None]),
        ],
    )
    def test_multi_member_pipe_unions(self, pipe_type, typing_union_equiv) -> None:
        """Multi-member pipe unions (3+ types) must be identified as simple types."""
        assert is_simple_type(pipe_type) is True
        assert is_simple_type(typing_union_equiv) is True
        assert is_simple_type(pipe_type) == is_simple_type(typing_union_equiv)

    def test_literal_pipe_unions(self) -> None:
        """Unions combining Literal types via pipe syntax must be simple types."""
        pipe_literal = Literal["red", "green"] | Literal["blue", "yellow"]
        typing_literal = Union[Literal["red", "green"], Literal["blue", "yellow"]]

        assert is_simple_type(pipe_literal) is True
        assert is_simple_type(typing_literal) is True
        assert is_simple_type(pipe_literal) == is_simple_type(typing_literal)

    def test_annotated_pipe_unions(self) -> None:
        """Unions containing Annotated types via pipe syntax must be simple types."""
        pipe_annotated = Annotated[int, Field(gt=0)] | str
        typing_annotated = Union[Annotated[int, Field(gt=0)], str]

        assert is_simple_type(pipe_annotated) is True
        assert is_simple_type(typing_annotated) is True
        assert is_simple_type(pipe_annotated) == is_simple_type(typing_annotated)

    def test_model_pipe_unions(self) -> None:
        """Unions of BaseModel classes via pipe syntax must be simple types."""
        pipe_models = User | Org
        typing_models = Union[User, Org]

        assert is_simple_type(pipe_models) is True
        assert is_simple_type(typing_models) is True
        assert is_simple_type(pipe_models) == is_simple_type(typing_models)

    def test_three_way_model_pipe_unions(self) -> None:
        """Three-way model pipe unions must be simple types."""
        pipe_models = User | Org | Guest
        typing_models = Union[User, Org, Guest]

        assert is_simple_type(pipe_models) is True
        assert is_simple_type(typing_models) is True
        assert is_simple_type(pipe_models) == is_simple_type(typing_models)


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="PEP 604 pipe union syntax requires Python 3.10+",
)
class TestPEP604ModelAdapter:
    """Test suite verifying ModelAdapter behavior with PEP 604 pipe unions."""

    def test_model_adapter_creation_primitive_pipe(self) -> None:
        """ModelAdapter[int | str] produces a valid Response model."""
        adapter = ModelAdapter[int | str]
        assert issubclass(adapter, AdapterBase)
        assert issubclass(adapter, ResponseSchema)
        assert "content" in adapter.model_fields

    def test_model_adapter_parsing_values(self) -> None:
        """ModelAdapter[int | str] accepts valid values for either union branch."""
        adapter = ModelAdapter[int | str]

        instance_int = adapter(content=42)
        assert instance_int.content == 42
        assert isinstance(instance_int.content, int)

        instance_str = adapter(content="hello")
        assert instance_str.content == "hello"
        assert isinstance(instance_str.content, str)

    def test_model_adapter_rejects_invalid_values(self) -> None:
        """ModelAdapter[int | str] rejects values that do not match any union branch."""
        adapter = ModelAdapter[int | str]
        with pytest.raises(ValidationError):
            adapter(content=[1, 2, 3])

        with pytest.raises(ValidationError):
            adapter(content={"key": "val"})

    def test_model_adapter_optional_pipe(self) -> None:
        """ModelAdapter[str | None] correctly supports None and str."""
        adapter = ModelAdapter[str | None]

        instance_none = adapter(content=None)
        assert instance_none.content is None

        instance_str = adapter(content="valid")
        assert instance_str.content == "valid"

    def test_model_adapter_model_union(self) -> None:
        """ModelAdapter[User | Org] parses either BaseModel instance."""
        adapter = ModelAdapter[User | Org]

        user_inst = adapter(content=User(name="Alice"))
        assert isinstance(user_inst.content, User)
        assert user_inst.content.name == "Alice"

        org_inst = adapter(content={"title": "Acme Corp"})
        assert isinstance(org_inst.content, Org)
        assert org_inst.content.title == "Acme Corp"

    def test_model_adapter_json_schema_matches_typing_union(self) -> None:
        """ModelAdapter[int | str] generates the same schema structure as typing.Union."""
        pipe_adapter = ModelAdapter[int | str]
        typing_adapter = ModelAdapter[Union[int, str]]

        pipe_schema = pipe_adapter.model_json_schema()
        typing_schema = typing_adapter.model_json_schema()

        assert (
            pipe_schema["properties"]["content"]
            == typing_schema["properties"]["content"]
        )


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="PEP 604 pipe union syntax requires Python 3.10+",
)
class TestPEP604PrepareResponseModel:
    """Test suite verifying prepare_response_model handles PEP 604 unions without crashing."""

    def test_prepare_primitive_pipe(self) -> None:
        """prepare_response_model(int | str) returns an AdapterBase response model."""
        prepared = prepare_response_model(int | str)
        assert prepared is not None
        assert issubclass(prepared, AdapterBase)
        assert issubclass(prepared, ResponseSchema)
        assert "content" in prepared.model_fields

    def test_prepare_optional_pipe(self) -> None:
        """prepare_response_model(str | None) returns an AdapterBase response model."""
        prepared = prepare_response_model(str | None)
        assert prepared is not None
        assert issubclass(prepared, AdapterBase)

    def test_prepare_model_pipe(self) -> None:
        """prepare_response_model(User | Org) returns an AdapterBase response model."""
        prepared = prepare_response_model(User | Org)
        assert prepared is not None
        assert issubclass(prepared, AdapterBase)

    def test_prepare_list_of_primitive_pipe(self) -> None:
        """prepare_response_model(list[int | str]) wraps as simple type adapter."""
        prepared = prepare_response_model(list[int | str])
        assert prepared is not None
        assert issubclass(prepared, AdapterBase)

    def test_prepare_list_of_model_pipe_remains_iterable(self) -> None:
        """prepare_response_model(list[User | Org]) routes to IterableBase, not AdapterBase."""
        from instructor.v2.dsl.iterable import IterableBase

        prepared = prepare_response_model(list[User | Org])
        assert prepared is not None
        assert issubclass(prepared, IterableBase)
        assert not issubclass(prepared, AdapterBase)


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="PEP 604 pipe union syntax requires Python 3.10+",
)
class TestPEP604ResponseProcessingUnwrap:
    """End-to-end unwrap simulation for responses generated from PEP 604 response models."""

    def test_unwrap_integer_from_adapter(self) -> None:
        """Parsed adapter returns the inner int value upon unwrapping."""
        prepared = prepare_response_model(int | str)
        instance = prepared.model_validate_json('{"content": 42}')
        assert isinstance(instance, AdapterBase)
        assert instance.content == 42
        assert type(instance.content) is int

    def test_unwrap_string_from_adapter(self) -> None:
        """Parsed adapter returns the inner str value upon unwrapping."""
        prepared = prepare_response_model(int | str)
        instance = prepared.model_validate_json('{"content": "hello"}')
        assert isinstance(instance, AdapterBase)
        assert instance.content == "hello"
        assert type(instance.content) is str

    def test_unwrap_none_from_optional_adapter(self) -> None:
        """Parsed adapter returns None for optional union."""
        prepared = prepare_response_model(str | None)
        instance = prepared.model_validate_json('{"content": null}')
        assert isinstance(instance, AdapterBase)
        assert instance.content is None

    def test_unwrap_model_from_adapter(self) -> None:
        """Parsed adapter returns the inner User model."""
        prepared = prepare_response_model(User | Org)
        instance = prepared.model_validate_json('{"content": {"name": "Bob"}}')
        assert isinstance(instance, AdapterBase)
        assert isinstance(instance.content, User)
        assert instance.content.name == "Bob"
