import sys
from collections.abc import Iterable
from enum import Enum
from typing import Annotated, Literal, Union, List, cast, get_origin, get_args  # noqa: UP035

import pytest
from pydantic import BaseModel, Field

from instructor.dsl import is_simple_type, Partial
from instructor.utils.core import prepare_response_model


# Basic types tests - using parameterization
@pytest.mark.parametrize("basic_type", [str, int, float, bool])
def test_standard_types(basic_type):
    """Test that standard Python types are identified as simple types."""
    assert is_simple_type(basic_type), f"Failed for type: {basic_type}"


def test_enum_simple():
    """Test that Enum types are identified as simple types."""

    class Color(Enum):
        RED = 1
        GREEN = 2
        BLUE = 3

    assert is_simple_type(Color), f"Failed for type: {Color}"


def test_base_model_not_simple():
    """Test that BaseModel types are NOT identified as simple types."""

    class MyModel(BaseModel):
        label: str

    assert not is_simple_type(MyModel), "BaseModel should not be a simple type"


def test_partial_not_simple():
    """Test that Partial types are NOT identified as simple types."""

    class SampleModel(BaseModel):
        data: int

    assert not is_simple_type(Partial[SampleModel]), (
        "Failed for type: Partial[SampleModel]"
    )


def test_annotated_simple():
    """Test that Annotated types are identified as simple types."""
    new_type = Annotated[int, Field(description="test")]

    assert is_simple_type(new_type), f"Failed for type: {new_type}"


def test_literal_simple():
    """Test that Literal types are identified as simple types."""
    new_type = Literal[1, 2, 3]

    assert is_simple_type(new_type), f"Failed for type: {new_type}"


def test_union_simple():
    """Test that Union types are identified as simple types."""
    new_type = Union[int, str]

    assert is_simple_type(new_type), f"Failed for type: {new_type}"


def test_iterable_not_simple():
    """Test that Iterable types are NOT identified as simple types."""
    new_type = Iterable[int]

    assert not is_simple_type(new_type), f"Failed for type: {new_type}"


def test_list_of_base_model_not_simple():
    """list[BaseModel] must route through iterable handling."""

    class Item(BaseModel):
        value: int

    assert not is_simple_type(list[Item])
    assert not is_simple_type(List[Item])  # noqa: UP006


def test_list_of_custom_class_not_simple():
    """A user-defined class is not a union just because ``type`` exposes ``__or__``."""

    class CustomClass:
        pass

    assert not is_simple_type(list[CustomClass])

    with pytest.raises(TypeError, match="iterable elements must be Pydantic models"):
        prepare_response_model(list[CustomClass])


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="Union pipe syntax is only available in Python 3.10+",
)
def test_list_with_union_pipe_syntax():
    """Test that list[int | str] is correctly identified as a simple type."""
    response_model = list[int | str]
    assert is_simple_type(response_model), (
        f"list[int | str] should be a simple type in Python {sys.version_info.major}.{sys.version_info.minor}. Instead it was identified as {type(response_model)} with origin {get_origin(response_model)} and args {get_args(response_model)}"
    )


def test_list_with_union_typing_syntax():
    """Test that List[Union[int, str]] is correctly identified as a simple type."""
    response_model = List[Union[int, str]]  # noqa: UP006
    assert is_simple_type(response_model), (
        f"List[Union[int, str]] should be a simple type in Python {sys.version_info.major}.{sys.version_info.minor}"
    )


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="Union pipe syntax is only available in Python 3.10+",
)
def test_prepare_response_model_with_list_union():
    """Test that list[int | str] works correctly as a response model with prepare_response_model."""
    # This is the type used in the fizzbuzz example
    response_model = list[int | str]

    # First check that it's correctly identified as a simple type
    assert is_simple_type(response_model), (
        f"list[int | str] should be a simple type in Python {sys.version_info.major}.{sys.version_info.minor}"
    )

    # Then check that prepare_response_model handles it correctly
    prepared_model = prepare_response_model(response_model)
    assert prepared_model is not None, (
        "prepare_response_model should not return None for list[int | str]"
    )


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="Union pipe syntax is only available in Python 3.10+",
)
def test_list_of_model_pipe_union_is_treated_as_iterable():
    from instructor.v2.dsl.iterable import IterableBase

    class A(BaseModel):
        x: int

    class B(BaseModel):
        y: str

    pipe = prepare_response_model(list[A | B])
    typing_union = prepare_response_model(List[Union[A, B]])  # noqa: UP006

    assert isinstance(typing_union, type) and issubclass(typing_union, IterableBase)
    assert isinstance(pipe, type) and issubclass(pipe, IterableBase)

    pipe_model = cast(type[BaseModel], pipe)
    pipe_schema = pipe_model.model_json_schema()
    assert "tasks" in pipe_schema["properties"]
    assert "content" not in pipe_schema["properties"]
    assert set(pipe_schema.get("$defs", {})) == {"A", "B"}


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="Union pipe syntax is only available in Python 3.10+",
)
def test_list_of_model_pipe_union_generates_clean_name():
    class A(BaseModel):
        x: int

    class B(BaseModel):
        y: str

    prepared = prepare_response_model(list[A | B])
    assert prepared is not None

    assert "|" not in prepared.__name__
    assert "." not in prepared.__name__
    assert prepared.__name__ == "IterableAOrB"


def test_list_of_model_typing_union_generates_clean_name():
    """`List[Union[A, B]]` must derive its Iterable name from the union members.

    Both `typing.Union[A, B]` and PEP 604 `A | B` expose ``__name__ == "Union"``, so a
    naive ``__name__`` lookup collapses every union to ``IterableUnion``. The name should
    instead be built from the member names (``IterableAOrB``).
    """

    class A(BaseModel):
        x: int

    class B(BaseModel):
        y: str

    prepared = prepare_response_model(List[Union[A, B]])  # noqa: UP006
    assert prepared is not None
    assert prepared.__name__ == "IterableAOrB"


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="Union pipe syntax is only available in Python 3.10+",
)
def test_list_of_model_three_way_union_generates_clean_name():
    """A union with more than two members joins every member name with ``Or``."""

    class A(BaseModel):
        x: int

    class B(BaseModel):
        y: str

    class C(BaseModel):
        z: float

    prepared = prepare_response_model(list[A | B | C])
    assert prepared is not None
    assert prepared.__name__ == "IterableAOrBOrC"
