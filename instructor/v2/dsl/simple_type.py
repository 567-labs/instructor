from __future__ import annotations
from inspect import isclass
import types
import typing
from pydantic import BaseModel, create_model
from enum import Enum

from instructor.v2.dsl.partial import Partial

T = typing.TypeVar("T")


class AdapterBase(BaseModel):
    pass


class ModelAdapter(typing.Generic[T]):
    """
    Accepts a response model and returns a BaseModel with the response model as the content.
    """

    def __class_getitem__(cls, response_model: type[BaseModel]) -> type[BaseModel]:
        # Import at runtime to avoid circular import
        from instructor.v2.core.function_calls import ResponseSchema

        assert is_simple_type(response_model), "Only simple types are supported"
        return create_model(
            "Response",
            content=(response_model, ...),
            __doc__="Correctly Formatted and Extracted Response.",
            __base__=(AdapterBase, ResponseSchema),
        )


def validateIsSubClass(response_model: type):
    """
    Temporary guard against issues with generics in Python 3.9
    """
    import sys

    if sys.version_info < (3, 10):
        if len(typing.get_args(response_model)) == 0:
            return False
        return issubclass(typing.get_args(response_model)[0], BaseModel)
    try:
        # Add a guard here to prevent issues with GenericAlias
        if isinstance(response_model, types.GenericAlias):
            return False
    except Exception:
        pass

    return issubclass(response_model, BaseModel)


def is_simple_type(
    response_model: type[BaseModel] | str | int | float | bool | typing.Any,
) -> bool:
    # ! we're getting mixes between classes and instances due to how we handle some
    # ! response model types, we should fix this in later PRs

    try:
        if isclass(response_model) and validateIsSubClass(response_model):
            return False
    except TypeError:
        # ! In versions < 3.11, typing.Iterable is not a class, so we can't use isclass
        # ! for now if `response_model` is an Iterable isclass and issubclass will raise
        # ! TypeError, so we need to check if `response_model` is an Iterable
        # ! This is a workaround for now, we should fix this in later PRs
        return False

    # Get the origin of the response model
    origin = typing.get_origin(response_model)

    # Handle special case for list[int | str], list[Union[int, str]] or similar type patterns
    # Identify a list type by checking for various origins it might have
    if origin in {typing.Iterable, Partial, list}:
        # For list types, check the contents before deciding
        if origin is list:
            # Extract the inner types from the list
            args = typing.get_args(response_model)
            if args and len(args) == 1:
                inner_arg = args[0]
                # Special handling for Union types
                inner_origin = typing.get_origin(inner_arg)

                # Explicit check for Union types - try different patterns across Python versions
                if (
                    inner_origin is typing.Union
                    or inner_origin == typing.Union
                    or str(inner_origin) == "typing.Union"
                    or str(type(inner_arg)) == "<class 'typing._UnionGenericAlias'>"
                ):
                    return True

                # Check if inner type is a BaseModel - if so, not a simple type
                try:
                    if isclass(inner_arg) and issubclass(inner_arg, BaseModel):
                        return False
                except TypeError:
                    pass

                # Check for Python 3.10+ pipe syntax (str | int creates types.UnionType;
                # plain classes also have __or__ since Python 3.10, so check the type)
                if hasattr(types, "UnionType") and isinstance(inner_arg, types.UnionType):
                    return True

                # For simple list with basic types, also return True
                if inner_arg in {str, int, float, bool}:
                    return True

            # If no args or unknown pattern, treat as simple list
            return len(args) == 0

        # Extract the inner types from the list for other iterable types
        args = typing.get_args(response_model)
        if args and len(args) == 1:
            inner_arg = args[0]
            # Special handling for Union types
            inner_origin = typing.get_origin(inner_arg)

            # Explicit check for Union types - try different patterns across Python versions
            if (
                inner_origin is typing.Union
                or inner_origin == typing.Union
                or str(inner_origin) == "typing.Union"
                or str(type(inner_arg)) == "<class 'typing._UnionGenericAlias'>"
            ):
                return True

            # Check for Python 3.10+ pipe syntax (str | int creates types.UnionType;
            # plain classes also have __or__ since Python 3.10, so check the type)
            if hasattr(types, "UnionType") and isinstance(inner_arg, types.UnionType):
                return True

            # For simple list with basic types, also return True
            if inner_arg in {str, int, float, bool}:
                return True

        # For other iterable patterns, return False (e.g., streaming types)
        return False

    if response_model in {
        str,
        int,
        float,
        bool,
    }:
        return True

    # If the response_model is a simple type like annotated
    if origin in {
        typing.Annotated,
        typing.Literal,
        typing.Union,
        list,  # origin of List[T] is list
    }:
        return True

    if isclass(response_model) and issubclass(response_model, Enum):
        return True

    return False
