from __future__ import annotations
from inspect import isclass
import typing
import types
from pydantic import BaseModel, TypeAdapter, create_model
from pydantic.errors import PydanticSchemaGenerationError
from enum import Enum

from instructor.v2.dsl.partial import Partial

T = typing.TypeVar("T")

if hasattr(types, "UnionType"):
    _UNION_ORIGINS = (typing.Union, types.UnionType)
else:  # pragma: no cover - Python 3.9 has no PEP 604 union type
    _UNION_ORIGINS = (typing.Union,)


def has_pydantic_schema(typehint: type) -> bool:
    """Whether pydantic can generate a schema for ``typehint``.

    Used to tell a custom class pydantic cannot describe (``list[MyClass]``, the
    #2613 reproducer) apart from the classes it can (``Enum``, ``date``,
    ``Decimal``, ``UUID``, dataclasses, typed dictionaries, or anything
    implementing ``__get_pydantic_core_schema__``).

    The probe runs at most twice per ``prepare_response_model`` call and costs a
    few microseconds against a ~200-550 µs call, so it is deliberately not cached:
    an ``lru_cache`` here would pin dynamically created classes alive for no gain.
    """
    try:
        TypeAdapter(typehint)
    except PydanticSchemaGenerationError:
        return False
    return True


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
        import types

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
                    inner_origin in _UNION_ORIGINS
                    or inner_origin is typing.Union
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

                # A member that is neither a class nor a typing construct (``None``,
                # ``"ForwardRef"``, ``3``) is not a simple type either. The old
                # ``hasattr(inner_arg, "__or__")`` probe covered this for free,
                # because those members are exactly the ones without ``__or__``.
                # ``None`` matters: ``prepare_response_model`` reports
                # ``list[None]`` with its own "must be parameterized" error.
                if not isclass(inner_arg) and inner_origin is None:
                    return False

                # A custom class pydantic cannot describe must not reach the content
                # adapter: it used to, and only blew up later with an opaque
                # ``PydanticSchemaGenerationError`` (#2613). Falling through lets the
                # iterable guard in ``prepare_response_model`` raise an actionable
                # ``TypeError`` instead.
                if isclass(inner_arg) and not has_pydantic_schema(inner_arg):
                    return False

                # Every remaining member keeps the content-adapter routing it had
                # before the ``hasattr(inner_arg, "__or__")`` probe was replaced:
                # scalars, enums, dates, decimals, UUIDs, dataclasses, typed
                # dictionaries, ``Annotated`` and ``Literal`` members, classes
                # implementing ``__get_pydantic_core_schema__``, and PEP 604 unions.
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
                inner_origin in _UNION_ORIGINS
                or inner_origin is typing.Union
                or inner_origin == typing.Union
                or str(inner_origin) == "typing.Union"
                or str(type(inner_arg)) == "<class 'typing._UnionGenericAlias'>"
            ):
                return True

            # This branch is only reachable when ``get_origin`` reports
            # ``typing.Iterable`` itself, which is the pre-3.9 spelling; on modern
            # runtimes ``Iterable[X]`` resolves to ``collections.abc.Iterable`` and
            # falls through to the bottom of the function. Keep the legacy
            # scalar-only rule so the behaviour is unchanged where it does apply.
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
