"""Response-model preparation helpers owned by the v2 runtime."""

from __future__ import annotations

import inspect
import sys
from collections.abc import Iterable
from typing import Any, Callable, TypeVar, Union, cast, get_args, get_origin

from pydantic import BaseModel, create_model
from typing_extensions import NotRequired, Required
from typing import get_type_hints

T = TypeVar("T")

_create_dynamic_model = cast(Callable[..., type[BaseModel]], create_model)

if sys.version_info >= (3, 10):
    from types import UnionType

    _UNION_ORIGINS: tuple[Any, ...] = (Union, UnionType)
else:  # pragma: no cover - Python 3.9 has no runtime ``X | Y`` syntax
    _UNION_ORIGINS = (Union,)


def is_typed_dict(cls: Any) -> bool:
    return (
        isinstance(cls, type)
        and issubclass(cls, dict)
        and hasattr(cls, "__annotations__")
    )


def _typed_dict_to_model(typed_dict: type[Any]) -> type[BaseModel]:
    """Convert a TypedDict while preserving per-key requiredness."""
    annotations = get_type_hints(typed_dict, include_extras=True)
    required_keys = set(getattr(typed_dict, "__required_keys__", set()))
    optional_keys = set(getattr(typed_dict, "__optional_keys__", set()))
    total = getattr(typed_dict, "__total__", True)
    fields: dict[str, tuple[Any, Any]] = {}

    for name, annotation in annotations.items():
        annotation_origin = get_origin(annotation)
        if annotation_origin is Required:
            field_annotation = get_args(annotation)[0]
            is_required = True
        elif annotation_origin is NotRequired:
            field_annotation = get_args(annotation)[0]
            is_required = False
        else:
            field_annotation = annotation
            is_required = name in required_keys or (name not in optional_keys and total)

        fields[name] = (field_annotation, ... if is_required else None)

    return _create_dynamic_model(
        getattr(typed_dict, "__name__", "TypedDictModel"),
        **fields,
    )


def is_simple_type(typehint: type[T]) -> bool:
    from instructor.v2.dsl.simple_type import is_simple_type as _is_simple_type

    return _is_simple_type(typehint)


def prepare_response_model(response_model: type[T] | None) -> type[T] | None:
    """Normalize user response-model inputs into runtime-ready model classes."""
    from instructor.v2.validation.async_validators import reject_async_validators

    reject_async_validators(response_model)
    if response_model is None:
        return None

    working_model: Any = response_model
    origin = get_origin(working_model)
    if origin is list and is_simple_type(working_model):
        args = get_args(working_model)
        inner = args[0] if args else None

        def _is_model_type(candidate: Any) -> bool:
            if is_typed_dict(candidate):
                return True
            if inspect.isclass(candidate) and issubclass(candidate, BaseModel):
                return True
            return get_origin(candidate) in _UNION_ORIGINS and all(
                inspect.isclass(member) and issubclass(member, BaseModel)
                for member in get_args(candidate)
            )

        if inner is not None and _is_model_type(inner):
            origin = list
        else:
            from instructor.v2.dsl.simple_type import ModelAdapter

            working_model = ModelAdapter.__class_getitem__(working_model)
            origin = get_origin(working_model)

    if is_typed_dict(working_model):
        working_model = _typed_dict_to_model(working_model)

    origin = get_origin(working_model)
    if origin in {Iterable, list}:
        from instructor.v2.dsl.iterable import IterableModel

        args = get_args(working_model)
        if not args or args[0] is None:
            raise ValueError(
                "response_model must be parameterized, e.g. list[User] or Iterable[User]"
            )
        iterable_element_class = args[0]
        if inspect.isclass(iterable_element_class) and not is_typed_dict(
            iterable_element_class
        ):
            try:
                is_base_model = issubclass(iterable_element_class, BaseModel)
            except TypeError:
                is_base_model = False
            if not is_base_model:
                raise TypeError(
                    "response_model iterable elements must be Pydantic models or typed dictionaries; "
                    f"got {iterable_element_class!r}"
                )
        if is_typed_dict(iterable_element_class):
            iterable_element_class = _typed_dict_to_model(iterable_element_class)
        working_model = IterableModel(cast(type[BaseModel], iterable_element_class))

    if is_simple_type(working_model):
        from instructor.v2.dsl.simple_type import ModelAdapter

        working_model = ModelAdapter.__class_getitem__(working_model)

    from instructor.v2.core.function_calls import (
        ResponseSchema,
        openai_schema,
        response_schema,
    )

    if inspect.isclass(working_model) and not issubclass(working_model, ResponseSchema):
        working_model = response_schema(working_model)
    elif not inspect.isclass(working_model):
        working_model = openai_schema(working_model)

    reject_async_validators(working_model)
    return cast(type[T], working_model)
