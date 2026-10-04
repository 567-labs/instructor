"""Small generic helpers owned by the v2 runtime."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Generic, TypeVar, cast

from pydantic import ValidationError

R_co = TypeVar("R_co", covariant=True)
_validation_error_original_str: Callable[[ValidationError], str] | None = None


def is_async(func: Callable[..., Any]) -> bool:
    """Return whether a callable is async, following wrapped callables."""
    while True:
        if inspect.iscoroutinefunction(func) or inspect.iscoroutinefunction(
            type(func).__call__
        ):
            return True
        wrapped = getattr(func, "__wrapped__", None)
        if not callable(wrapped):
            return False
        func = wrapped


class classproperty(Generic[R_co]):
    """Descriptor for class-level properties."""

    def __init__(self, method: Callable[[Any], R_co]) -> None:
        self.cproperty = method

    def __get__(self, _instance: object, cls: type[Any]) -> R_co:
        return self.cproperty(cls)


def disable_pydantic_error_url() -> None:
    """Disable URLs in Pydantic ValidationError messages."""
    global _validation_error_original_str
    if _validation_error_original_str is None:
        _validation_error_original_str = ValidationError.__str__

    original_str = _validation_error_original_str

    def __str__(self: ValidationError) -> str:
        output = original_str(self)
        return "\n".join(
            line
            for line in output.split("\n")
            if "https://errors.pydantic.dev" not in line
        )

    cast(Any, ValidationError).__str__ = __str__
