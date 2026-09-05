"""Async validation decorators owned by the v2 runtime."""

from __future__ import annotations

from inspect import signature
from typing import Any, Callable, TypeVar, get_args

from pydantic import BaseModel, ValidationInfo

from instructor.v2.core.errors import AsyncValidationError

ASYNC_VALIDATOR_KEY = "__async_validator__"
ASYNC_MODEL_VALIDATOR_KEY = "__async_model_validator__"
T = TypeVar("T", bound=Callable[..., Any])


class AsyncValidationContext:
    """Carry context through async validation hooks."""

    context: dict[str, Any]

    def __init__(self, context: dict[str, Any]):
        self.context = context


def async_field_validator(field: str, *fields: str) -> Callable[[T], T]:
    """Mark a callable as an async field validator."""
    field_names = field, *fields

    def decorator(func: T) -> T:
        params = signature(func).parameters
        requires_validation_context = False
        if len(params) == 3:
            if "info" not in params:
                raise ValueError(
                    "Async validator can only have a value parameter and an optional info parameter"
                )
            if params["info"].annotation not in (ValidationInfo, "ValidationInfo"):
                raise ValueError(
                    "Async validator info parameter must be of type ValidationInfo"
                )
            requires_validation_context = True

        setattr(
            func, ASYNC_VALIDATOR_KEY, (field_names, func, requires_validation_context)
        )
        return func

    return decorator


def async_model_validator() -> Callable[[T], T]:
    """Mark a callable as an async model validator."""

    def decorator(func: T) -> T:
        params = signature(func).parameters
        requires_validation_context = False
        if len(params) > 2:
            raise ValueError("Invalid Parameter Count!")

        if len(params) == 2:
            if "info" not in params:
                raise ValueError(
                    "Async validator can only have a value parameter and an optional info parameter"
                )
            if params["info"].annotation not in (ValidationInfo, "ValidationInfo"):
                raise ValueError(
                    "Async validator info parameter must be of type ValidationInfo"
                )
            requires_validation_context = True

        setattr(
            func,
            ASYNC_MODEL_VALIDATOR_KEY,
            (func, requires_validation_context),
        )
        return func

    return decorator


def _collect_markers(model_cls: type[BaseModel], key: str) -> list[Any]:
    """Walk the MRO (base classes first) collecting distinct decorator markers.

    A subclass redefining a validator under the same attribute name overrides
    the base class's version, matching normal Python method-override semantics.
    """
    markers: dict[str, Any] = {}
    for klass in reversed(model_cls.__mro__):
        for name, member in vars(klass).items():
            marker = getattr(member, key, None)
            if marker is not None:
                markers[name] = marker
            else:
                markers.pop(name, None)
    return list(markers.values())


def model_declares_async_validators(model_cls: Any) -> bool:
    """Detect validators on a model or any nested model, including containers."""
    visited: set[type[BaseModel]] = set()

    def declares(annotation: Any) -> bool:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            if annotation in visited:
                return False
            visited.add(annotation)
            if _collect_markers(annotation, ASYNC_VALIDATOR_KEY) or _collect_markers(
                annotation, ASYNC_MODEL_VALIDATOR_KEY
            ):
                return True
            return any(
                declares(field.annotation) for field in annotation.model_fields.values()
            )
        return any(declares(arg) for arg in get_args(annotation))

    return declares(model_cls)


async def run_async_validators(value: Any, *, context: dict[str, Any] | None) -> Any:
    """Recursively run declared async field/model validators over a parsed value.

    Nested `BaseModel` instances (directly, or inside lists/tuples/dicts) are
    validated depth-first so a parent model's async validators see already
    -validated children. Returns the (possibly updated) value; raises
    `AsyncValidationError` aggregating every failure found in the subtree.
    """
    if isinstance(value, BaseModel):
        return await _run_on_model(value, context=context)
    if isinstance(value, list):
        return [await run_async_validators(item, context=context) for item in value]
    if isinstance(value, tuple):
        return tuple(
            [await run_async_validators(item, context=context) for item in value]
        )
    if isinstance(value, dict):
        return {
            key: await run_async_validators(item, context=context)
            for key, item in value.items()
        }
    return value


async def _run_on_model(
    model: BaseModel, *, context: dict[str, Any] | None
) -> BaseModel:
    model_cls = type(model)
    info = AsyncValidationContext(context or {})
    errors: list[ValueError] = []
    updates: dict[str, Any] = {}

    for field_name in model_cls.model_fields:
        current = getattr(model, field_name)
        try:
            new_value = await run_async_validators(current, context=context)
        except AsyncValidationError as exc:
            errors.extend(exc.errors)
            continue
        if new_value is not current:
            updates[field_name] = new_value
    if updates:
        model = model.model_copy(update=updates)
        updates = {}

    for field_names, validator, needs_info in _collect_markers(
        model_cls, ASYNC_VALIDATOR_KEY
    ):
        for field_name in field_names:
            if field_name not in model_cls.model_fields:
                continue
            current = updates.get(field_name, getattr(model, field_name))
            try:
                new_value = (
                    await validator(model_cls, current, info)
                    if needs_info
                    else await validator(model_cls, current)
                )
            except ValueError as exc:
                errors.append(exc)
                continue
            if new_value is not current:
                updates[field_name] = new_value
    if updates:
        model = model.model_copy(update=updates)

    for validator, needs_info in _collect_markers(model_cls, ASYNC_MODEL_VALIDATOR_KEY):
        try:
            result = (
                await validator(model, info) if needs_info else await validator(model)
            )
        except ValueError as exc:
            errors.append(exc)
            continue
        if result is not None:
            model = result

    if errors:
        summary = "; ".join(str(error) for error in errors)
        raise AsyncValidationError(
            f"Async validation failed for {model_cls.__name__}: {summary}",
            errors=errors,
        )

    return model
