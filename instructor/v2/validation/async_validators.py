"""Async validation decorators owned by the v2 runtime."""

from __future__ import annotations

from dataclasses import InitVar
from inspect import signature
from typing import (
    Any,
    Callable,
    ClassVar,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel, ValidationInfo
from typing_extensions import TypeAliasType

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
        origin = get_origin(annotation)
        if origin is not None and origin is not annotation:
            return any(declares(arg) for arg in get_args(annotation)) or declares(
                origin
            )
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

    Nested models inside lists, tuples, sets, frozensets and dictionary values
    are validated depth-first so parent validators see the validated children. Returns the (possibly updated) value; raises
    `AsyncValidationError` aggregating every failure found in the subtree.
    """
    if isinstance(value, BaseModel):
        return await _run_on_model(value, context=context)
    if isinstance(value, (list, tuple, set, frozenset, dict)):
        errors: list[ValueError] = []
        validated: list[Any] = []
        items = value.values() if isinstance(value, dict) else value
        for item in items:
            try:
                validated.append(await run_async_validators(item, context=context))
            except AsyncValidationError as exc:
                errors.extend(exc.errors)
        if errors:
            summary = "; ".join(str(error) for error in errors)
            raise AsyncValidationError(
                f"Async validation failed for {type(value).__name__}: {summary}",
                errors=errors,
            )
        if isinstance(value, dict):
            return dict(zip(value, validated))
        if isinstance(value, list):
            return validated
        if isinstance(value, tuple):
            return tuple(validated)
        if isinstance(value, set):
            return set(validated)
        return frozenset(validated)
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


def reject_async_validators(response_model: Any) -> None:
    """Fail closed: the runtime does not execute marked asynchronous validators."""
    seen: set[int] = set()
    seen_schemas: set[int] = set()
    schema_definitions: dict[str, Any] = {}

    def visit_schema(node: Any) -> None:
        if id(node) in seen_schemas:
            return
        seen_schemas.add(id(node))
        if isinstance(node, (list, tuple)):
            for child in node:
                visit_schema(child)
        elif isinstance(node, dict):
            if node.get("type") == "definitions":
                for definition in node.get("definitions", ()):
                    if isinstance(definition, dict) and "ref" in definition:
                        schema_definitions[definition["ref"]] = definition
            if node.get("type") == "definition-ref":
                visit_schema(schema_definitions.get(node.get("schema_ref")))
                return
            if node.get("type") == "model":
                visit(node.get("cls"))
            # Follow validation edges only: default values, user metadata,
            # computed fields and serialization schemas are not validated inputs.
            for key in (
                "schema",
                "items_schema",
                "keys_schema",
                "values_schema",
                "extras_schema",
                "extras_keys_schema",
                "steps",
                "lax_schema",
                "strict_schema",
                "json_schema",
                "python_schema",
                "arguments_schema",
                "var_args_schema",
                "var_kwargs_schema",
                "return_schema",
            ):
                if key in node:
                    visit_schema(node[key])
            for key in ("fields", "choices"):
                children = node.get(key)
                if isinstance(children, dict):
                    for child in children.values():
                        visit_schema(child)
                elif children is not None:
                    visit_schema(children)

    def visit(model: Any) -> None:
        if id(model) in seen:
            return
        seen.add(id(model))
        if get_origin(model) is ClassVar:
            return
        if isinstance(model, InitVar):
            visit(model.type)
            return
        if isinstance(model, TypeAliasType):
            visit(model.__value__)
        origin = get_origin(model)
        for argument in get_args(model):
            visit(argument)
        if origin is not None:
            # Some generic aliases pass the type check but are invalid for issubclass.
            visit(origin)
            return
        if not isinstance(model, type):
            for child in getattr(model, "models", ()):
                visit(child)
            return
        if issubclass(model, BaseModel):
            # Inspect effective attributes, including inherited descriptors.
            members = {}
            for base in reversed(model.__mro__):
                members.update(vars(base))
            for name, member in members.items():
                if isinstance(member, (classmethod, staticmethod)):
                    member = member.__func__
                if (
                    getattr(member, ASYNC_VALIDATOR_KEY, None) is not None
                    or getattr(member, ASYNC_MODEL_VALIDATOR_KEY, None) is not None
                ):
                    raise ValueError(
                        f"{model.__name__}.{name}: Instructor async validators are "
                        "not supported by the runtime. Use Pydantic validators or "
                        "explicitly await validation before consuming results."
                    )
            for field in model.model_fields.values():
                visit(field.annotation)
            # Pydantic may already have resolved local ForwardRefs that normal
            # get_type_hints cannot recover after the defining scope has exited.
            schema = getattr(model, "__pydantic_core_schema__", None)
            if isinstance(schema, dict):
                visit_schema(schema)
        else:
            try:
                annotations = get_type_hints(model, include_extras=True)
            except (NameError, TypeError):
                # Leave unresolved annotations to Pydantic's normal schema handling.
                annotations = getattr(model, "__annotations__", {})
            for annotation in annotations.values():
                visit(annotation)

    visit(response_model)
