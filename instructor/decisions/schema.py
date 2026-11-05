"""Translate Pydantic models to decision questions and validate answers."""

from __future__ import annotations

import json
import math
import types
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Annotated, Literal, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict

from instructor.v2.core.templating import apply_template
from .types import Choice, Choices, Level, Noul, Question, Score

_UNION_TYPE = getattr(types, "UnionType", ())
T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class _Field:
    name: str
    input_name: str
    kind: str
    question: dict[str, Any]
    scale: tuple[float, float] | None = None
    level_count: int = 0


def _json_value(value: Any, label: str) -> None:
    """Reject values JSON would silently change, including non-string object keys."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    if isinstance(value, list):
        for item in value:
            _json_value(item, label)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _json_value(item, label)
        return
    raise ValueError(f"{label} must contain JSON-compatible values")


def _entry(value: Any, *, nullable: bool = False) -> Any:
    if value is None and nullable:
        return None
    if not isinstance(value, (str, dict, list)):
        raise ValueError(
            "Instructions and criteria must be strings, objects, or arrays"
        )
    _json_value(value, "Instructions and criteria")
    return value


def _instructions(value: Any, context: dict[str, Any]) -> Any:
    value = _entry(value)
    if not value:
        raise ValueError("Decision questions require instructions")

    def render(item: Any) -> Any:
        if isinstance(item, str):
            return apply_template(item, context)
        if isinstance(item, dict):
            return {key: render(val) for key, val in item.items()}
        if isinstance(item, list):
            return [render(val) for val in item]
        return item

    return render(value)


def _with_examples(meaning: Any, examples: list[Any]) -> Any:
    meaning = deepcopy(_entry(meaning, nullable=True))
    if not examples:
        return meaning
    return {"meaning": meaning, "examples": [str(example) for example in examples]}


def _choice_options(annotation: Any) -> tuple[dict[str, Any], Any]:
    options: dict[str, Any] = {}
    doc = None

    def add(
        value: Any, description: Any = None, examples: list[Any] | None = None
    ) -> None:
        if not isinstance(value, str) or not value or value in options:
            raise ValueError("Choice values must be unique nonempty strings")
        options[value] = _with_examples(description, examples or [])

    def visit(annotation: Any) -> None:
        nonlocal doc
        origin = get_origin(annotation)
        if origin is Annotated:
            inner, *metadata = get_args(annotation)
            choices = [item for item in metadata if isinstance(item, Choice)]
            if len(choices) != 1 or get_origin(inner) is not Literal:
                raise ValueError(
                    "Inline choices require Annotated[Literal[...], Choice(...)]"
                )
            spec = choices[0]
            if spec.value is not None:
                raise ValueError("Choice(value=...) is only supported in Choices enums")
            for value in get_args(inner):
                add(value, spec.description, spec.examples)
        elif origin is Literal:
            for value in get_args(annotation):
                add(value)
        elif origin in (Union, _UNION_TYPE):
            for branch in get_args(annotation):
                visit(branch)
        elif isinstance(annotation, type) and issubclass(annotation, Enum):
            doc = annotation.__dict__.get("__doc__")
            for member in annotation.__members__.values():
                if isinstance(member, Choices):
                    add(member.value, member.choice.description, member.choice.examples)
                else:
                    add(member.value)
        else:
            raise ValueError(
                "Expected a string Literal, enum, or union of inline choices"
            )

    visit(annotation)
    if not 1 <= len(options) <= 255:
        raise ValueError("A Choice must have between 1 and 255 options")
    return options, doc


def build_questions(model: type[BaseModel], context: dict[str, Any]) -> list[_Field]:
    if not isinstance(model, type) or not issubclass(model, BaseModel):
        raise TypeError("response_model must be a Pydantic BaseModel")
    from instructor.v2.validation.async_validators import reject_async_validators

    reject_async_validators(model)
    if not isinstance(context, dict):
        raise TypeError("context must be a dictionary")
    _json_value(context, "context")

    fields = []
    input_names: set[str] = set()
    for name, field in model.model_fields.items():
        metadata = field.metadata
        specs = [item for item in metadata if isinstance(item, (Noul, Score, Question))]
        if len(specs) > 1:
            raise ValueError(f"{name}: use only one decision annotation")
        spec = specs[0] if specs else None
        input_name = name
        # Before Pydantic 2.11, this config key is ignored by validation.
        if not (
            "validate_by_alias" in ConfigDict.__annotations__
            and model.model_config.get("validate_by_alias") is False
        ):
            if field.validation_alias is not None:
                input_name = field.validation_alias
        if not isinstance(input_name, str):
            raise ValueError(f"{name}: complex validation aliases are not supported")
        if input_name in input_names:
            raise ValueError(f"{name}: validation aliases must be unique")
        input_names.add(input_name)
        if isinstance(spec, (Noul, Score)) and any(
            isinstance(item, Choice) for item in metadata
        ):
            raise ValueError(
                f"{name}: Choice cannot be combined with {type(spec).__name__}"
            )
        if isinstance(spec, Noul):
            if field.annotation is not float:
                raise ValueError(f"{name}: Noul requires a float field")
            grouped: dict[bool, list[Any]] = {True: [], False: []}
            for example, label in spec.examples:
                if type(label) is not bool:
                    raise ValueError(f"{name}: Noul example labels must be booleans")
                grouped[label].append(example)
            if spec.criteria is not None and (
                spec.when_true is not None or spec.when_false is not None
            ):
                raise ValueError(
                    f"{name}: criteria cannot be combined with when_true or when_false"
                )
            if spec.criteria is not None and (
                not isinstance(spec.criteria, dict)
                or set(spec.criteria) - {"true", "false"}
            ):
                raise ValueError(
                    f"{name}: Noul criteria must contain only 'true' and 'false'"
                )
            criteria = {}
            true_meaning = (
                spec.criteria.get("true")
                if spec.criteria is not None
                else spec.when_true
            )
            false_meaning = (
                spec.criteria.get("false")
                if spec.criteria is not None
                else spec.when_false
            )
            for label, meaning in ((True, true_meaning), (False, false_meaning)):
                if meaning is not None or grouped[label]:
                    criteria[str(label).lower()] = _with_examples(
                        meaning, grouped[label]
                    )
            question = {
                "type": "noul",
                "instructions": _instructions(spec.instructions, context),
            }
            if criteria:
                question["criteria"] = criteria
            fields.append(_Field(name, input_name, "noul", question))
        elif isinstance(spec, Score):
            if field.annotation is not float:
                raise ValueError(f"{name}: Score requires a float field")
            if not 2 <= len(spec.levels) <= 10:
                raise ValueError(f"{name}: Score requires 2 to 10 levels")
            if spec.scale is not None:
                if (
                    len(spec.scale) != 2
                    or any(
                        isinstance(x, bool)
                        or not isinstance(x, (int, float))
                        or not math.isfinite(x)
                        for x in spec.scale
                    )
                    or spec.scale[0] >= spec.scale[1]
                ):
                    raise ValueError(
                        f"{name}: scale bounds must be finite and increasing"
                    )
            examples: list[list[Any]] = [[] for _ in spec.levels]
            for example, index in spec.examples:
                if type(index) is not int or not 0 <= index < len(examples):
                    raise ValueError(f"{name}: invalid Score example level index")
                examples[index].append(example)
            criteria = []
            for index, level in enumerate(spec.levels):
                level = level if isinstance(level, Level) else Level(level)
                if level.meaning is None:
                    raise ValueError(f"{name}: Score levels require a meaning")
                criteria.append(
                    _with_examples(level.meaning, level.examples + examples[index])
                )
            question = {
                "type": "score",
                "instructions": _instructions(spec.instructions, context),
                "criteria": criteria,
            }
            fields.append(
                _Field(name, input_name, "score", question, spec.scale, len(criteria))
            )
        else:
            inline = [item for item in metadata if isinstance(item, Choice)]
            if len(inline) > 1:
                raise ValueError(f"{name}: use only one Choice annotation")
            annotation = (
                Annotated[field.annotation, inline[0]] if inline else field.annotation
            )
            options, doc = _choice_options(annotation)
            instruction = (
                spec.instructions
                if isinstance(spec, Question)
                else field.description or doc
            )
            question = {
                "type": "choice",
                "instructions": _instructions(instruction, context),
                "criteria": options,
            }
            fields.append(_Field(name, input_name, "choice", question))
    if not fields:
        raise ValueError("A decision model must contain at least one question")
    return fields


def parse_answers(
    model: type[T],
    fields: list[_Field],
    raw: Any,
    context: dict[str, Any],
    strict: bool,
) -> T:
    if not isinstance(raw, dict) or not isinstance(raw.get("answers"), dict):
        raise ValueError("Decision response must contain an answers object")
    values = {}
    for field in fields:
        answer = raw["answers"].get(field.name)
        if not isinstance(answer, dict) or answer.get("type") != field.kind:
            raise ValueError(f"{field.name}: missing or mismatched decision answer")
        value = answer.get(field.kind)
        if field.kind == "choice":
            if not isinstance(value, str) or value not in field.question["criteria"]:
                raise ValueError(f"{field.name}: invalid choice answer")
        else:
            maximum = 1 if field.kind == "noul" else field.level_count - 1
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= maximum
            ):
                raise ValueError(f"{field.name}: invalid {field.kind} answer")
            if field.scale is not None:
                low, high = field.scale
                ratio = value / maximum
                # Avoid overflowing the difference of large finite bounds.
                value = low * (1 - ratio) + high * ratio
        values[field.input_name] = value
    return model.model_validate_json(json.dumps(values), context=context, strict=strict)
