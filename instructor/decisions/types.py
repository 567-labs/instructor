"""Small, provider-independent annotations for decision models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, EnumMeta
from typing import Any, cast


@dataclass(frozen=True, eq=False)
class Choice:
    description: Any = None
    examples: list[Any] = field(default_factory=list)
    value: str | None = None


class _ChoicesMeta(EnumMeta):
    def __new__(metacls, name, bases, namespace, **kwargs):
        cls = super().__new__(metacls, name, bases, namespace, **kwargs)
        values = {}
        for member in cast(Any, cls):
            spec = member.value
            if not isinstance(spec, Choice):
                raise TypeError(f"{name}.{member.name} must be a Choice")
            value = spec.value if spec.value is not None else member.name.lower()
            if not isinstance(value, str) or not value or value in values:
                raise ValueError(
                    f"{name}: choice values must be unique nonempty strings"
                )
            member._value_ = value
            member.choice = spec
            values[value] = member
        cls._value2member_map_ = values
        return cls


class Choices(Enum, metaclass=_ChoicesMeta):
    """Reusable choices. A member's value defaults to its lowercase name."""

    choice: Choice


@dataclass(frozen=True)
class Question:
    instructions: Any


@dataclass(frozen=True)
class Noul:
    instructions: Any
    when_true: Any = None
    when_false: Any = None
    examples: list[tuple[Any, bool]] = field(default_factory=list)
    criteria: dict[str, Any] | None = None


@dataclass(frozen=True)
class Level:
    meaning: Any
    examples: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class Score:
    instructions: Any
    levels: list[Level | str | dict | list]
    scale: tuple[float, float] | None = None
    examples: list[tuple[Any, int]] = field(default_factory=list)
