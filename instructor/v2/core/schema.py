"""Compatibility exports for provider-owned schema generation helpers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from instructor.v2.providers.anthropic.schema import (
    generate_anthropic_schema as _generate_anthropic_schema,
)
from instructor.v2.providers.gemini.schema import (
    generate_gemini_schema as _generate_gemini_schema,
)
from instructor.v2.providers.openai.schema import (
    generate_openai_schema as _generate_openai_schema,
    make_strict_schema as _make_strict_schema,
)

__all__ = [
    "generate_openai_schema",
    "generate_anthropic_schema",
    "generate_gemini_schema",
    "make_strict_schema",
]


def generate_openai_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Compatibility wrapper for the provider-owned OpenAI schema helper."""
    return _generate_openai_schema(model)


def make_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Compatibility wrapper for the provider-owned strict schema helper."""
    return _make_strict_schema(schema)


def generate_anthropic_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Compatibility wrapper for the provider-owned Anthropic schema helper."""
    return _generate_anthropic_schema(model)


def generate_gemini_schema(model: type[BaseModel]) -> Any:
    """Compatibility wrapper for the provider-owned Gemini schema helper."""
    return _generate_gemini_schema(model)
