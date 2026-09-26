"""Compatibility exports for v2-owned schema generation helpers."""

from instructor.v2.core.schema import (
    generate_anthropic_schema,
    generate_gemini_schema,
    generate_openai_schema,
    make_strict_schema,
)

__all__ = [
    "generate_openai_schema",
    "generate_anthropic_schema",
    "generate_gemini_schema",
    "make_strict_schema",
]
