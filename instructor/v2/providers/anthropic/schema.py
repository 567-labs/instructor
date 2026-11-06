"""Anthropic-specific schema helpers."""

from __future__ import annotations

import functools
from typing import Any

from pydantic import BaseModel

from instructor.v2.providers.openai.schema import generate_openai_schema


@functools.lru_cache(maxsize=256)
def generate_anthropic_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Generate an Anthropic tool schema from a Pydantic model."""
    openai_schema = generate_openai_schema(model)
    return {
        "name": openai_schema["name"],
        "description": openai_schema["description"],
        "input_schema": model.model_json_schema(),
    }
