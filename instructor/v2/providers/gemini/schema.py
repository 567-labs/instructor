"""Legacy Gemini-specific schema helpers."""

from __future__ import annotations

import importlib
import functools
import warnings
from typing import Any, cast

from pydantic import BaseModel

from instructor.v2.providers.gemini.utils import map_to_gemini_function_schema
from instructor.v2.providers.openai.schema import generate_openai_schema


@functools.lru_cache(maxsize=256)
def generate_gemini_schema(model: type[BaseModel]) -> Any:
    """Generate a legacy Gemini function schema from a Pydantic model."""
    warnings.warn(
        "generate_gemini_schema is deprecated. The google-generativeai library is being replaced by google-genai.",
        DeprecationWarning,
        stacklevel=2,
    )

    try:
        genai_types = cast(Any, importlib.import_module("google.generativeai.types"))
        openai_schema = generate_openai_schema(model)
        return genai_types.FunctionDeclaration(
            name=openai_schema["name"],
            description=openai_schema["description"],
            parameters=map_to_gemini_function_schema(openai_schema["parameters"]),
        )
    except ImportError as e:
        raise ImportError(
            "google-generativeai is deprecated. Please install google-genai instead: pip install google-genai"
        ) from e
