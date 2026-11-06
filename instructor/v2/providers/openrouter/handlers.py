"""OpenRouter v2 mode handlers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.decorators import register_mode_handler

# Register OpenAI-compatible handlers (TOOLS, MD_JSON, PARALLEL_TOOLS) for OpenRouter.
from instructor.v2.providers.openai import handlers as _openai_handlers  # noqa: F401
from instructor.v2.providers.openai.handlers import (
    OpenAIJSONSchemaHandler,
    handle_openrouter_structured_outputs,
    reask_default,
)


@register_mode_handler(Provider.OPENROUTER, Mode.JSON_SCHEMA)
class OpenRouterJSONSchemaHandler(OpenAIJSONSchemaHandler):
    """Handler for OpenRouter structured outputs."""

    mode = Mode.JSON_SCHEMA

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        if response_model is None:
            return None, kwargs
        new_kwargs = kwargs.copy()
        new_kwargs["messages"] = list(kwargs.get("messages", []))
        return handle_openrouter_structured_outputs(response_model, new_kwargs)

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        return reask_default(kwargs, response, exception)


__all__ = ["OpenRouterJSONSchemaHandler"]
