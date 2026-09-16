"""Perplexity v2 mode handlers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.decorators import register_mode_handler
from instructor.v2.providers.openai.handlers import OpenAIMDJSONHandler


def reask_perplexity_json(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """Handle reask for Perplexity JSON mode when validation fails."""
    from instructor.v2.core.messages import dump_message

    kwargs = kwargs.copy()
    kwargs["messages"] = list(kwargs.get("messages", []))
    reask_msgs = [dump_message(response.choices[0].message)]
    reask_msgs.append(
        {
            "role": "user",
            "content": (
                "Correct your JSON ONLY RESPONSE, based on the following errors:\n"
                f"{exception}"
            ),
        }
    )
    kwargs["messages"].extend(reask_msgs)
    return kwargs


def handle_perplexity_json(
    response_model: type[Any], new_kwargs: dict[str, Any]
) -> tuple[type[Any], dict[str, Any]]:
    """Handle Perplexity JSON mode."""
    new_kwargs["response_format"] = {
        "type": "json_schema",
        "json_schema": {"schema": response_model.model_json_schema()},
    }
    return response_model, new_kwargs


@register_mode_handler(Provider.PERPLEXITY, Mode.MD_JSON)
class PerplexityMDJSONHandler(OpenAIMDJSONHandler):
    """Handler for Perplexity JSON mode."""

    mode = Mode.MD_JSON

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        if response_model is None:
            return None, kwargs
        new_kwargs = kwargs.copy()
        return handle_perplexity_json(response_model, new_kwargs)

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        return reask_perplexity_json(kwargs, response, exception)


__all__ = ["PerplexityMDJSONHandler"]
