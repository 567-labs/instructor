"""MiniMax v2 mode handlers."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.decorators import register_mode_handler
from instructor.v2.providers.openai.handlers import (
    OpenAIToolsHandler,
    OpenAIMDJSONHandler,
)


def reask_minimax_tools(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
) -> dict[str, Any]:
    """Handle reask for MiniMax tools mode when validation fails."""
    from instructor.v2.core.messages import dump_message

    kwargs = kwargs.copy()
    reask_msgs = [dump_message(response.choices[0].message)]
    for tool_call in response.choices[0].message.tool_calls:
        reask_msgs.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": (
                    f"Validation Error found:\n{exception}\nRecall the function correctly, "
                    f"fix the errors and call the tool {tool_call.function.name} again."
                ),
            }
        )
    kwargs["messages"].extend(reask_msgs)
    return kwargs


def reask_minimax_json(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
) -> dict[str, Any]:
    """Handle reask for MiniMax JSON mode when validation fails."""
    from instructor.v2.core.messages import dump_message

    kwargs = kwargs.copy()
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


@register_mode_handler(Provider.MINIMAX, Mode.TOOLS)
class MiniMaxToolsHandler(OpenAIToolsHandler):
    """Handler for MiniMax tool-calling mode."""

    mode = Mode.TOOLS

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        return reask_minimax_tools(kwargs, response, exception)


@register_mode_handler(Provider.MINIMAX, Mode.MD_JSON)
class MiniMaxMDJSONHandler(OpenAIMDJSONHandler):
    """Handler for MiniMax JSON mode via system-prompt injection.

    MiniMax does not support ``response_format``, so schema is injected into
    the system message. Reasoning models may prepend <think>...</think> blocks;
    these are stripped before JSON parsing by the base MD_JSON handler.
    """

    mode = Mode.MD_JSON

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        if response_model is None:
            return None, kwargs

        schema = json.dumps(response_model.model_json_schema(), indent=2)
        instruction = (
            "You are a helpful assistant that returns structured data.\n"
            "Return ONLY a valid JSON object that matches the following schema — "
            "no extra text, no markdown fences, no explanations:\n\n"
            f"{schema}\n\n"
            f"Your response must be parseable by `{response_model.__name__}.model_validate_json()`."
        )

        new_kwargs = kwargs.copy()
        messages = list(new_kwargs.get("messages", []))
        if messages and messages[0]["role"] == "system":
            messages[0] = {
                **messages[0],
                "content": messages[0]["content"] + f"\n\n{instruction}",
            }
        else:
            messages = [{"role": "system", "content": instruction}] + messages
        new_kwargs["messages"] = messages
        return response_model, new_kwargs

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        return reask_minimax_json(kwargs, response, exception)


__all__ = ["MiniMaxToolsHandler", "MiniMaxMDJSONHandler"]
