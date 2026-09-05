"""MiniMax-specific utilities.

This module contains utilities specific to the MiniMax provider,
including reask functions, response handlers, and message formatting.
"""

from __future__ import annotations

import json
from typing import Any

from ...mode import Mode
from ...utils.core import dump_message
from ...processing.schema import generate_openai_schema


def reask_minimax_tools(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """
    Handle reask for MiniMax tools mode when validation fails.

    Kwargs modifications:
    - Adds: "messages" (tool response messages indicating validation errors)
    """
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
):
    """
    Handle reask for MiniMax JSON mode when validation fails.

    Kwargs modifications:
    - Adds: "messages" (user message requesting JSON correction)
    """
    kwargs = kwargs.copy()
    reask_msgs = [dump_message(response.choices[0].message)]
    reask_msgs.append(
        {
            "role": "user",
            "content": f"Correct your JSON ONLY RESPONSE, based on the following errors:\n{exception}",
        }
    )
    kwargs["messages"].extend(reask_msgs)
    return kwargs


def handle_minimax_tools(
    response_model: type[Any], new_kwargs: dict[str, Any]
) -> tuple[type[Any], dict[str, Any]]:
    """
    Handle MiniMax tools mode.

    Kwargs modifications:
    - Adds: "tools" (list with function schema)
    - Adds: "tool_choice" (forced function call)
    """
    new_kwargs["tools"] = [
        {
            "type": "function",
            "function": generate_openai_schema(response_model),
        }
    ]
    new_kwargs["tool_choice"] = {
        "type": "function",
        "function": {"name": generate_openai_schema(response_model)["name"]},
    }
    return response_model, new_kwargs


def handle_minimax_json(
    response_model: type[Any], new_kwargs: dict[str, Any]
) -> tuple[type[Any], dict[str, Any]]:
    """
    Handle MiniMax JSON mode via system prompt injection.

    MiniMax does not support the ``response_format`` parameter, so structured
    output is requested by injecting a schema instruction into the system
    message.  Reasoning models may prepend ``<think>...</think>`` blocks to
    their output; these are stripped in ``function_calls.py`` before JSON
    parsing.

    Kwargs modifications:
    - Prepends / extends: "messages" system message with JSON schema instruction
    """
    schema = json.dumps(response_model.model_json_schema(), indent=2)
    instruction = (
        "You are a helpful assistant that returns structured data.\n"
        "Return ONLY a valid JSON object that matches the following schema — "
        "no extra text, no markdown fences, no explanations:\n\n"
        f"{schema}\n\n"
        f"Your response must be parseable by `{response_model.__name__}.model_validate_json()`."
    )

    messages = new_kwargs.get("messages", [])
    if messages and messages[0]["role"] == "system":
        messages[0] = {
            "role": "system",
            "content": messages[0]["content"] + f"\n\n{instruction}",
        }
    else:
        messages = [{"role": "system", "content": instruction}] + messages
    new_kwargs["messages"] = messages
    return response_model, new_kwargs


# Handler registry for MiniMax
MINIMAX_HANDLERS = {
    Mode.MINIMAX_TOOLS: {
        "reask": reask_minimax_tools,
        "response": handle_minimax_tools,
    },
    Mode.MINIMAX_JSON: {
        "reask": reask_minimax_json,
        "response": handle_minimax_json,
    },
}
