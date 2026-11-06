"""Writer v2 mode handlers.

Writer supports TOOLS, JSON_SCHEMA, and MD_JSON modes. The API is similar to OpenAI but uses
`client.chat.chat` instead of `client.chat.completions.create`.

The handlers reuse some patterns from OpenAI but have Writer-specific
request preparation and response parsing.
"""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Any

from pydantic import BaseModel

from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.errors import IncompleteOutputException, ResponseParsingError
from instructor.v2.core.json import extract_json_from_codeblock
from instructor.v2.providers.openai.schema import generate_openai_schema
from instructor.v2.core.messages import (
    copy_messages_for_mutation,
    dump_message,
    merge_consecutive_messages,
)
from instructor.v2.core.decorators import register_mode_handler
from instructor.v2.providers.openai.handlers import OpenAIHandlerBase


def _extract_reask_message(response: Any) -> dict[str, Any]:
    """Best-effort extraction of a message dict for Writer reask flows."""
    if hasattr(response, "choices") and response.choices:
        message = response.choices[0].message
        try:
            return dict(dump_message(message))
        except Exception:
            return {
                "role": getattr(message, "role", "assistant"),
                "content": getattr(message, "content", ""),
            }
    if hasattr(response, "text"):
        return {"role": "assistant", "content": response.text}
    return {"role": "assistant", "content": getattr(response, "content", str(response))}


def reask_writer_tools(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """Handle reask for Writer tools mode when validation fails."""
    kwargs = kwargs.copy()
    reask_msgs = [_extract_reask_message(response)]
    reask_msgs.append(
        {
            "role": "user",
            "content": (
                f"Validation Error found:\n{exception}\n"
                " Fix errors and fill tool call arguments/name "
                "correctly. Just update arguments dict values or update name. Don't change "
                "the structure of them. You have to call function by passing desired "
                "functions name/args as part of special attribute with name tools_calls, "
                "not as text in attribute with name content. IT'S IMPORTANT!"
            ),
        }
    )
    kwargs["messages"].extend(reask_msgs)
    return kwargs


def reask_writer_json(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """Handle reask for Writer JSON mode when validation fails."""
    kwargs = kwargs.copy()
    base_message = _extract_reask_message(response)
    reask_msgs = [base_message]
    reask_msgs.append(
        {
            "role": "user",
            "content": (
                f"Correct your JSON response: {base_message.get('content', '')}, "
                f"based on the following errors:\n{exception}"
            ),
        }
    )
    kwargs["messages"].extend(reask_msgs)
    return kwargs


def handle_writer_tools(
    response_model: type[Any], new_kwargs: dict[str, Any]
) -> tuple[type[Any], dict[str, Any]]:
    """Handle Writer tools mode."""
    new_kwargs["tools"] = [
        {
            "type": "function",
            "function": generate_openai_schema(response_model),
        }
    ]
    new_kwargs["tool_choice"] = "auto"
    return response_model, new_kwargs


def handle_writer_json(
    response_model: type[Any], new_kwargs: dict[str, Any]
) -> tuple[type[Any], dict[str, Any]]:
    """Handle Writer JSON mode."""
    new_kwargs["response_format"] = {
        "type": "json_schema",
        "json_schema": {"schema": response_model.model_json_schema()},
    }
    return response_model, new_kwargs


class WriterHandlerBase(OpenAIHandlerBase):
    """Shared Writer helpers for OpenAI-compatible streaming responses."""


@register_mode_handler(Provider.WRITER, Mode.TOOLS)
class WriterToolsHandler(WriterHandlerBase):
    """Handler for Writer TOOLS mode.

    Writer uses OpenAI-compatible tool calling format. Tools are defined
    with function schemas and the model returns tool calls with arguments.
    """

    mode = Mode.TOOLS

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        """Prepare request with tool definitions for Writer."""
        if response_model is None:
            return None, kwargs

        from instructor.v2.core.response_model import prepare_response_model

        response_model = prepare_response_model(response_model)
        self._register_streaming_from_kwargs(response_model, kwargs)

        new_kwargs = kwargs.copy()
        new_kwargs["messages"] = list(kwargs.get("messages", []))
        schema = generate_openai_schema(response_model)

        new_kwargs["tools"] = [{"type": "function", "function": schema}]
        new_kwargs["tool_choice"] = "auto"

        return response_model, new_kwargs

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        """Handle reask for Writer tools mode."""
        return reask_writer_tools(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,  # noqa: ARG002
        is_async: bool = False,  # noqa: ARG002
    ) -> BaseModel:
        """Parse tool call response from Writer."""
        if isinstance(response_model, type) and self._consume_streaming_flag(
            response_model
        ):
            return self._parse_streaming_response(
                response_model,
                response,
                validation_context,
                strict,
            )
        # Check for truncated output
        choices = getattr(response, "choices", None)
        if not choices:
            raise ResponseParsingError(
                "No choices in Writer response",
                mode=str(self.mode.value),
                raw_response=response,
            )
        if choices[0].finish_reason == "length":
            raise IncompleteOutputException(last_completion=response)

        # Extract JSON from tool call
        tool_calls = choices[0].message.tool_calls
        if not tool_calls:
            raise ResponseParsingError(
                "No tool calls in Writer response",
                mode=str(self.mode.value),
                raw_response=response,
            )
        tool_call = tool_calls[0]
        json_str = tool_call.function.arguments

        return response_model.model_validate_json(
            json_str,
            context=validation_context,
            strict=strict,
        )


@register_mode_handler(Provider.WRITER, Mode.MD_JSON)
class WriterMDJSONHandler(WriterHandlerBase):
    """Handler for Writer MD_JSON mode.

    Extracts JSON from markdown code blocks in the response text.
    This is a fallback mode when tool calling is not suitable.
    """

    mode = Mode.MD_JSON

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        """Prepare request with JSON schema instruction in messages."""
        if response_model is None:
            return None, kwargs

        new_kwargs = kwargs.copy()
        self._register_streaming_from_kwargs(response_model, new_kwargs)
        schema = response_model.model_json_schema()

        message = dedent(
            f"""
            As a genius expert, your task is to understand the content and provide
            the parsed objects in json that match the following json_schema:\n

            {json.dumps(schema, indent=2, ensure_ascii=False)}

            Make sure to return an instance of the JSON, not the schema itself
            """
        )

        # Add system message with schema
        messages = copy_messages_for_mutation(new_kwargs.get("messages", []))
        if messages and messages[0]["role"] != "system":
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": message,
                },
            )
        elif messages and isinstance(messages[0]["content"], str):
            messages[0]["content"] += f"\n\n{message}"
        elif messages and isinstance(messages[0]["content"], list):
            messages[0]["content"][0]["text"] += f"\n\n{message}"
        else:
            messages.insert(0, {"role": "system", "content": message})

        # Add user message requesting JSON in code block
        messages.append(
            {
                "role": "user",
                "content": "Return the correct JSON response within a ```json codeblock. not the JSON_SCHEMA",
            },
        )
        new_kwargs["messages"] = merge_consecutive_messages(messages)

        return response_model, new_kwargs

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        """Handle reask for Writer MD_JSON mode."""
        return reask_writer_json(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,  # noqa: ARG002
        is_async: bool = False,  # noqa: ARG002
    ) -> BaseModel:
        """Parse JSON from markdown code block in response."""
        if isinstance(response_model, type) and self._consume_streaming_flag(
            response_model
        ):
            return self._parse_streaming_response(
                response_model,
                response,
                validation_context,
                strict,
            )
        text = response.choices[0].message.content or ""
        json_str = extract_json_from_codeblock(text)

        return response_model.model_validate_json(
            json_str,
            context=validation_context,
            strict=strict,
        )


@register_mode_handler(Provider.WRITER, Mode.JSON_SCHEMA)
class WriterJSONSchemaHandler(WriterHandlerBase):
    """Handler for Writer native JSON schema mode."""

    mode = Mode.JSON_SCHEMA

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        """Prepare request with Writer's native JSON schema response format."""
        if response_model is None:
            return None, kwargs

        new_kwargs = kwargs.copy()
        new_kwargs["messages"] = list(kwargs.get("messages", []))
        self._register_streaming_from_kwargs(response_model, new_kwargs)
        return handle_writer_json(response_model, new_kwargs)

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        """Handle reask for Writer JSON schema mode."""
        return reask_writer_json(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,  # noqa: ARG002
        is_async: bool = False,  # noqa: ARG002
    ) -> BaseModel:
        """Parse Writer native JSON schema responses."""
        if isinstance(response_model, type) and self._consume_streaming_flag(
            response_model
        ):
            return self._parse_streaming_response(
                response_model,
                response,
                validation_context,
                strict,
            )
        text = response.choices[0].message.content or ""
        return response_model.model_validate_json(
            text,
            context=validation_context,
            strict=strict,
        )


__all__ = [
    "WriterToolsHandler",
    "WriterJSONSchemaHandler",
    "WriterMDJSONHandler",
]
