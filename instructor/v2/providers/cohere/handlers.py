"""Cohere v2 mode handlers.

Cohere supports both V1 and V2 client APIs. The handlers detect which version
is being used and format requests/responses accordingly.

V1 format: Uses chat_history + message
V2 format: Uses OpenAI-style messages
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Generator
from typing import Any, cast

from pydantic import BaseModel

from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.errors import ConfigurationError, ResponseParsingError
from instructor.v2.core.json import extract_json_from_codeblock
from instructor.v2.core.decorators import register_mode_handler
from instructor.v2.core.handler import ModeHandler
from instructor.v2.core.messages import copy_messages_for_mutation
from instructor.v2.dsl.iterable import IterableBase
from instructor.v2.dsl.partial import PartialBase


def _detect_client_version(kwargs: dict[str, Any]) -> str:
    """Detect Cohere client version from kwargs."""
    version = kwargs.get("_cohere_client_version")
    if version:
        return version
    # Fallback detection based on kwargs structure
    if "messages" in kwargs:
        return "v2"
    if "chat_history" in kwargs or "message" in kwargs:
        return "v1"
    return "v2"  # Default to v2


def _convert_messages_to_cohere_v1(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Convert OpenAI-style messages to Cohere V1 format."""
    new_kwargs = kwargs.copy()
    new_kwargs.pop("_cohere_client_version", None)

    messages = new_kwargs.pop("messages", [])
    chat_history = []

    for message in messages[:-1]:
        chat_history.append(
            {
                "role": message["role"],
                "message": message["content"],
            }
        )

    if messages:
        new_kwargs["message"] = messages[-1]["content"]
    new_kwargs["chat_history"] = chat_history

    # Rename model_name to model if needed
    if "model_name" in new_kwargs and "model" not in new_kwargs:
        new_kwargs["model"] = new_kwargs.pop("model_name")

    new_kwargs.pop("strict", None)
    return new_kwargs


def _convert_messages_to_cohere_v2(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Clean up kwargs for Cohere V2 format (OpenAI-compatible)."""
    new_kwargs = kwargs.copy()
    new_kwargs.pop("_cohere_client_version", None)
    if "messages" in new_kwargs:
        new_kwargs["messages"] = copy_messages_for_mutation(new_kwargs["messages"])

    # Rename model_name to model if needed
    if "model_name" in new_kwargs and "model" not in new_kwargs:
        new_kwargs["model"] = new_kwargs.pop("model_name")

    new_kwargs.pop("strict", None)
    return new_kwargs


def _convert_messages(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Convert messages based on detected client version."""
    version = _detect_client_version(kwargs)
    if version == "v1":
        return _convert_messages_to_cohere_v1(kwargs)
    return _convert_messages_to_cohere_v2(kwargs)


def _extract_text_from_response(response: Any) -> str:
    """Extract text content from Cohere response (V1 or V2)."""
    # V1 format: direct text access
    if hasattr(response, "text"):
        return response.text

    # V2 format: message.content[].text
    if hasattr(response, "message") and hasattr(response.message, "content"):
        content_items = response.message.content
        if content_items:
            for item in content_items:
                if (
                    hasattr(item, "type")
                    and item.type == "text"
                    and hasattr(item, "text")
                ):
                    return item.text

    raise ResponseParsingError(
        f"Could not extract text from Cohere response: {type(response)}",
        mode="COHERE",
        raw_response=response,
    )


def _extract_text_from_stream_chunk(chunk: Any) -> str | None:
    # Cohere V2 stream events
    chunk_type = getattr(chunk, "type", None)
    if chunk_type == "content-delta":
        delta = getattr(chunk, "delta", None)
        message = getattr(delta, "message", None)
        content = getattr(message, "content", None)
        text = getattr(content, "text", None)
        if text:
            return text

    # Cohere V1 stream events (best-effort)
    text = getattr(chunk, "text", None)
    if text:
        return text
    return None


class CohereHandlerBase(ModeHandler):
    """Base class for Cohere handlers with shared utilities."""

    mode: Mode

    def _create_reask_message(
        self,
        response: Any,
        exception: Exception,
    ) -> str:
        """Create a reask message for validation errors."""
        try:
            response_text = _extract_text_from_response(response)
        except ResponseParsingError:
            response_text = str(response)

        return (
            "Correct the following JSON response, based on the errors given below:\n\n"
            f"JSON:\n{response_text}\n\nExceptions:\n{exception}"
        )

    def extract_streaming_json(self, completion: Any) -> Generator[str, None, None]:
        for chunk in completion:
            text = _extract_text_from_stream_chunk(chunk)
            if text:
                yield text

    async def extract_streaming_json_async(
        self, completion: AsyncGenerator[Any, None]
    ) -> AsyncGenerator[str, None]:
        async for chunk in completion:
            text = _extract_text_from_stream_chunk(chunk)
            if text:
                yield text

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        kwargs = kwargs.copy()
        correction_msg = self._create_reask_message(response, exception)

        if "messages" in kwargs:
            # V2 format
            kwargs["messages"] = list(kwargs.get("messages", []))
            kwargs["messages"].append({"role": "user", "content": correction_msg})
        else:
            # V1 format
            message = kwargs.get("message", "")
            if "chat_history" in kwargs:
                kwargs["chat_history"] = list(kwargs["chat_history"])
                kwargs["chat_history"].append({"role": "user", "message": message})
            else:
                kwargs["chat_history"] = [{"role": "user", "message": message}]
            kwargs["message"] = correction_msg

        return kwargs


@register_mode_handler(Provider.COHERE, Mode.TOOLS)
class CohereToolsHandler(CohereHandlerBase):
    """Handler for Cohere TOOLS mode.

    Uses prompt-based extraction with JSON schema instructions.
    Cohere doesn't have native tool calling like OpenAI, so we use
    prompt engineering to get structured JSON output.
    """

    mode = Mode.TOOLS

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        new_kwargs = _convert_messages(kwargs)

        if response_model is None:
            return None, new_kwargs

        # Prepare response model for simple types
        from instructor.v2.core.response_model import prepare_response_model

        prepared_model = prepare_response_model(response_model)
        assert prepared_model is not None  # Already checked response_model is not None

        # Create extraction instruction
        instruction = f"""\
Extract a valid {prepared_model.__name__} object based on the chat history and the json schema below.
{prepared_model.model_json_schema()}
The JSON schema was obtained by running:
```python
schema = {prepared_model.__name__}.model_json_schema()
```

The output must be a valid JSON object that `{prepared_model.__name__}.model_validate_json()` can successfully parse.
Respond with JSON only. Do not include code fences, markdown, or extra text.
"""

        # Add instruction based on client version
        if "messages" in new_kwargs:
            # V2 format: prepend to messages (copy to avoid mutating caller's list)
            new_kwargs["messages"] = list(new_kwargs["messages"])
            new_kwargs["messages"].insert(0, {"role": "user", "content": instruction})
        else:
            # V1 format: prepend to chat_history
            new_kwargs["chat_history"] = [
                {"role": "user", "message": instruction}
            ] + new_kwargs.get("chat_history", [])

        return prepared_model, new_kwargs

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,
        is_async: bool = False,  # noqa: ARG002
    ) -> Any:
        if stream:
            if not (
                isinstance(response_model, type)
                and issubclass(response_model, (IterableBase, PartialBase))
            ):
                raise ConfigurationError(
                    "Streaming is only supported for Iterable or Partial response models in Cohere TOOLS mode."
                )

            parse_kwargs: dict[str, Any] = {}
            if validation_context is not None:
                parse_kwargs["context"] = validation_context
            if strict is not None:
                parse_kwargs["strict"] = strict

            streaming_model = cast(Any, response_model)
            if is_async:
                return streaming_model.from_streaming_response_async(
                    response,
                    stream_extractor=self.extract_streaming_json_async,
                    **parse_kwargs,
                )

            return streaming_model.from_streaming_response(
                response,
                stream_extractor=self.extract_streaming_json,
                **parse_kwargs,
            )
        # Check for V1 native tool calls first
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_call = response.tool_calls[0]
            if hasattr(tool_call, "parameters"):
                json_str = json.dumps(tool_call.parameters)
                return response_model.model_validate_json(
                    json_str,
                    context=validation_context,
                    strict=strict,
                )

        # Fall back to text extraction
        text = _extract_text_from_response(response)
        extra_text = extract_json_from_codeblock(text)
        return response_model.model_validate_json(
            extra_text,
            context=validation_context,
            strict=strict,
        )


@register_mode_handler(Provider.COHERE, Mode.JSON_SCHEMA)
class CohereJSONSchemaHandler(CohereHandlerBase):
    """Handler for Cohere JSON_SCHEMA mode.

    Uses Cohere's native response_format with json_object type
    and schema enforcement.
    """

    mode = Mode.JSON_SCHEMA

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        new_kwargs = _convert_messages(kwargs)

        if response_model is None:
            return None, new_kwargs

        # Prepare response model for simple types
        from instructor.v2.core.response_model import prepare_response_model

        prepared_model = prepare_response_model(response_model)
        assert prepared_model is not None  # Already checked response_model is not None

        # Set response_format with JSON schema
        new_kwargs["response_format"] = {
            "type": "json_object",
            "schema": prepared_model.model_json_schema(),
        }

        return prepared_model, new_kwargs

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,
        is_async: bool = False,  # noqa: ARG002
    ) -> BaseModel:
        if stream:
            raise ConfigurationError(
                "Streaming is not supported for Cohere in JSON_SCHEMA mode."
            )
        text = _extract_text_from_response(response)
        return response_model.model_validate_json(
            text,
            context=validation_context,
            strict=strict,
        )


@register_mode_handler(Provider.COHERE, Mode.MD_JSON)
class CohereMDJSONHandler(CohereHandlerBase):
    """Handler for Cohere MD_JSON mode.

    Extracts JSON from markdown code blocks in text responses.
    This is a fallback mode when structured outputs aren't available.
    """

    mode = Mode.MD_JSON

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        new_kwargs = _convert_messages(kwargs)

        if response_model is None:
            return None, new_kwargs

        # Prepare response model for simple types
        from instructor.v2.core.response_model import prepare_response_model

        prepared_model = prepare_response_model(response_model)
        assert prepared_model is not None  # Already checked response_model is not None

        schema = prepared_model.model_json_schema()

        # Add instruction to return JSON in markdown code block
        instruction = (
            f"Return your answer as JSON in a markdown code block.\n"
            f"Schema: {json.dumps(schema, indent=2)}"
        )

        if "messages" in new_kwargs:
            # V2 format: append to last message
            if new_kwargs["messages"]:
                last_msg = new_kwargs["messages"][-1]
                last_msg["content"] = f"{last_msg.get('content', '')}\n\n{instruction}"
        else:
            # V1 format: append to message
            message = new_kwargs.get("message", "")
            new_kwargs["message"] = f"{message}\n\n{instruction}"

        return prepared_model, new_kwargs

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,
        is_async: bool = False,  # noqa: ARG002
    ) -> BaseModel:
        if stream:
            raise ConfigurationError(
                "Streaming is not supported for Cohere in MD_JSON mode."
            )
        text = _extract_text_from_response(response)
        extra_text = extract_json_from_codeblock(text)
        return response_model.model_validate_json(
            extra_text,
            context=validation_context,
            strict=strict,
        )


__all__ = [
    "CohereToolsHandler",
    "CohereJSONSchemaHandler",
    "CohereMDJSONHandler",
]
