"""xAI v2 mode handlers.

This module implements mode handlers for xAI's Grok models using the v2 registry system.
Supports TOOLS, JSON_SCHEMA, and MD_JSON modes.

The xAI SDK has a unique API that differs from OpenAI. It uses:
- `xchat.tool()` for defining tools
- `xchat.user()`, `xchat.assistant()`, `xchat.system()` for messages
- `chat.parse()` for JSON schema parsing
- `chat.sample()` for regular completions
"""

from __future__ import annotations

import inspect
import json
from textwrap import dedent
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Generator,
    Iterable as TypingIterable,
)
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel

if TYPE_CHECKING:
    from xai_sdk import chat as xchat
else:
    try:
        from xai_sdk import chat as xchat
    except ImportError:
        xchat = None

from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.dsl.iterable import IterableBase
from instructor.v2.dsl.parallel import (
    ParallelBase,
    ParallelModel,
    get_types_array,
    handle_parallel_model,
)
from instructor.v2.dsl.simple_type import AdapterBase
from instructor.v2.core.json import (
    extract_json_from_codeblock,
    extract_json_from_stream,
    extract_json_from_stream_async,
)
from instructor.v2.core.decorators import register_mode_handler
from instructor.v2.core.handler import ModeHandler
from instructor.v2.core.streaming import StreamingModelState
from instructor.v2.core.messages import copy_messages_for_mutation


def _convert_messages_with_chat(messages: list[dict[str, Any]], chat: Any) -> list[Any]:
    """Convert OpenAI-style messages to xAI format."""
    if chat is None:
        raise ImportError("xai_sdk is required for xAI provider")

    converted = []
    for m in messages:
        role = m["role"]
        content = m.get("content", "")
        if isinstance(content, str):
            c = chat.text(content)
        else:
            raise ValueError("Only string content supported for xAI provider")
        if role == "user":
            converted.append(chat.user(c))
        elif role == "assistant":
            converted.append(chat.assistant(c))
        elif role == "system":
            converted.append(chat.system(c))
        elif role == "tool":
            converted.append(chat.tool_result(content))
        else:
            raise ValueError(f"Unsupported role: {role}")
    return converted


def _convert_messages(messages: list[dict[str, Any]]) -> list[Any]:
    return _convert_messages_with_chat(messages, xchat)


def reask_xai_json(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """Handle reask for xAI JSON mode when validation fails."""
    kwargs = kwargs.copy()
    reask_msg = {
        "role": "user",
        "content": (
            "Validation Errors found:\n"
            f"{exception}\n"
            "Recall the function correctly, fix the errors found in the following attempt:\n"
            f"{response}"
        ),
    }
    kwargs["messages"].append(reask_msg)
    return kwargs


def reask_xai_tools(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """Handle reask for xAI tools mode when validation fails."""
    kwargs = kwargs.copy()

    assistant_msg = {
        "role": "assistant",
        "content": str(response),
    }
    kwargs["messages"].append(assistant_msg)

    reask_msg = {
        "role": "user",
        "content": (
            "Validation Error found:\n"
            f"{exception}\n"
            "Recall the function correctly, fix the errors"
        ),
    }
    kwargs["messages"].append(reask_msg)
    return kwargs


def handle_xai_json(
    response_model: type[Any] | None, new_kwargs: dict[str, Any]
) -> tuple[type[Any] | None, dict[str, Any]]:
    """Handle xAI JSON mode."""
    messages = new_kwargs.get("messages", [])
    new_kwargs["x_messages"] = _convert_messages(messages)

    new_kwargs.pop("max_retries", None)
    new_kwargs.pop("context", None)
    new_kwargs.pop("hooks", None)

    return response_model, new_kwargs


def handle_xai_tools(
    response_model: type[Any] | None, new_kwargs: dict[str, Any]
) -> tuple[type[Any] | None, dict[str, Any]]:
    """Handle xAI tools mode."""
    messages = new_kwargs.get("messages", [])
    new_kwargs["x_messages"] = _convert_messages(messages)

    new_kwargs.pop("max_retries", None)
    new_kwargs.pop("context", None)
    new_kwargs.pop("hooks", None)

    if response_model is not None and xchat is not None:
        new_kwargs["tool"] = xchat.tool(
            name=response_model.__name__,
            description=response_model.__doc__ or "",
            parameters=response_model.model_json_schema(),
        )

    return response_model, new_kwargs


class XAIHandlerBase(StreamingModelState, ModeHandler):
    """Base class for xAI handlers with shared utilities."""

    mode: Mode

    def extract_streaming_json(
        self, completion: TypingIterable[Any]
    ) -> Generator[str, None, None]:
        """Extract JSON chunks from xAI streaming responses."""

        def _raw_chunks() -> Generator[str, None, None]:
            last_tool_args: dict[str, str] = {}
            last_args_value = ""
            for chunk in completion:
                choices = getattr(chunk, "choices", None)
                for choice in choices or [chunk]:
                    delta = getattr(choice, "delta", None)
                    if delta is None:
                        continue
                    if self.mode in {Mode.JSON_SCHEMA, Mode.MD_JSON}:
                        json_chunk = getattr(delta, "content", None)
                        if json_chunk:
                            yield json_chunk
                    elif self.mode == Mode.TOOLS:
                        tool_calls = getattr(delta, "tool_calls", None) or []
                        for index, tool_call in enumerate(tool_calls):
                            function = getattr(tool_call, "function", None)
                            json_chunk = getattr(function, "arguments", None)
                            if not json_chunk:
                                continue
                            tool_id = getattr(tool_call, "id", None) or str(index)
                            previous = last_tool_args.get(tool_id, "")
                            if previous and json_chunk.startswith(previous):
                                json_chunk = json_chunk[len(previous) :]
                            elif last_args_value and json_chunk.startswith(
                                last_args_value
                            ):
                                json_chunk = json_chunk[len(last_args_value) :]
                            last_tool_args[tool_id] = getattr(function, "arguments", "")
                            last_args_value = getattr(function, "arguments", "")
                            if json_chunk:
                                yield json_chunk

        raw_chunks = _raw_chunks()
        if self.mode == Mode.MD_JSON:
            yield from extract_json_from_stream(raw_chunks)
            return
        yield from raw_chunks

    async def extract_streaming_json_async(
        self, completion: AsyncGenerator[Any, None]
    ) -> AsyncGenerator[str, None]:
        """Extract JSON chunks from xAI async streams."""

        async def _raw_chunks() -> AsyncGenerator[str, None]:
            last_tool_args: dict[str, str] = {}
            last_args_value = ""
            async for chunk in completion:
                choices = getattr(chunk, "choices", None)
                for choice in choices or [chunk]:
                    delta = getattr(choice, "delta", None)
                    if delta is None:
                        continue
                    if self.mode in {Mode.JSON_SCHEMA, Mode.MD_JSON}:
                        json_chunk = getattr(delta, "content", None)
                        if json_chunk:
                            yield json_chunk
                    elif self.mode == Mode.TOOLS:
                        tool_calls = getattr(delta, "tool_calls", None) or []
                        for index, tool_call in enumerate(tool_calls):
                            function = getattr(tool_call, "function", None)
                            json_chunk = getattr(function, "arguments", None)
                            if not json_chunk:
                                continue
                            tool_id = getattr(tool_call, "id", None) or str(index)
                            previous = last_tool_args.get(tool_id, "")
                            if previous and json_chunk.startswith(previous):
                                json_chunk = json_chunk[len(previous) :]
                            elif last_args_value and json_chunk.startswith(
                                last_args_value
                            ):
                                json_chunk = json_chunk[len(last_args_value) :]
                            last_tool_args[tool_id] = getattr(function, "arguments", "")
                            last_args_value = getattr(function, "arguments", "")
                            if json_chunk:
                                yield json_chunk

        raw_chunks = _raw_chunks()
        if self.mode == Mode.MD_JSON:
            async for chunk in extract_json_from_stream_async(raw_chunks):
                yield chunk
            return
        async for chunk in raw_chunks:
            yield chunk

    def _parse_streaming_response(
        self,
        response_model: type[BaseModel],
        response: Any,
        validation_context: dict[str, Any] | None,
        strict: bool | None,
    ) -> Any:
        """Parse a streaming response using DSL methods."""
        parse_kwargs: dict[str, Any] = {}
        if validation_context is not None:
            parse_kwargs["context"] = validation_context
        if strict is not None:
            parse_kwargs["strict"] = strict

        if inspect.isasyncgen(response) or isinstance(response, AsyncIterator):
            if inspect.isclass(response_model) and issubclass(
                response_model, IterableBase
            ):

                async def _iter_tasks() -> AsyncGenerator[BaseModel, None]:
                    buffer = ""
                    async for chunk in self.extract_streaming_json_async(response):
                        buffer += chunk
                        try:
                            data = json.loads(buffer)
                        except json.JSONDecodeError:
                            continue
                        for item in data.get("tasks", []):
                            yield response_model.extract_cls_task_type(  # type: ignore[attr-defined]
                                json.dumps(item), **parse_kwargs
                            )
                        break

                return _iter_tasks()

            streaming_model = cast(Any, response_model)
            return streaming_model.from_streaming_response_async(
                response,
                stream_extractor=self.extract_streaming_json_async,
                **parse_kwargs,
            )

        if inspect.isclass(response_model) and issubclass(response_model, IterableBase):

            def _iter_tasks() -> Generator[BaseModel, None, None]:
                buffer = ""
                for chunk in self.extract_streaming_json(response):
                    buffer += chunk
                    try:
                        data = json.loads(buffer)
                    except json.JSONDecodeError:
                        continue
                    for item in data.get("tasks", []):
                        yield response_model.extract_cls_task_type(  # type: ignore[attr-defined]
                            json.dumps(item), **parse_kwargs
                        )
                    break

            generator = _iter_tasks()
        else:
            streaming_model = cast(Any, response_model)
            generator = streaming_model.from_streaming_response(
                response,
                stream_extractor=self.extract_streaming_json,
                **parse_kwargs,
            )
        if inspect.isclass(response_model) and issubclass(response_model, IterableBase):
            return generator
        return list(generator)

    def _finalize_parsed_result(
        self,
        response_model: type[BaseModel] | ParallelBase,
        response: Any,
        parsed: Any,
    ) -> Any:
        """Finalize parsed result, handling DSL types."""
        if isinstance(parsed, IterableBase):
            return [task for task in parsed.tasks]
        if isinstance(response_model, ParallelBase):
            return parsed
        if isinstance(parsed, AdapterBase):
            return parsed.content
        if isinstance(parsed, BaseModel):
            parsed._raw_response = response  # type: ignore[attr-defined]
        return parsed


@register_mode_handler(Provider.XAI, Mode.TOOLS)
class XAIToolsHandler(XAIHandlerBase):
    """Handler for xAI TOOLS mode.

    Uses xAI's tool calling API for structured output extraction.
    """

    mode = Mode.TOOLS

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        """Prepare request with tool definitions for xAI."""
        new_kwargs = kwargs.copy()
        new_kwargs["messages"] = list(kwargs.get("messages", []))

        if response_model is None:
            return None, new_kwargs

        from instructor.v2.core.response_model import prepare_response_model

        prepared_model = prepare_response_model(response_model)
        assert prepared_model is not None  # Already checked response_model is not None
        self._register_streaming_from_kwargs(prepared_model, new_kwargs)

        # Generate tool schema
        schema = prepared_model.model_json_schema()
        tool_name = getattr(prepared_model, "__name__", "response")
        tool_description = prepared_model.__doc__ or ""

        # Store tool info for xAI SDK
        new_kwargs["_xai_tool"] = {
            "name": tool_name,
            "description": tool_description,
            "parameters": schema,
        }

        return prepared_model, new_kwargs

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        return reask_xai_tools(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool | None = None,
        is_async: bool = False,  # noqa: ARG002
    ) -> Any:
        """Parse tool call response from xAI."""
        # Check for streaming
        if self._should_parse_streaming(response_model, stream):
            return self._parse_streaming_response(
                response_model,
                response,
                validation_context,
                strict,
            )

        # Handle xAI response format
        # xAI returns tool_calls in the response
        if hasattr(response, "tool_calls") and response.tool_calls:
            args = response.tool_calls[0].function.arguments
            if isinstance(args, dict):
                args = json.dumps(args)
            parsed = response_model.model_validate_json(
                args,
                context=validation_context,
                strict=strict,
            )
            return self._finalize_parsed_result(response_model, response, parsed)

        # Fallback: try to extract from text content
        text_content = ""
        if hasattr(response, "text") and response.text:
            text_content = str(response.text)
        elif hasattr(response, "content") and response.content:
            content = response.content
            if isinstance(content, str):
                text_content = content
            elif isinstance(content, list) and content:
                text_content = str(content[0])

        if text_content:
            json_str = extract_json_from_codeblock(text_content)
            parsed = response_model.model_validate_json(
                json_str,
                context=validation_context,
                strict=strict,
            )
            return self._finalize_parsed_result(response_model, response, parsed)

        raise ValueError(
            f"No tool calls returned from xAI and no text content available. "
            f"Response: {response}"
        )


@register_mode_handler(Provider.XAI, Mode.PARALLEL_TOOLS)
class XAIParallelToolsHandler(XAIHandlerBase):
    """Handler for xAI parallel tool calling."""

    mode = Mode.PARALLEL_TOOLS

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        """Prepare request for parallel tool calling."""
        if response_model is None:
            return None, kwargs

        new_kwargs = kwargs.copy()
        new_kwargs["messages"] = list(kwargs.get("messages", []))
        if new_kwargs.get("stream", False):
            from instructor.v2.core.errors import ConfigurationError

            raise ConfigurationError(
                "stream=True is not supported when using PARALLEL_TOOLS mode"
            )

        new_kwargs["_xai_tools"] = handle_parallel_model(response_model)  # type: ignore[arg-type]
        parallel_model = ParallelModel(cast(Any, response_model))
        return cast(type[BaseModel], parallel_model), new_kwargs

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        """Handle reask for parallel tools mode."""
        return reask_xai_tools(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,  # noqa: ARG002
        is_async: bool = False,  # noqa: ARG002
    ) -> Any:
        """Parse parallel tool calls from xAI."""
        the_types = get_types_array(response_model)  # type: ignore[arg-type]
        type_registry = {model.__name__: model for model in the_types}

        results = []
        for tool_call in getattr(response, "tool_calls", []) or []:
            name = tool_call.function.name
            args = tool_call.function.arguments
            if isinstance(args, dict):
                args = json.dumps(args)
            if name in type_registry:
                results.append(
                    type_registry[name].model_validate_json(
                        args,
                        context=validation_context,
                        strict=strict,
                    )
                )
        return iter(results)


@register_mode_handler(Provider.XAI, Mode.JSON_SCHEMA)
class XAIJSONSchemaHandler(XAIHandlerBase):
    """Handler for xAI JSON_SCHEMA mode.

    Uses xAI's native JSON schema parsing via chat.parse().
    """

    mode = Mode.JSON_SCHEMA

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        """Prepare request with JSON schema for xAI."""
        self._register_streaming_from_kwargs(response_model, kwargs)

        if response_model is None:
            return None, kwargs

        new_kwargs = kwargs.copy()
        new_kwargs["messages"] = list(kwargs.get("messages", []))
        schema = response_model.model_json_schema()

        # Store schema info for xAI SDK's parse() method
        new_kwargs["_xai_json_schema"] = {
            "schema": schema,
            "name": response_model.__name__,
        }

        return response_model, new_kwargs

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        return reask_xai_json(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool | None = None,
        is_async: bool = False,  # noqa: ARG002
    ) -> Any:
        """Parse JSON schema response from xAI."""
        # Check for streaming
        if self._should_parse_streaming(response_model, stream):
            return self._parse_streaming_response(
                response_model,
                response,
                validation_context,
                strict,
            )

        # xAI's parse() returns (raw_response, parsed_model)
        # If we receive a tuple, extract the parsed model
        if isinstance(response, tuple) and len(response) == 2:
            raw_response, parsed = response
            if isinstance(parsed, BaseModel):
                parsed._raw_response = raw_response  # type: ignore[attr-defined]
                return parsed

        # Handle direct response object
        text_content = ""
        if hasattr(response, "text") and response.text:
            text_content = str(response.text)
        elif hasattr(response, "content") and response.content:
            content = response.content
            if isinstance(content, str):
                text_content = content
            elif isinstance(content, list) and content:
                text_content = str(content[0])

        if text_content:
            parsed = response_model.model_validate_json(
                text_content,
                context=validation_context,
                strict=strict,
            )
            return self._finalize_parsed_result(response_model, response, parsed)

        raise ValueError(f"Could not parse xAI response: {response}")


@register_mode_handler(Provider.XAI, Mode.MD_JSON)
class XAIMDJSONHandler(XAIHandlerBase):
    """Handler for xAI MD_JSON mode.

    Extracts JSON from markdown code blocks in text responses.
    """

    mode = Mode.MD_JSON

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        """Prepare request with JSON schema instruction in messages."""
        self._register_streaming_from_kwargs(response_model, kwargs)

        if response_model is None:
            return None, kwargs

        new_kwargs = kwargs.copy()
        schema = response_model.model_json_schema()

        message = dedent(
            f"""
            As a genius expert, your task is to understand the content and provide
            the parsed objects in json that match the following json_schema:

            {json.dumps(schema, indent=2, ensure_ascii=False)}

            Make sure to return an instance of the JSON, not the schema itself.
            Return the JSON in a markdown code block.
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
        elif (
            messages
            and isinstance(messages[0]["content"], list)
            and messages[0]["content"]
        ):
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
        new_kwargs["messages"] = messages

        return response_model, new_kwargs

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        return reask_xai_json(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool | None = None,
        is_async: bool = False,  # noqa: ARG002
    ) -> Any:
        """Parse JSON from markdown code block in xAI response."""
        # Check for streaming
        if self._should_parse_streaming(response_model, stream):
            return self._parse_streaming_response(
                response_model,
                response,
                validation_context,
                strict,
            )

        # Extract text content from response
        text_content = ""
        if hasattr(response, "text") and response.text:
            text_content = str(response.text)
        elif hasattr(response, "content") and response.content:
            content = response.content
            if isinstance(content, str):
                text_content = content
            elif isinstance(content, list) and content:
                text_content = str(content[0])

        if text_content:
            json_str = extract_json_from_codeblock(text_content)
            parsed = response_model.model_validate_json(
                json_str,
                context=validation_context,
                strict=strict,
            )
            return self._finalize_parsed_result(response_model, response, parsed)

        raise ValueError(f"Could not extract JSON from xAI response: {response}")


__all__ = [
    "handle_xai_json",
    "handle_xai_tools",
    "reask_xai_json",
    "reask_xai_tools",
    "XAIToolsHandler",
    "XAIParallelToolsHandler",
    "XAIJSONSchemaHandler",
    "XAIMDJSONHandler",
]
