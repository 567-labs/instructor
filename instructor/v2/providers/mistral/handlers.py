"""Mistral v2 mode handlers.

This module implements mode handlers for Mistral AI using the v2 registry system.
Supports TOOLS, JSON_SCHEMA, and MD_JSON modes.

Mistral has its own API format that differs from OpenAI:
- Uses `chat.complete()` and `chat.complete_async()` instead of `chat.completions.create()`
- Uses `chat.stream()` and `chat.stream_async()` for streaming
- Tool calling uses `tool_choice="any"` instead of specific tool selection
- Structured outputs use `response_format_from_pydantic_model()` helper
"""

from __future__ import annotations

import inspect
import json
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Generator,
    Iterable as TypingIterable,
)
from textwrap import dedent
from typing import Any, cast, get_origin

from pydantic import BaseModel

from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.errors import (
    IncompleteOutputException,
    ResponseParsingError,
)
from instructor.v2.dsl.iterable import IterableBase
from instructor.v2.dsl.parallel import ParallelBase, get_types_array
from instructor.v2.dsl.simple_type import AdapterBase
from instructor.v2.core.multimodal import convert_messages as convert_messages_v1
from instructor.v2.core.json import (
    extract_json_from_codeblock,
    extract_json_from_stream,
    extract_json_from_stream_async,
)
from instructor.v2.providers.openai.schema import generate_openai_schema
from instructor.v2.core.messages import (
    copy_messages_for_mutation,
    dump_message,
    merge_consecutive_messages,
)
from instructor.v2.core.decorators import register_mode_handler
from instructor.v2.core.handler import ModeHandler
from instructor.v2.core.streaming import StreamingModelState


class MistralHandlerBase(StreamingModelState, ModeHandler):
    """Base class for Mistral handlers with shared utilities."""

    mode: Mode

    def extract_streaming_json(
        self, completion: TypingIterable[Any]
    ) -> Generator[str, None, None]:
        """Extract JSON chunks from Mistral streaming responses."""

        def _raw_chunks() -> Generator[str, None, None]:
            for chunk in completion:
                try:
                    if self.mode == Mode.TOOLS:
                        if not chunk.data.choices[0].delta.tool_calls:
                            continue
                        yield (
                            chunk.data.choices[0].delta.tool_calls[0].function.arguments
                        )
                    else:
                        yield chunk.data.choices[0].delta.content
                except AttributeError:
                    continue

        raw_chunks = _raw_chunks()
        if self.mode == Mode.MD_JSON:
            yield from extract_json_from_stream(raw_chunks)
            return
        yield from raw_chunks

    async def extract_streaming_json_async(
        self, completion: AsyncGenerator[Any, None]
    ) -> AsyncGenerator[str, None]:
        """Extract JSON chunks from Mistral async streams."""

        async def _raw_chunks() -> AsyncGenerator[str, None]:
            async for chunk in completion:
                try:
                    if self.mode == Mode.TOOLS:
                        if not chunk.data.choices[0].delta.tool_calls:
                            continue
                        yield (
                            chunk.data.choices[0].delta.tool_calls[0].function.arguments
                        )
                    else:
                        yield chunk.data.choices[0].delta.content
                except AttributeError:
                    continue

        raw_chunks = _raw_chunks()
        if self.mode == Mode.MD_JSON:
            async for chunk in extract_json_from_stream_async(raw_chunks):
                yield chunk
            return
        async for chunk in raw_chunks:
            yield chunk

    def convert_messages(
        self, messages: list[dict[str, Any]], autodetect_images: bool = False
    ) -> list[dict[str, Any]]:
        """Convert messages for Mistral-compatible multimodal payloads."""
        if self.mode == Mode.TOOLS:
            target_mode = Mode.MISTRAL_TOOLS
        elif self.mode == Mode.JSON_SCHEMA:
            target_mode = Mode.MISTRAL_STRUCTURED_OUTPUTS
        else:
            target_mode = Mode.MD_JSON
        return convert_messages_v1(
            messages, target_mode, autodetect_images=autodetect_images
        )

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

        streaming_model = cast(Any, response_model)
        task_parser = None
        if (
            self.mode == Mode.TOOLS
            and inspect.isclass(response_model)
            and issubclass(response_model, IterableBase)
        ):
            task_parser = streaming_model.tasks_from_task_list_chunks

        if inspect.isasyncgen(response) or isinstance(response, AsyncIterator):
            if task_parser is not None:
                parse_kwargs["task_parser"] = (
                    streaming_model.tasks_from_task_list_chunks_async
                )
            return streaming_model.from_streaming_response_async(
                response,
                stream_extractor=self.extract_streaming_json_async,
                **parse_kwargs,
            )

        if task_parser is not None:
            parse_kwargs["task_parser"] = task_parser
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

    def _extract_tool_call_json(self, response: Any) -> str:
        """Extract JSON from tool call response.

        Mistral returns tool call arguments as either a string or a dict,
        so we need to handle both cases.
        """
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls:
            # The model replied in prose instead of calling the tool (e.g.
            # models that don't honor tool_choice="any", refusals, or
            # OpenAI-compatible gateways). Raise a retryable parsing error so
            # the retry machinery re-asks instead of crashing with TypeError.
            raise ResponseParsingError(
                "No tool calls found in Mistral response",
                mode=self.mode.name,
                raw_response=response,
            )
        args = tool_calls[0].function.arguments
        if isinstance(args, dict):
            return json.dumps(args)
        return args

    def _extract_text_content(self, response: Any) -> str:
        """Extract text content from response."""
        return response.choices[0].message.content or ""


@register_mode_handler(Provider.MISTRAL, Mode.TOOLS)
class MistralToolsHandler(MistralHandlerBase):
    """Handler for Mistral TOOLS mode.

    Uses Mistral's tool calling API with tool_choice="any".
    """

    mode = Mode.TOOLS

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        """Prepare request with tool definitions."""
        new_kwargs = kwargs.copy()
        new_kwargs["messages"] = list(kwargs.get("messages", []))

        if response_model is None:
            return None, new_kwargs

        # Detect if this is a parallel tools request (Iterable[Union[...]])
        # When streaming, treat Iterable[T] as streaming instead of parallel tools.
        origin = get_origin(response_model)
        is_parallel = origin is TypingIterable and not new_kwargs.get("stream")

        # Prepare response model: wrap simple types in ModelAdapter
        if not is_parallel:
            from instructor.v2.core.response_model import prepare_response_model

            response_model = prepare_response_model(response_model)

        self._register_streaming_from_kwargs(response_model, new_kwargs)

        if is_parallel:
            # Handle parallel model - generate tools for each type
            the_types = get_types_array(cast(Any, response_model))
            tools = []
            for model_type in the_types:
                schema = generate_openai_schema(model_type)
                tools.append({"type": "function", "function": schema})
            new_kwargs["tools"] = tools
        else:
            schema = generate_openai_schema(response_model)
            new_kwargs["tools"] = [{"type": "function", "function": schema}]

        # Mistral uses tool_choice="any" to force tool use
        new_kwargs["tool_choice"] = "any"

        return response_model, new_kwargs

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        """Handle reask for tools mode."""
        kwargs = kwargs.copy()
        message = response.choices[0].message
        reask_msgs: list[Any] = [dump_message(message)]

        tool_calls = message.tool_calls or []
        if not tool_calls:
            # The model answered in prose without calling the tool, so there is
            # no tool_call_id to attach a tool-role message to. Fall back to a
            # plain user correction instead of iterating None.
            reask_msgs.append(
                {
                    "role": "user",
                    "content": (
                        f"Validation Error found:\n{exception}\n"
                        "Recall the function correctly, fix the errors"
                    ),
                }
            )

        for tool_call in tool_calls:
            reask_msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": (
                        f"Validation Error found:\n{exception}\n"
                        "Recall the function correctly, fix the errors"
                    ),
                }
            )

        kwargs["messages"].extend(reask_msgs)
        return kwargs

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool | None = None,
        is_async: bool = False,  # noqa: ARG002
    ) -> Any:
        """Parse tool call response."""
        # Check for streaming
        if self._should_parse_streaming(response_model, stream):
            return self._parse_streaming_response(
                response_model,
                response,
                validation_context,
                strict,
            )

        # Check for incomplete output
        if hasattr(response, "choices") and response.choices:
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            if finish_reason == "length":
                raise IncompleteOutputException(last_completion=response)

        # Handle parallel tools (Iterable[Union[...]])
        origin = get_origin(response_model)
        if origin is TypingIterable:
            the_types = get_types_array(response_model)  # type: ignore[arg-type]
            type_registry = {t.__name__: t for t in the_types}

            def parallel_generator() -> Generator[BaseModel, None, None]:
                for tool_call in response.choices[0].message.tool_calls or []:
                    name = tool_call.function.name
                    if name in type_registry:
                        model_class = type_registry[name]
                        args = tool_call.function.arguments
                        if isinstance(args, dict):
                            args = json.dumps(args)
                        yield model_class.model_validate_json(
                            args,
                            context=validation_context,
                            strict=strict,
                        )

            return parallel_generator()

        # Standard tool call parsing
        json_str = self._extract_tool_call_json(response)
        parsed = response_model.model_validate_json(
            json_str,
            context=validation_context,
            strict=strict,
        )
        return self._finalize_parsed_result(response_model, response, parsed)


@register_mode_handler(Provider.MISTRAL, Mode.JSON_SCHEMA)
class MistralJSONSchemaHandler(MistralHandlerBase):
    """Handler for Mistral structured outputs (JSON_SCHEMA mode).

    Uses Mistral's native structured outputs via response_format parameter.
    Requires the mistralai SDK's response_format_from_pydantic_model helper.
    """

    mode = Mode.JSON_SCHEMA

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        """Prepare request with JSON schema response format."""
        self._register_streaming_from_kwargs(response_model, kwargs)

        if response_model is None:
            return None, kwargs

        new_kwargs = kwargs.copy()
        new_kwargs["messages"] = list(kwargs.get("messages", []))

        # Use Mistral's helper to create response format
        from mistralai.extra import response_format_from_pydantic_model

        new_kwargs["response_format"] = response_format_from_pydantic_model(
            response_model
        )

        # Remove any tool-related kwargs
        new_kwargs.pop("tools", None)
        new_kwargs.pop("tool_choice", None)

        return response_model, new_kwargs

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        """Handle reask for JSON schema mode."""
        kwargs = kwargs.copy()
        reask_msgs = [
            {
                "role": "assistant",
                "content": response.choices[0].message.content,
            },
            {
                "role": "user",
                "content": (
                    f"Validation Error found:\n{exception}\n"
                    "Recall the function correctly, fix the errors"
                ),
            },
        ]
        kwargs["messages"].extend(reask_msgs)
        return kwargs

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool | None = None,
        is_async: bool = False,  # noqa: ARG002
    ) -> Any:
        """Parse JSON schema response."""
        # Check for streaming
        if self._should_parse_streaming(response_model, stream):
            return self._parse_streaming_response(
                response_model,
                response,
                validation_context,
                strict,
            )

        # Check for incomplete output
        if hasattr(response, "choices") and response.choices:
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            if finish_reason == "length":
                raise IncompleteOutputException(last_completion=response)

        text = self._extract_text_content(response)
        parsed = response_model.model_validate_json(
            text,
            context=validation_context,
            strict=strict,
        )
        return self._finalize_parsed_result(response_model, response, parsed)


@register_mode_handler(Provider.MISTRAL, Mode.MD_JSON)
class MistralMDJSONHandler(MistralHandlerBase):
    """Handler for MD_JSON mode - extract JSON from markdown code blocks."""

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
        new_kwargs["messages"] = merge_consecutive_messages(messages)

        return response_model, new_kwargs

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        """Handle reask for MD_JSON mode."""
        kwargs = kwargs.copy()
        reask_msgs = [
            {
                "role": "assistant",
                "content": response.choices[0].message.content,
            },
            {
                "role": "user",
                "content": (
                    f"Validation Error found:\n{exception}\n"
                    "Recall the function correctly, fix the errors"
                ),
            },
        ]
        kwargs["messages"].extend(reask_msgs)
        return kwargs

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool | None = None,
        is_async: bool = False,  # noqa: ARG002
    ) -> Any:
        """Parse JSON from markdown code block in response."""
        # Check for streaming
        if self._should_parse_streaming(response_model, stream):
            return self._parse_streaming_response(
                response_model,
                response,
                validation_context,
                strict,
            )

        # Check for incomplete output
        if hasattr(response, "choices") and response.choices:
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            if finish_reason == "length":
                raise IncompleteOutputException(last_completion=response)

        text = self._extract_text_content(response)
        json_str = extract_json_from_codeblock(text)
        parsed = response_model.model_validate_json(
            json_str,
            context=validation_context,
            strict=strict,
        )
        return self._finalize_parsed_result(response_model, response, parsed)


__all__ = [
    "MistralToolsHandler",
    "MistralJSONSchemaHandler",
    "MistralMDJSONHandler",
]
