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
from functools import partial
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Generator,
    Iterable as TypingIterable,
)
from typing import Any, cast, get_origin

from pydantic import BaseModel

from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.errors import (
    IncompleteOutputException,
    ResponseParsingError,
)
from instructor.v2.dsl.iterable import IterableBase
from instructor.v2.dsl.parallel import (
    ParallelBase as ParallelBase,
    ParallelModel,
    model_for_tool_name,
)
from instructor.v2.dsl.simple_type import AdapterBase as AdapterBase
from instructor.v2.core.multimodal import convert_messages as convert_messages_v1
from instructor.v2.providers.openai.handlers import OpenAIHandlerBase
from instructor.v2.providers.openai.schema import generate_openai_schema
from instructor.v2.core.messages import (
    add_json_schema_prompt,
    dump_message,
    first_choice_message,
    text_from_content_blocks,
)
from instructor.v2.core.decorators import register_mode_handler
from instructor.v2.core.handler import ModeHandler
from instructor.v2.core.streaming import (
    StreamingModelState,
    adrain_stream,
    drain_stream,
)


class MistralHandlerBase(StreamingModelState, ModeHandler):
    """Base class for Mistral handlers with shared utilities."""

    mode: Mode

    def _require_choice_message(self, response: Any) -> Any:
        message = first_choice_message(response)
        if message is None:
            raise ResponseParsingError(
                "No choices in Mistral response",
                mode=self.mode.name,
                raw_response=response,
            )
        return message

    _parse_json_response = OpenAIHandlerBase._parse_json_response

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        """Replay the assistant text (if any) with a correction prompt."""
        kwargs = kwargs.copy()
        message = first_choice_message(response)
        if message is not None:
            kwargs["messages"].append({"role": "assistant", "content": message.content})
        kwargs["messages"].append(
            {
                "role": "user",
                "content": (
                    f"Validation Error found:\n{exception}\n"
                    "Recall the function correctly, fix the errors"
                ),
            }
        )
        return kwargs

    def _stream_chunk_json(self, chunk: Any) -> str | None:
        """Return the JSON fragment carried by one stream chunk, if any."""
        delta = chunk.data.choices[0].delta
        if self.mode == Mode.TOOLS:
            return delta.tool_calls[0].function.arguments if delta.tool_calls else None
        content = delta.content
        if isinstance(content, list):
            # Reasoning models stream [ThinkChunk, TextChunk] deltas.
            return text_from_content_blocks(content)
        return content if isinstance(content, str) else None

    @staticmethod
    def _is_length_stop(chunk: Any) -> bool:
        try:
            return chunk.data.choices[0].finish_reason == "length"
        except (AttributeError, IndexError):
            return False

    def extract_streaming_json(
        self, completion: TypingIterable[Any], *, single_model: bool = False
    ) -> Generator[str, None, None]:
        """Extract JSON chunks from Mistral streaming responses."""
        return drain_stream(
            completion,
            self._stream_chunk_json,
            truncated=None if single_model else self._is_length_stop,
            extract_json=self.mode == Mode.MD_JSON,
            single_model=single_model,
        )

    def extract_streaming_json_async(
        self, completion: AsyncGenerator[Any, None], *, single_model: bool = False
    ) -> AsyncGenerator[str, None]:
        """Extract JSON chunks from Mistral async streams."""
        return adrain_stream(
            completion,
            self._stream_chunk_json,
            truncated=None if single_model else self._is_length_stop,
            extract_json=self.mode == Mode.MD_JSON,
            single_model=single_model,
        )

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
                stream_extractor=partial(
                    self.extract_streaming_json_async,
                    single_model=not issubclass(response_model, IterableBase),
                ),
                **parse_kwargs,
            )

        if task_parser is not None:
            parse_kwargs["task_parser"] = task_parser
        generator = streaming_model.from_streaming_response(
            response,
            stream_extractor=partial(
                self.extract_streaming_json,
                single_model=not issubclass(response_model, IterableBase),
            ),
            **parse_kwargs,
        )
        if inspect.isclass(response_model) and issubclass(response_model, IterableBase):
            return generator
        return list(generator)

    def _extract_tool_call_json(self, response: Any) -> str:
        """Extract JSON from tool call response.

        Mistral returns tool call arguments as either a string or a dict,
        so we need to handle both cases.
        """
        message = self._require_choice_message(response)
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
        message = self._require_choice_message(response)
        content = message.content
        if isinstance(content, list):
            # Reasoning models (Magistral) return [ThinkChunk, TextChunk].
            return text_from_content_blocks(content) or ""
        return content or ""


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
            the_types = ParallelModel(cast(Any, response_model)).models
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
        # With no completion choice there is no assistant message to replay;
        # fall through to the plain user correction below.
        message = first_choice_message(response)
        reask_msgs: list[Any] = [dump_message(message)] if message is not None else []

        tool_calls = (message.tool_calls or []) if message is not None else []
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
            type_registry: dict[str, type[BaseModel]] = ParallelModel(
                cast(Any, response_model)
            ).registry
            message = self._require_choice_message(response)
            results = []
            for tool_call in message.tool_calls or []:
                model_class = model_for_tool_name(
                    type_registry,
                    tool_call.function.name,
                    mode=self.mode,
                    raw_response=response,
                )
                args = tool_call.function.arguments
                if isinstance(args, dict):
                    args = json.dumps(args)
                results.append(
                    model_class.model_validate_json(
                        args, context=validation_context, strict=strict
                    )
                )
            return iter(results)

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
        return self._parse_json_response(
            response, response_model, validation_context, strict, stream
        )


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
        new_kwargs["messages"] = add_json_schema_prompt(
            new_kwargs.get("messages", []),
            response_model.model_json_schema(),
            codeblock=True,
        )
        return response_model, new_kwargs

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
        return self._parse_json_response(
            response, response_model, validation_context, strict, stream, codeblock=True
        )


__all__ = [
    "MistralToolsHandler",
    "MistralJSONSchemaHandler",
    "MistralMDJSONHandler",
]
