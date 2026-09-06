"""OpenAI v2 mode handlers with DSL-aware parsing.

This module implements mode handlers for OpenAI using the v2 registry system.
Supports TOOLS, JSON_SCHEMA, MD_JSON, PARALLEL_TOOLS, and RESPONSES_TOOLS modes.
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Generator,
    Iterable as TypingIterable,
)
from textwrap import dedent
from typing import TYPE_CHECKING, Any, cast, get_origin
from weakref import WeakKeyDictionary

from pydantic import BaseModel

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openai.types.chat import ChatCompletion

from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.errors import (
    ConfigurationError,
    IncompleteOutputException,
    ResponseParsingError,
)
from instructor.v2.dsl.iterable import IterableBase
from instructor.v2.dsl.parallel import ParallelBase, ParallelModel, get_types_array
from instructor.v2.dsl.partial import PartialBase
from instructor.v2.dsl.simple_type import AdapterBase
from instructor.v2.core.multimodal import convert_messages as convert_messages_v1
from instructor.v2.core.json import (
    extract_json_from_codeblock,
    extract_json_from_stream,
    extract_json_from_stream_async,
)
from instructor.v2.core.messages import (
    copy_messages_for_mutation,
    dump_message,
    merge_consecutive_messages,
)
from instructor.v2.providers.openai.schema import generate_openai_schema
from instructor.v2.core.decorators import register_mode_handler
from instructor.v2.core.handler import ModeHandler

logger = logging.getLogger("instructor")


OPENAI_COMPAT_PROVIDERS = [
    Provider.OPENAI,
    Provider.HUBRIS,
    Provider.ANYSCALE,
    Provider.TOGETHER,
    Provider.DATABRICKS,
    Provider.DEEPSEEK,
    Provider.OPENROUTER,
    Provider.GROQ,
    Provider.FIREWORKS,
    Provider.CEREBRAS,
]

OPENAI_PARALLEL_TOOL_PROVIDERS = [
    Provider.OPENAI,
    Provider.HUBRIS,
    Provider.ANYSCALE,
    Provider.TOGETHER,
    Provider.DATABRICKS,
    Provider.DEEPSEEK,
    Provider.OPENROUTER,
    Provider.CEREBRAS,
]

OPENAI_JSON_SCHEMA_PROVIDERS = [
    Provider.OPENAI,
    Provider.HUBRIS,
    Provider.ANYSCALE,
    Provider.TOGETHER,
    Provider.DATABRICKS,
    Provider.DEEPSEEK,
    Provider.GROQ,
    Provider.FIREWORKS,
    Provider.CEREBRAS,
]


def _is_stream_response(response: Any) -> bool:
    """Check if response is a Stream object rather than a ChatCompletion."""
    return response is None or not hasattr(response, "choices")


def _filter_responses_tool_calls(output_items: list[Any]) -> list[Any]:
    """Return response output items that represent tool calls."""
    tool_calls: list[Any] = []
    for item in output_items:
        item_type = getattr(item, "type", None)
        if item_type in {"function_call", "tool_call"}:
            tool_calls.append(item)
            continue
        if item_type is None and hasattr(item, "arguments"):
            tool_calls.append(item)
    return tool_calls


def _format_responses_tool_call_details(tool_call: Any) -> str:
    """Format tool call name/id details for reask messages."""
    tool_name = getattr(tool_call, "name", None)
    tool_id = (
        getattr(tool_call, "id", None)
        or getattr(tool_call, "call_id", None)
        or getattr(tool_call, "tool_call_id", None)
    )
    details: list[str] = []
    if tool_name:
        details.append(f"name={tool_name}")
    if tool_id:
        details.append(f"id={tool_id}")
    if not details:
        return ""
    return f" (tool call {', '.join(details)})"


def _ensure_text_format_config(
    new_kwargs: dict[str, Any],
    response_model: type[Any],
    parameters_schema: dict[str, Any],
) -> None:
    """Align Responses API text.format with the forced tool schema."""
    desired_format = {
        "type": "json_schema",
        "name": response_model.__name__,
        "strict": True,
        "schema": parameters_schema,
    }

    existing_text = new_kwargs.get("text")
    if isinstance(existing_text, dict):
        existing_format = existing_text.get("format")
        if existing_format == desired_format:
            return
        if isinstance(existing_format, dict):
            logger.debug(
                "Overriding user-provided text.format with tool schema "
                "to prevent competing Responses output paths."
            )
        updated_text = existing_text.copy()
        updated_text["format"] = desired_format
        new_kwargs["text"] = updated_text
        return

    new_kwargs["text"] = {"format": desired_format}


def reask_tools(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """Handle reask for OpenAI tools mode when validation fails."""
    kwargs = kwargs.copy()

    if _is_stream_response(response):
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

    message = response.choices[0].message
    reask_msgs: list[Any] = [dump_message(message)]
    tool_calls = message.tool_calls or []
    if not tool_calls:
        # Model replied without calling the tool (e.g. OpenAI-compatible
        # providers that don't honor forced tool_choice). Fall back to a plain
        # user correction instead of iterating None, mirroring the streaming
        # branch and the GenAI reask handler.
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


def reask_responses_tools(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """Handle reask for OpenAI responses tools mode when validation fails."""
    kwargs = kwargs.copy()

    if response is None or not hasattr(response, "output"):
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

    reask_messages = []
    for tool_call in _filter_responses_tool_calls(response.output):
        details = _format_responses_tool_call_details(tool_call)
        args = getattr(tool_call, "arguments", "")
        is_empty_args = not args or (
            isinstance(args, str) and args.strip() in {"", "{}"}
        )
        if is_empty_args:
            reask_messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Validation Error found:\n{exception}\n"
                        "The function was called with empty arguments '{}'. "
                        "You MUST populate ALL required fields in the function "
                        "call according to the schema. Do not return empty "
                        f"arguments.{details}"
                    ),
                }
            )
            continue

        reask_messages.append(
            {
                "role": "user",
                "content": (
                    f"Validation Error found:\n{exception}\n"
                    "Recall the function correctly, fix the errors with "
                    f"{args}{details}"
                ),
            }
        )

    if not reask_messages:
        # Model produced no tool calls at all (e.g. a reasoning-only or plain
        # message output). Fall back to a plain user correction so the retry
        # carries feedback instead of resending the identical request,
        # mirroring reask_tools and the Anthropic reask handler.
        reask_messages.append(
            {
                "role": "user",
                "content": (
                    f"Validation Error found:\n{exception}\n"
                    "Recall the function correctly, fix the errors"
                ),
            }
        )

    kwargs["messages"].extend(reask_messages)
    return kwargs


def reask_md_json(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """Handle reask for OpenAI JSON modes when validation fails."""
    kwargs = kwargs.copy()

    if _is_stream_response(response):
        kwargs["messages"].append(
            {
                "role": "user",
                "content": (
                    "Correct your JSON ONLY RESPONSE, based on the following errors:\n"
                    f"{exception}"
                ),
            }
        )
        return kwargs

    message = response.choices[0].message
    try:
        assistant_message = dump_message(message)
    except (AttributeError, KeyError, TypeError):
        assistant_message = {
            "role": getattr(message, "role", "assistant"),
            "content": getattr(message, "content", "") or "",
        }

    reask_msgs = [assistant_message]
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


def reask_default(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """Handle reask for OpenAI default mode when validation fails."""
    kwargs = kwargs.copy()

    if _is_stream_response(response):
        kwargs["messages"].append(
            {
                "role": "user",
                "content": (
                    "Recall the function correctly, fix the errors, exceptions found\n"
                    f"{exception}"
                ),
            }
        )
        return kwargs

    reask_msgs = [dump_message(response.choices[0].message)]
    reask_msgs.append(
        {
            "role": "user",
            "content": (
                "Recall the function correctly, fix the errors, exceptions found\n"
                f"{exception}"
            ),
        }
    )
    kwargs["messages"].extend(reask_msgs)
    return kwargs


def handle_openrouter_structured_outputs(
    response_model: type[Any], new_kwargs: dict[str, Any]
) -> tuple[type[Any], dict[str, Any]]:
    """Handle OpenRouter structured outputs mode."""
    schema = response_model.model_json_schema()
    schema["additionalProperties"] = False
    new_kwargs["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "name": response_model.__name__,
            "schema": schema,
            "strict": True,
        },
    }
    return response_model, new_kwargs


class OpenAIHandlerBase(ModeHandler):
    """Base class for OpenAI handlers with shared utilities."""

    mode: Mode

    def __init__(self) -> None:
        self._streaming_models: WeakKeyDictionary[type[Any], None] = WeakKeyDictionary()

    def _register_streaming_from_kwargs(
        self, response_model: type[BaseModel] | None, kwargs: dict[str, Any]
    ) -> None:
        """Register model for streaming if stream=True in kwargs."""
        if response_model is None:
            return
        if kwargs.get("stream"):
            self.mark_streaming_model(response_model, True)

    def mark_streaming_model(
        self, response_model: type[BaseModel] | None, stream: bool
    ) -> None:
        """Record that the response model expects streaming output."""
        if not stream or response_model is None:
            return
        if inspect.isclass(response_model) and issubclass(
            response_model, (IterableBase, PartialBase)
        ):
            self._streaming_models[response_model] = None

    def _consume_streaming_flag(
        self, response_model: type[BaseModel] | ParallelBase | None
    ) -> bool:
        """Check and consume streaming flag for a model."""
        if response_model is None:
            return False
        if not inspect.isclass(response_model):
            return False
        if response_model in self._streaming_models:
            del self._streaming_models[response_model]
            return True
        return False

    def _should_parse_streaming(
        self,
        response_model: type[BaseModel] | ParallelBase | None,
        stream: bool,
    ) -> bool:
        """Return whether a response needs DSL streaming parsing."""
        registered = self._consume_streaming_flag(response_model)
        return bool(
            inspect.isclass(response_model)
            and issubclass(response_model, (IterableBase, PartialBase))
            and (stream or registered)
        )

    def extract_streaming_json(
        self, completion: TypingIterable[Any]
    ) -> Generator[str, None, None]:
        """Extract JSON chunks from OpenAI-compatible streaming responses."""

        def _raw_chunks() -> Generator[str, None, None]:
            for chunk in completion:
                try:
                    if self.mode == Mode.RESPONSES_TOOLS:
                        from openai.types.responses import (
                            ResponseFunctionCallArgumentsDeltaEvent,
                        )

                        if isinstance(chunk, ResponseFunctionCallArgumentsDeltaEvent):
                            yield chunk.delta
                        continue

                    if not getattr(chunk, "choices", None):
                        continue

                    if self.mode == Mode.FUNCTIONS:
                        Mode.warn_mode_functions_deprecation()
                        if json_chunk := chunk.choices[0].delta.function_call.arguments:
                            yield json_chunk
                    elif self.mode in {
                        Mode.JSON,
                        Mode.MD_JSON,
                        Mode.JSON_SCHEMA,
                    }:
                        if json_chunk := chunk.choices[0].delta.content:
                            yield json_chunk
                    elif self.mode in {
                        Mode.TOOLS,
                        Mode.TOOLS_STRICT,
                        Mode.PARALLEL_TOOLS,
                    }:
                        if json_chunk := chunk.choices[0].delta.tool_calls:
                            if json_chunk[0].function.arguments is not None:
                                yield json_chunk[0].function.arguments
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
        """Extract JSON chunks from OpenAI-compatible async streams."""

        async def _raw_chunks() -> AsyncGenerator[str, None]:
            async for chunk in completion:
                try:
                    if self.mode == Mode.RESPONSES_TOOLS:
                        from openai.types.responses import (
                            ResponseFunctionCallArgumentsDeltaEvent,
                        )

                        if isinstance(chunk, ResponseFunctionCallArgumentsDeltaEvent):
                            yield chunk.delta
                        continue

                    if not getattr(chunk, "choices", None):
                        continue

                    if self.mode == Mode.FUNCTIONS:
                        Mode.warn_mode_functions_deprecation()
                        if json_chunk := chunk.choices[0].delta.function_call.arguments:
                            yield json_chunk
                    elif self.mode in {
                        Mode.JSON,
                        Mode.MD_JSON,
                        Mode.JSON_SCHEMA,
                    }:
                        if json_chunk := chunk.choices[0].delta.content:
                            yield json_chunk
                    elif self.mode in {
                        Mode.TOOLS,
                        Mode.TOOLS_STRICT,
                        Mode.PARALLEL_TOOLS,
                    }:
                        if json_chunk := chunk.choices[0].delta.tool_calls:
                            if json_chunk[0].function.arguments is not None:
                                yield json_chunk[0].function.arguments
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
        """Convert multimodal messages for OpenAI-compatible formats."""
        return convert_messages_v1(
            messages, self.mode, autodetect_images=autodetect_images
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
        streaming_kwargs = dict(parse_kwargs)
        try:
            if (
                "mode"
                in inspect.signature(streaming_model.from_streaming_response).parameters
            ):
                streaming_kwargs["mode"] = self.mode
        except (TypeError, ValueError):
            pass

        if inspect.isasyncgen(response) or isinstance(response, AsyncIterator):
            return streaming_model.from_streaming_response_async(
                response,
                stream_extractor=self.extract_streaming_json_async,
                **streaming_kwargs,
            )

        generator = streaming_model.from_streaming_response(
            response,
            stream_extractor=self.extract_streaming_json,
            **streaming_kwargs,
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
            return [task for task in parsed.tasks]  # type: ignore[attr-defined]
        if isinstance(response_model, ParallelBase):
            return parsed
        if isinstance(parsed, AdapterBase):
            return parsed.content  # type: ignore[attr-defined]
        if isinstance(parsed, BaseModel):
            parsed._raw_response = response  # type: ignore[attr-defined]
        return parsed

    def _extract_tool_call_json(self, response: Any) -> str:
        """Extract JSON from tool call response."""
        choices = getattr(response, "choices", None)
        if not choices:
            raise ResponseParsingError(
                "No choices in OpenAI response",
                mode=str(self.mode.value),
                raw_response=response,
            )
        message = choices[0].message
        refusal = getattr(message, "refusal", None)
        if refusal is not None:
            raise AssertionError(f"Unable to generate a response due to {refusal}")

        def _normalize_args(args: Any) -> str:
            if args is None:
                raise ResponseParsingError(
                    "Tool call arguments missing in response",
                    mode="TOOLS",
                    raw_response=response,
                )
            if isinstance(args, dict):
                return json.dumps(args)
            if isinstance(args, str):
                return args
            try:
                return json.dumps(args)
            except TypeError as exc:
                raise ResponseParsingError(
                    "Tool call arguments must be JSON-serializable",
                    mode="TOOLS",
                    raw_response=response,
                ) from exc

        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            return _normalize_args(tool_calls[0].function.arguments)

        function_call = getattr(message, "function_call", None)
        if function_call is not None:
            return _normalize_args(getattr(function_call, "arguments", None))

        raise ResponseParsingError(
            "No tool calls or function call found in response",
            mode="TOOLS",
            raw_response=response,
        )

    def _extract_text_content(self, response: Any) -> str:
        """Extract text content from response."""
        choices = getattr(response, "choices", None)
        if not choices:
            raise ResponseParsingError(
                "No choices in OpenAI response",
                mode=str(self.mode.value),
                raw_response=response,
            )
        return choices[0].message.content or ""


@register_mode_handler(OPENAI_COMPAT_PROVIDERS, Mode.TOOLS)
class OpenAIToolsHandler(OpenAIHandlerBase):
    """Handler for OpenAI TOOLS mode.

    Supports `strict=True` parameter for strict schema validation.
    """

    mode = Mode.TOOLS

    def prepare_request(
        self,
        response_model: Any,
        kwargs: dict[str, Any],
    ) -> tuple[Any, dict[str, Any]]:
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
            # Handle parallel model
            from instructor.v2.dsl.parallel import handle_parallel_model

            new_kwargs["tools"] = handle_parallel_model(cast(Any, response_model))
            new_kwargs["tool_choice"] = "auto"
        else:
            # Shallow-copy to avoid mutating the lru_cache return value.
            # generate_openai_schema caches the same dict object; writing
            # "strict" into it would permanently affect every future call for
            # this model, even when strict=False.
            schema = dict(generate_openai_schema(response_model))

            # Check for strict parameter
            use_strict = new_kwargs.pop("strict", False)
            if use_strict:
                schema["strict"] = True

            new_kwargs["tools"] = [{"type": "function", "function": schema}]
            new_kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": schema["name"]},
            }

        return response_model, new_kwargs

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: ChatCompletion,
        exception: Exception,
    ) -> dict[str, Any]:
        """Handle reask for tools mode."""
        return reask_tools(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,
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
            if response.choices[0].finish_reason == "length":
                raise IncompleteOutputException(last_completion=response)

        # Handle parallel tools (Iterable[Union[...]])
        origin = get_origin(response_model)
        if origin is TypingIterable:
            the_types = get_types_array(response_model)  # type: ignore[arg-type]
            type_registry = {t.__name__: t for t in the_types}

            def parallel_generator() -> Generator[BaseModel, None, None]:
                for tool_call in response.choices[0].message.tool_calls:
                    name = tool_call.function.name
                    if name in type_registry:
                        model_class = type_registry[name]
                        yield model_class.model_validate_json(
                            tool_call.function.arguments,
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


@register_mode_handler(OPENAI_JSON_SCHEMA_PROVIDERS, Mode.JSON_SCHEMA)
class OpenAIJSONSchemaHandler(OpenAIHandlerBase):
    """Handler for OpenAI structured outputs (JSON_SCHEMA mode).

    Uses OpenAI's native structured outputs via response_format parameter.
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
        schema = response_model.model_json_schema()
        new_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": schema,
            },
        }
        return response_model, new_kwargs

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: ChatCompletion,
        exception: Exception,
    ) -> dict[str, Any]:
        """Handle reask for JSON schema mode."""
        return reask_md_json(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,
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
            if response.choices[0].finish_reason == "length":
                raise IncompleteOutputException(last_completion=response)

        text = self._extract_text_content(response)
        parsed = response_model.model_validate_json(
            text,
            context=validation_context,
            strict=strict,
        )
        return self._finalize_parsed_result(response_model, response, parsed)


@register_mode_handler(OPENAI_COMPAT_PROVIDERS, Mode.JSON)
class OpenAIJSONHandler(OpenAIHandlerBase):
    """Handler for OpenAI JSON mode (response_format=json_object)."""

    mode = Mode.JSON

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        self._register_streaming_from_kwargs(response_model, kwargs)

        if response_model is None:
            return None, kwargs

        new_kwargs = kwargs.copy()
        new_kwargs["response_format"] = {"type": "json_object"}
        schema = response_model.model_json_schema()
        message = dedent(
            f"""
            As a genius expert, your task is to understand the content and provide
            the parsed objects in json that match the following json_schema:\n

            {json.dumps(schema, indent=2, ensure_ascii=False)}

            Make sure to return an instance of the JSON, not the schema itself
            """
        )
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
        new_kwargs["messages"] = merge_consecutive_messages(messages)
        return response_model, new_kwargs

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: ChatCompletion,
        exception: Exception,
    ) -> dict[str, Any]:
        return reask_md_json(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,
        is_async: bool = False,  # noqa: ARG002
    ) -> Any:
        if self._should_parse_streaming(response_model, stream):
            return self._parse_streaming_response(
                response_model,
                response,
                validation_context,
                strict,
            )

        if hasattr(response, "choices") and response.choices:
            if response.choices[0].finish_reason == "length":
                raise IncompleteOutputException(last_completion=response)

        text = self._extract_text_content(response)
        parsed = response_model.model_validate_json(
            text,
            context=validation_context,
            strict=strict,
        )
        return self._finalize_parsed_result(response_model, response, parsed)


@register_mode_handler(OPENAI_COMPAT_PROVIDERS, Mode.MD_JSON)
class OpenAIMDJSONHandler(OpenAIHandlerBase):
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
        response: ChatCompletion,
        exception: Exception,
    ) -> dict[str, Any]:
        """Handle reask for MD_JSON mode."""
        return reask_md_json(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,
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
            if response.choices[0].finish_reason == "length":
                raise IncompleteOutputException(last_completion=response)

        text = self._extract_text_content(response)
        json_str = extract_json_from_codeblock(text)
        parsed = response_model.model_validate_json(
            json_str,
            context=validation_context,
            strict=strict,
        )
        return self._finalize_parsed_result(response_model, response, parsed)


@register_mode_handler(OPENAI_PARALLEL_TOOL_PROVIDERS, Mode.PARALLEL_TOOLS)
class OpenAIParallelToolsHandler(OpenAIHandlerBase):
    """Handler for OpenAI parallel tool calling."""

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
            raise ConfigurationError(
                "stream=True is not supported when using PARALLEL_TOOLS mode"
            )

        from instructor.v2.dsl.parallel import handle_parallel_model

        new_kwargs["tools"] = handle_parallel_model(cast(Any, response_model))
        new_kwargs["tool_choice"] = "auto"

        # Wrap in ParallelModel for proper parsing
        parallel_model = ParallelModel(typehint=response_model)
        return cast(type[BaseModel], parallel_model), new_kwargs

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: ChatCompletion,
        exception: Exception,
    ) -> dict[str, Any]:
        """Handle reask for parallel tools mode."""
        return reask_tools(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,  # noqa: ARG002
        is_async: bool = False,  # noqa: ARG002
    ) -> Any:
        """Parse parallel tool response."""
        # Check for incomplete output
        choices = getattr(response, "choices", None)
        if not choices:
            raise ResponseParsingError(
                "No choices in OpenAI response",
                mode=str(self.mode.value),
                raw_response=response,
            )
        if choices[0].finish_reason == "length":
            raise IncompleteOutputException(last_completion=response)

        # Extract model types from response_model
        the_types = get_types_array(response_model)  # type: ignore[arg-type]
        type_registry = {t.__name__: t for t in the_types}

        results = []
        tool_calls = choices[0].message.tool_calls
        if not tool_calls:
            raise ResponseParsingError(
                "No tool calls in response",
                mode="PARALLEL_TOOLS",
                raw_response=response,
            )
        for tool_call in tool_calls:
            name = tool_call.function.name
            args = tool_call.function.arguments
            if name in type_registry:
                model = type_registry[name].model_validate_json(
                    args,
                    context=validation_context,
                    strict=strict,
                )
                results.append(model)

        return iter(results)


@register_mode_handler(Provider.OPENAI, Mode.RESPONSES_TOOLS)
class OpenAIResponsesToolsHandler(OpenAIHandlerBase):
    """Handler for OpenAI Responses API with tools."""

    mode = Mode.RESPONSES_TOOLS

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        """Prepare request for Responses API with tools."""
        self._register_streaming_from_kwargs(response_model, kwargs)

        new_kwargs = kwargs.copy()
        new_kwargs["messages"] = list(kwargs.get("messages", []))

        # Handle max_tokens to max_output_tokens conversion
        if new_kwargs.get("max_tokens") is not None:
            new_kwargs["max_output_tokens"] = new_kwargs.pop("max_tokens")

        if response_model is None:
            return None, new_kwargs

        from typing import cast
        from instructor.v2.core.response_model import prepare_response_model

        prepared_model = cast(type[BaseModel], prepare_response_model(response_model))

        from openai import pydantic_function_tool

        schema = pydantic_function_tool(prepared_model)
        del schema["function"]["strict"]

        schema_function = schema["function"]
        tool_definition: dict[str, Any] = {
            "type": "function",
            "name": schema_function["name"],
            "parameters": schema_function.get("parameters", {}),
        }

        if "description" in schema_function:
            tool_definition["description"] = schema_function["description"]
        else:
            tool_definition["description"] = (
                f"Correctly extracted `{prepared_model.__name__}` with all "
                f"the required parameters with correct types"
            )

        new_kwargs["tools"] = [tool_definition]
        new_kwargs["tool_choice"] = {
            "type": "function",
            "name": schema["function"]["name"],
        }
        _ensure_text_format_config(
            new_kwargs,
            prepared_model,
            tool_definition["parameters"],
        )

        return prepared_model, new_kwargs

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        """Handle reask for Responses API."""
        return reask_responses_tools(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,
        is_async: bool = False,  # noqa: ARG002
    ) -> Any:
        """Parse Responses API response."""
        # Check for streaming
        if (
            isinstance(response_model, type)
            and (stream or self._consume_streaming_flag(response_model))
            and issubclass(response_model, (IterableBase, PartialBase))
        ):
            return self._parse_streaming_response(
                response_model,
                response,
                validation_context,
                strict,
            )

        # Handle Responses API format - output is a list of items
        if hasattr(response, "output"):
            for item in response.output:
                item_type = getattr(item, "type", None)
                if item_type in {"function_call", "tool_call"}:
                    args = getattr(item, "arguments", None)
                    if not args or (
                        isinstance(args, str) and args.strip() in {"", "{}"}
                    ):
                        logger.warning(
                            "RESPONSES_TOOLS: tool '%s' returned empty arguments. "
                            "This can indicate a text.format/tool_choice conflict "
                            "or insufficient reasoning budget.",
                            getattr(item, "name", "unknown"),
                        )
                    if args:
                        parsed = response_model.model_validate_json(
                            args,
                            context=validation_context,
                            strict=strict,
                        )
                        return self._finalize_parsed_result(
                            response_model, response, parsed
                        )

        # Fallback to standard tool call parsing
        json_str = self._extract_tool_call_json(response)
        parsed = response_model.model_validate_json(
            json_str,
            context=validation_context,
            strict=strict,
        )
        return self._finalize_parsed_result(response_model, response, parsed)


__all__ = [
    "handle_openrouter_structured_outputs",
    "reask_default",
    "reask_md_json",
    "reask_responses_tools",
    "reask_tools",
    "OpenAIToolsHandler",
    "OpenAIJSONSchemaHandler",
    "OpenAIMDJSONHandler",
    "OpenAIParallelToolsHandler",
    "OpenAIResponsesToolsHandler",
]
