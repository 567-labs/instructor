"""Anthropic v2 mode handlers with DSL-aware parsing."""

from __future__ import annotations

import inspect
import json
import warnings
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Generator,
    Iterable as TypingIterable,
)
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, TypedDict, cast, get_origin

from pydantic import BaseModel, Field, TypeAdapter
from typing import Annotated

if TYPE_CHECKING:  # pragma: no cover - typing only
    from anthropic.types import Message

from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.errors import ConfigurationError, IncompleteOutputException
from instructor.v2.dsl.iterable import IterableBase
from instructor.v2.dsl.parallel import (
    ParallelBase,
    get_types_array,
)
from instructor.v2.providers.anthropic.parallel import (
    handle_parallel_model as handle_anthropic_parallel_model,
)
from instructor.v2.dsl.simple_type import AdapterBase
from instructor.v2.core.multimodal import Audio, Image, PDF
from instructor.v2.core.multimodal import convert_messages as convert_messages_v1
from instructor.v2.core.json import extract_json_from_codeblock
from instructor.v2.providers.anthropic.schema import generate_anthropic_schema
from instructor.v2.core.decorators import register_mode_handler
from instructor.v2.core.handler import ModeHandler
from instructor.v2.core.streaming import StreamingModelState


class SystemMessage(TypedDict, total=False):
    type: str
    text: str
    cache_control: dict[str, str]


def combine_system_messages(
    existing_system: str | list[SystemMessage] | None,
    new_system: str | list[SystemMessage],
) -> str | list[SystemMessage]:
    """Combine existing and new system messages."""
    if not isinstance(new_system, (str, list)) or (
        existing_system is not None and not isinstance(existing_system, (str, list))
    ):
        raise ValueError(
            "System messages must be strings or lists, got "
            f"{type(existing_system)} and {type(new_system)}"
        )

    if existing_system is None:
        return new_system

    if isinstance(existing_system, str):
        if isinstance(new_system, str):
            return f"{existing_system}\n\n{new_system}"
        result = [SystemMessage(type="text", text=existing_system)]
        result.extend(new_system)
        return result

    result = list(existing_system)
    if isinstance(new_system, list):
        result.extend(new_system)
    else:
        result.append(SystemMessage(type="text", text=new_system))
    return result


def extract_system_messages(messages: list[dict[str, Any]]) -> list[SystemMessage]:
    """Extract system messages from a list of messages."""
    if not messages:
        return []

    system_count = sum(1 for m in messages if m.get("role") == "system")
    if system_count == 0:
        return []

    def convert_message(content: Any) -> SystemMessage:
        if isinstance(content, str):
            return SystemMessage(type="text", text=content)
        if isinstance(content, dict):
            return SystemMessage(**content)
        raise ValueError(f"Unsupported content type: {type(content)}")

    result: list[SystemMessage] = []
    for message in messages:
        if message.get("role") == "system":
            content = message.get("content")
            if not content:
                continue
            if isinstance(content, list):
                for item in content:
                    if item:
                        result.append(convert_message(item))
            else:
                result.append(convert_message(content))

    return result


def serialize_message_content(content: Any) -> Any:
    """Serialize message content, converting Pydantic models to dicts."""

    if isinstance(content, Image):
        return content.to_anthropic()
    if isinstance(content, PDF):
        return content.to_anthropic()
    if isinstance(content, Audio):
        source = str(content.source)
        if source.startswith(("http://", "https://")):
            return {
                "type": "audio",
                "source": {"type": "url", "url": source},
            }
        return {
            "type": "audio",
            "source": {
                "type": "base64",
                "media_type": content.media_type,
                "data": content.data or source,
            },
        }
    if isinstance(content, str):
        return {"type": "text", "text": content}
    if isinstance(content, list):
        return [serialize_message_content(item) for item in content]
    if isinstance(content, dict):
        if "type" in content and isinstance(content.get("type"), str):
            return content
        return {k: serialize_message_content(v) for k, v in content.items()}
    if hasattr(content, "model_dump"):
        return content.model_dump()
    return content


def _anthropic_supports_output_format() -> bool:
    """Detect if the installed anthropic SDK supports output_format parameter."""

    try:
        from anthropic.resources.messages import Messages
    except (ImportError, AttributeError):
        return False

    try:
        signature = inspect.signature(Messages.create)
    except (ValueError, TypeError):
        return False

    return "output_format" in signature.parameters


def process_messages_for_anthropic(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Process messages to serialize any Pydantic models in content."""

    processed: list[dict[str, Any]] = []
    for message in messages:
        msg_copy = message.copy()
        if "content" in msg_copy:
            content = msg_copy["content"]
            if isinstance(content, list):
                msg_copy["content"] = serialize_message_content(content)
            elif isinstance(content, (Image, Audio, PDF)) or hasattr(
                content, "model_dump"
            ):
                msg_copy["content"] = serialize_message_content(content)
        processed.append(msg_copy)
    return processed


class AnthropicHandlerBase(StreamingModelState, ModeHandler):
    """Common utilities for Anthropic handlers."""

    mode: Mode

    def extract_streaming_json(
        self, completion: TypingIterable[Any]
    ) -> Generator[str, None, None]:
        """Extract JSON chunks from Anthropic streaming responses."""
        for chunk in completion:
            try:
                if self.mode in {Mode.TOOLS, Mode.PARALLEL_TOOLS}:
                    yield chunk.delta.partial_json
                elif self.mode in {Mode.JSON, Mode.JSON_SCHEMA}:
                    if json_chunk := chunk.delta.text:
                        yield json_chunk
            except AttributeError:
                continue

    async def extract_streaming_json_async(
        self, completion: AsyncGenerator[Any, None]
    ) -> AsyncGenerator[str, None]:
        """Extract JSON chunks from Anthropic async streams."""
        async for chunk in completion:
            try:
                if self.mode in {Mode.TOOLS, Mode.PARALLEL_TOOLS}:
                    yield chunk.delta.partial_json
                elif self.mode in {Mode.JSON, Mode.JSON_SCHEMA}:
                    if json_chunk := chunk.delta.text:
                        yield json_chunk
            except AttributeError:
                continue

    def convert_messages(
        self, messages: list[dict[str, Any]], autodetect_images: bool = False
    ) -> list[dict[str, Any]]:
        """Convert messages for Anthropic-compatible multimodal payloads."""
        if self.mode in {Mode.TOOLS, Mode.PARALLEL_TOOLS}:
            target_mode = Mode.ANTHROPIC_TOOLS
        else:
            target_mode = Mode.ANTHROPIC_JSON
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
        parse_kwargs: dict[str, Any] = {}
        if validation_context is not None:
            parse_kwargs["context"] = validation_context
        if strict is not None:
            parse_kwargs["strict"] = strict

        streaming_model = cast(Any, response_model)
        if inspect.isasyncgen(response) or isinstance(response, AsyncIterator):
            return streaming_model.from_streaming_response_async(
                response,
                stream_extractor=self.extract_streaming_json_async,
                **parse_kwargs,
            )

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
        if isinstance(parsed, IterableBase):
            return [task for task in parsed.tasks]
        if isinstance(response_model, ParallelBase):
            return parsed
        if isinstance(parsed, AdapterBase):
            return parsed.content
        if isinstance(parsed, BaseModel):
            parsed._raw_response = response  # type: ignore[attr-defined]
        return parsed

    def _parse_with_callback(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None,
        strict: bool | None,
        parser: Callable[
            [Any, type[BaseModel], dict[str, Any] | None, bool | None],
            Any,
        ],
        stream: bool | None = None,
    ) -> Any:
        if self._should_parse_streaming(response_model, stream):
            return self._parse_streaming_response(
                response_model,
                response,
                validation_context,
                strict,
            )

        parsed = parser(response, response_model, validation_context, strict)
        return self._finalize_parsed_result(response_model, response, parsed)


@register_mode_handler(Provider.ANTHROPIC, Mode.TOOLS)
class AnthropicToolsHandler(AnthropicHandlerBase):
    """Handler for Anthropic TOOLS mode."""

    mode = Mode.TOOLS

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        new_kwargs = kwargs.copy()
        system_messages = extract_system_messages(new_kwargs.get("messages", []))
        if system_messages:
            new_kwargs["system"] = combine_system_messages(
                new_kwargs.get("system"), system_messages
            )
        new_kwargs["messages"] = [
            m for m in new_kwargs.get("messages", []) if m["role"] != "system"
        ]
        new_kwargs["messages"] = process_messages_for_anthropic(new_kwargs["messages"])

        if response_model is None:
            return None, new_kwargs

        # Detect if this is a parallel tools request (Iterable[Union[...]])
        # When streaming, treat Iterable[T] as streaming instead of parallel tools.
        origin = get_origin(response_model)
        is_parallel = origin is TypingIterable and not new_kwargs.get("stream")

        # Prepare response model: wrap simple types in ModelAdapter
        # Skip for parallel tools as they're handled separately
        if not is_parallel:
            from instructor.v2.core.response_model import prepare_response_model

            # Use prepare_response_model to handle simple types, TypedDict, etc.
            response_model = cast(
                type[BaseModel], prepare_response_model(response_model)
            )

        self._register_streaming_from_kwargs(response_model, new_kwargs)

        if is_parallel:
            tool_schemas = handle_anthropic_parallel_model(response_model)
            new_kwargs["tools"] = tool_schemas
        else:
            tool_descriptions = generate_anthropic_schema(response_model)
            new_kwargs["tools"] = [tool_descriptions]

        if "tool_choice" not in new_kwargs:
            thinking_enabled = (
                "thinking" in new_kwargs
                and isinstance(new_kwargs.get("thinking"), dict)
                and new_kwargs.get("thinking", {}).get("type") == "enabled"
            )
            if thinking_enabled or is_parallel:
                new_kwargs["tool_choice"] = {"type": "auto"}
                if thinking_enabled:
                    new_kwargs["system"] = combine_system_messages(
                        new_kwargs.get("system"),
                        [
                            {
                                "type": "text",
                                "text": "Return only the tool call and no additional text.",
                            }
                        ],
                    )
            else:
                new_kwargs["tool_choice"] = {
                    "type": "tool",
                    "name": getattr(response_model, "__name__", "response"),
                    # Without this, Anthropic may still emit multiple tool_use
                    # blocks for the same forced tool, which the single-model
                    # parser then rejects even though the model's response was
                    # otherwise valid.
                    "disable_parallel_tool_use": True,
                }

        return response_model, new_kwargs

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Message,
        exception: Exception,
    ) -> dict[str, Any]:
        kwargs = kwargs.copy()
        if response is None or not hasattr(response, "content"):
            kwargs["messages"].append(
                {
                    "role": "user",
                    "content": (
                        "Validation Error found:\n"
                        f"{exception}\nRecall the function correctly, fix the errors"
                    ),
                }
            )
            return kwargs

        assistant_content = []
        # Anthropic requires a tool_result for *every* tool_use in the prior turn.
        # Collect all ids — overwriting a single id drops results for PARALLEL_TOOLS
        # and makes the reask request fail with 400 (issue #2485).
        tool_use_ids: list[str] = []
        for content in response.content:
            try:
                dumped_content = content.model_dump(exclude_none=True)  # type: ignore[attr-defined]
            except TypeError:
                dumped_content = content.model_dump()  # type: ignore[attr-defined]
            assistant_content.append(dumped_content)
            if content.type == "tool_use":
                tool_use_ids.append(content.id)

        reask_msgs = [{"role": "assistant", "content": assistant_content}]
        if tool_use_ids:
            error_content = (
                "Validation Error found:\n"
                f"{exception}\nRecall the function correctly, fix the errors"
            )
            reask_msgs.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": error_content,
                            "is_error": True,
                        }
                        for tool_use_id in tool_use_ids
                    ],
                }
            )
        else:
            reask_msgs.append(
                {
                    "role": "user",
                    "content": (
                        "Validation Error due to no tool invocation:\n"
                        f"{exception}\nRecall the function correctly, fix the errors"
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
        return self._parse_with_callback(
            response,
            response_model,
            validation_context,
            strict,
            self._parse_tool_response,
            stream,
        )

    def _parse_tool_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None,
        strict: bool | None,
    ) -> Any:
        from anthropic.types import Message

        if isinstance(response, Message) and response.stop_reason == "max_tokens":
            raise IncompleteOutputException(last_completion=response)

        origin = get_origin(response_model)
        if origin is TypingIterable:
            the_types = get_types_array(response_model)  # type: ignore[arg-type]
            type_registry = {t.__name__: t for t in the_types}

            def parallel_generator() -> Generator[BaseModel, None, None]:
                for content in response.content:
                    if getattr(content, "type", None) == "tool_use":
                        tool_name = content.name
                        if tool_name in type_registry:
                            model_class = type_registry[tool_name]
                            json_str = json.dumps(content.input)
                            yield model_class.model_validate_json(
                                json_str,
                                context=validation_context,
                                strict=strict,
                            )

            return parallel_generator()

        tool_calls = [
            json.dumps(c.input)
            for c in getattr(response, "content", [])
            if getattr(c, "type", None) == "tool_use"
        ]
        tool_calls_validator = TypeAdapter(
            Annotated[list[Any], Field(min_length=1, max_length=1)]
        )
        tool_call = tool_calls_validator.validate_python(tool_calls)[0]
        return response_model.model_validate_json(
            tool_call,
            context=validation_context,
            strict=strict,
        )


@register_mode_handler(Provider.ANTHROPIC, Mode.PARALLEL_TOOLS)
class AnthropicParallelToolsHandler(AnthropicHandlerBase):
    """Handler for Anthropic parallel tool calling."""

    mode = Mode.PARALLEL_TOOLS

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        self._register_streaming_from_kwargs(response_model, kwargs)

        new_kwargs = kwargs.copy()
        if new_kwargs.get("stream"):
            raise ConfigurationError(
                "stream=True is not supported when using PARALLEL_TOOLS mode"
            )

        system_messages = extract_system_messages(new_kwargs.get("messages", []))
        if system_messages:
            new_kwargs["system"] = combine_system_messages(
                new_kwargs.get("system"), system_messages
            )
        new_kwargs["messages"] = [
            m for m in new_kwargs.get("messages", []) if m["role"] != "system"
        ]

        if response_model is None:
            return None, new_kwargs

        new_kwargs["tools"] = handle_anthropic_parallel_model(response_model)
        new_kwargs["tool_choice"] = {"type": "auto"}

        return response_model, new_kwargs

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Message,
        exception: Exception,
    ) -> dict[str, Any]:
        return AnthropicToolsHandler().handle_reask(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool | None = None,
        is_async: bool = False,  # noqa: ARG002
    ) -> Any:
        return self._parse_with_callback(
            response,
            response_model,
            validation_context,
            strict,
            self._parse_parallel_response,
            stream,
        )

    def _parse_parallel_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None,
        strict: bool | None,
    ) -> Generator[BaseModel, None, None]:
        """Parse parallel tool response directly without using AnthropicParallelBase."""
        if not response or not hasattr(response, "content"):
            return

        # Extract model types from response_model (Iterable[Union[Model1, Model2, ...]])
        the_types = get_types_array(response_model)  # type: ignore[arg-type]
        type_registry = {
            model.__name__ if hasattr(model, "__name__") else str(model): model
            for model in the_types
        }

        # Parse tool_use blocks from response
        for content in response.content:
            if getattr(content, "type", None) == "tool_use":
                name = content.name
                arguments = content.input
                if name in type_registry:
                    model_class = type_registry[name]
                    json_str = json.dumps(arguments)
                    yield model_class.model_validate_json(
                        json_str,
                        context=validation_context,
                        strict=strict,
                    )


@register_mode_handler(Provider.ANTHROPIC, Mode.JSON)
class AnthropicJSONHandler(AnthropicHandlerBase):
    """Handler for Anthropic JSON mode."""

    mode = Mode.JSON

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        self._register_streaming_from_kwargs(response_model, kwargs)

        new_kwargs = kwargs.copy()
        system_messages = extract_system_messages(new_kwargs.get("messages", []))
        if system_messages:
            new_kwargs["system"] = combine_system_messages(
                new_kwargs.get("system"), system_messages
            )
        new_kwargs["messages"] = [
            m for m in new_kwargs.get("messages", []) if m["role"] != "system"
        ]
        new_kwargs["messages"] = process_messages_for_anthropic(new_kwargs["messages"])

        if response_model is None:
            return None, new_kwargs

        json_schema_message = dedent(
            f"""
            As a genius expert, your task is to understand the content and provide
            the parsed objects in json that match the following json_schema:\n
            {json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}

            Make sure to return an instance of the JSON, not the schema itself
            """
        )
        new_kwargs["system"] = combine_system_messages(
            new_kwargs.get("system"),
            [{"type": "text", "text": json_schema_message}],
        )
        return response_model, new_kwargs

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Message,
        exception: Exception,
    ) -> dict[str, Any]:
        kwargs = kwargs.copy()
        text_blocks = [c for c in response.content if c.type == "text"]
        if not text_blocks:
            text_content = "No text content found in response"
        else:
            text_content = text_blocks[-1].text
        reask_msg = {
            "role": "user",
            "content": (
                "Validation Errors found:\n"
                f"{exception}\nRecall the function correctly, fix the errors found in the following attempt:\n{text_content}"
            ),
        }
        kwargs["messages"].append(reask_msg)
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
        return self._parse_with_callback(
            response,
            response_model,
            validation_context,
            strict,
            self._parse_json_response,
            stream,
        )

    def _parse_json_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None,
        strict: bool | None,
    ) -> BaseModel:
        from anthropic.types import Message
        from instructor.v2.core.errors import ResponseParsingError

        if hasattr(response, "choices"):
            completion = response.choices[0]
            if completion.finish_reason == "length":
                raise IncompleteOutputException(last_completion=completion)
            text = completion.message.content
        else:
            if not isinstance(response, Message):
                raise ResponseParsingError(
                    "Response must be an Anthropic Message",
                    mode="JSON",
                    raw_response=response,
                )
            if response.stop_reason == "max_tokens":
                raise IncompleteOutputException(last_completion=response)
            text_blocks = [c for c in response.content if c.type == "text"]
            if not text_blocks:
                raise ResponseParsingError(
                    "No text content in response",
                    mode="MD_JSON",
                    raw_response=response,
                )
            last_block = text_blocks[-1]
            text = last_block.text

        extra_text = extract_json_from_codeblock(text)
        if strict:
            return response_model.model_validate_json(
                extra_text,
                context=validation_context,
                strict=strict,
            )

        parsed = json.loads(extra_text, strict=False)
        return response_model.model_validate(
            parsed,
            context=validation_context,
            strict=strict,
        )


@register_mode_handler(Provider.ANTHROPIC, Mode.JSON_SCHEMA)
class AnthropicStructuredOutputsHandler(AnthropicHandlerBase):
    """Handler for Anthropic structured outputs mode.

    Uses Claude's native structured output enforcement via the output_format parameter.
    Requires Anthropic SDK >=0.71.0 and the structured-outputs-2025-11-13 beta.
    """

    mode = Mode.JSON_SCHEMA

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        self._register_streaming_from_kwargs(response_model, kwargs)

        if response_model is None:
            from instructor.v2.core.errors import ConfigurationError

            raise ConfigurationError(
                "Mode.JSON_SCHEMA (Anthropic structured outputs) requires a `response_model`."
            )

        if not _anthropic_supports_output_format():
            warnings.warn(
                "Anthropic client does not support `output_format`; falling back to JSON mode instructions.",
                UserWarning,
                stacklevel=2,
            )
            json_handler = AnthropicJSONHandler()
            return json_handler.prepare_request(response_model, kwargs)

        new_kwargs = kwargs.copy()
        system_messages = extract_system_messages(new_kwargs.get("messages", []))
        if system_messages:
            new_kwargs["system"] = combine_system_messages(
                new_kwargs.get("system"), system_messages
            )
        new_kwargs["messages"] = [
            m for m in new_kwargs.get("messages", []) if m["role"] != "system"
        ]
        new_kwargs["messages"] = process_messages_for_anthropic(new_kwargs["messages"])

        import anthropic

        transform_schema = getattr(anthropic, "transform_schema", None)
        if transform_schema is None:
            warnings.warn(
                "Anthropic structured outputs works best with anthropic>=0.71.0. "
                "Falling back to response_model.model_json_schema().",
                UserWarning,
                stacklevel=2,
            )

            def transform_schema(model: type[BaseModel]) -> dict[str, Any]:
                return model.model_json_schema()

        new_kwargs["output_format"] = {
            "type": "json_schema",
            "schema": transform_schema(response_model),
        }

        required_beta = "structured-outputs-2025-11-13"
        betas = new_kwargs.get("betas")
        if betas is None:
            new_kwargs["betas"] = [required_beta]
        else:
            if isinstance(betas, str):
                betas = [betas]
            elif not isinstance(betas, list):
                betas = list(betas)
            if required_beta not in betas:
                betas.append(required_beta)
            new_kwargs["betas"] = betas

        # Ensure legacy tool kwargs are cleared
        new_kwargs.pop("tools", None)
        new_kwargs.pop("tool_choice", None)

        return response_model, new_kwargs

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Message,
        exception: Exception,
    ) -> dict[str, Any]:
        # Use same reask logic as JSON mode
        kwargs = kwargs.copy()
        text_blocks = [c for c in response.content if c.type == "text"]
        if not text_blocks:
            text_content = "No text content found in response"
        else:
            text_content = text_blocks[-1].text
        reask_msg = {
            "role": "user",
            "content": (
                "Validation Errors found:\n"
                f"{exception}\nRecall the function correctly, fix the errors found in the following attempt:\n{text_content}"
            ),
        }
        kwargs["messages"].append(reask_msg)
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
        return self._parse_with_callback(
            response,
            response_model,
            validation_context,
            strict,
            self._parse_structured_output_response,
            stream,
        )

    def _parse_structured_output_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None,
        strict: bool | None,
    ) -> BaseModel:
        from anthropic.types import Message
        from instructor.v2.core.errors import ResponseParsingError

        if not isinstance(response, Message):
            raise ResponseParsingError(
                "Response must be an Anthropic Message",
                mode="JSON_SCHEMA",
                raw_response=response,
            )
        if response.stop_reason == "max_tokens":
            raise IncompleteOutputException(last_completion=response)

        # Structured outputs returns content directly in text blocks
        text_blocks = [c for c in response.content if c.type == "text"]
        if not text_blocks:
            raise ResponseParsingError(
                "No text content found in structured output response",
                mode="JSON_SCHEMA",
                raw_response=response,
            )

        # Get the text content (should be valid JSON per schema)
        text_content = text_blocks[-1].text

        # Parse and validate
        if strict:
            return response_model.model_validate_json(
                text_content,
                context=validation_context,
                strict=strict,
            )
        return response_model.model_validate_json(
            text_content,
            context=validation_context,
        )


__all__ = [
    "AnthropicToolsHandler",
    "AnthropicParallelToolsHandler",
    "AnthropicJSONHandler",
    "AnthropicStructuredOutputsHandler",
]
