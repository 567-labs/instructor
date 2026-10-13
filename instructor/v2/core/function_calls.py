from __future__ import annotations

import inspect
import json
import logging
import warnings
import weakref
from functools import wraps
from typing import Any, TypeVar, cast
from typing_extensions import Self
from openai.types.chat import ChatCompletion
from pydantic import (
    BaseModel,
    ConfigDict,
    TypeAdapter,
    create_model,
)

from instructor.v2.core.errors import (
    IncompleteOutputException,
)
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import (
    Provider,
    normalize_mode_for_provider,
    provider_from_mode,
)
from instructor.v2.core.utils import classproperty


T = TypeVar("T")
Model = TypeVar("Model", bound=BaseModel)

logger = logging.getLogger("instructor")

# No schema cache


# Utility functions for common JSON parsing operations
def _handle_incomplete_output(completion: Any) -> None:
    """Check if a completion was incomplete and raise appropriate exception."""
    if (
        hasattr(completion, "choices")
        and completion.choices[0].finish_reason == "length"
    ):
        raise IncompleteOutputException(last_completion=completion)

    # Handle Anthropic format
    if hasattr(completion, "stop_reason") and completion.stop_reason == "max_tokens":
        raise IncompleteOutputException(last_completion=completion)


def _extract_text_content(completion: Any) -> str:
    """Extract text content from various completion formats."""
    # OpenAI format
    if hasattr(completion, "choices"):
        return completion.choices[0].message.content or ""

    # Simple text format
    if hasattr(completion, "text"):
        return completion.text

    # Anthropic format
    if hasattr(completion, "content"):
        text_blocks = [c for c in completion.content if c.type == "text"]
        if text_blocks:
            return text_blocks[0].text

    # Bedrock format
    if isinstance(completion, dict) and "output" in completion:
        try:
            output = completion.get("output")
            if not isinstance(output, dict):
                return ""
            message = output.get("message")
            if not isinstance(message, dict):
                return ""
            content = message.get("content")
            if not isinstance(content, list):
                return ""
            return content[0].get("text")
        except (AttributeError, IndexError):
            pass

    return ""


def _validate_model_from_json(
    cls: type[Any],
    json_str: str,
    validation_context: dict[str, Any] | None = None,
    strict: bool | None = None,
) -> Any:
    """Validate model from JSON string with appropriate error handling."""
    try:
        if hasattr(cls, "model_validate_json"):
            if strict:
                return cls.model_validate_json(
                    json_str, context=validation_context, strict=True
                )
            # Allow control characters
            parsed = json.loads(json_str, strict=False)
            return cls.model_validate(parsed, context=validation_context, strict=False)

        adapter = TypeAdapter(cls)
        if strict:
            return adapter.validate_json(
                json_str, context=validation_context, strict=True
            )
        parsed = json.loads(json_str, strict=False)
        return adapter.validate_python(parsed, context=validation_context, strict=False)
    except json.JSONDecodeError as e:
        logger.debug(f"JSON decode error: {e}")
        raise
    except Exception as e:
        logger.debug(f"Model validation error: {e}")
        raise


class ResponseSchema(BaseModel):
    # Ignore classproperty, since Pydantic doesn't understand it like it would a normal property.
    model_config = ConfigDict(ignored_types=(classproperty,))

    @classproperty
    def openai_schema(cls) -> dict[str, Any]:
        """
        Return the schema in the format of OpenAI's schema as jsonschema

        Note:
            Its important to add a docstring to describe how to best use this class, it will be included in the description attribute and be part of the prompt.

        Returns:
            model_json_schema (dict): A dictionary in the format of OpenAI's schema as jsonschema
        """
        from instructor.v2.providers.openai.schema import generate_openai_schema

        return generate_openai_schema(cls)

    @classproperty
    def anthropic_schema(cls) -> dict[str, Any]:
        from instructor.v2.providers.anthropic.schema import generate_anthropic_schema

        return generate_anthropic_schema(cls)

    @classproperty
    def gemini_schema(cls) -> Any:
        from instructor.v2.providers.gemini.schema import generate_gemini_schema

        return generate_gemini_schema(cls)

    @classmethod
    def from_response(
        cls: type[Self],
        completion: ChatCompletion,
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        mode: Mode = Mode.TOOLS,
        provider: Provider = Provider.OPENAI,
    ) -> Self:
        """Execute the function from the response of an openai chat completion

        Parameters:
            completion (openai.ChatCompletion): The response from an openai chat completion
            strict (bool): Whether to use strict json parsing
            mode (Mode): The completion mode
            provider (Provider): The provider for handler lookup

        Returns:
            cls (ResponseSchema): An instance of the class
        """

        from instructor.v2.validation.async_validators import reject_async_validators

        reject_async_validators(cls)
        import importlib

        from instructor.v2.core.registry import mode_registry

        importlib.import_module("instructor.v2")

        provider = provider_from_mode(mode, provider)
        mode = normalize_mode_for_provider(mode, provider)
        handlers = mode_registry.get_handlers(provider, mode)
        return handlers.response_parser(
            response=completion,
            response_model=cls,
            validation_context=validation_context,
            strict=strict,
            stream=False,
            is_async=False,
        )

    @classmethod
    def _parse_with_registry(
        cls: type[Self],
        completion: Any,
        *,
        mode: Mode,
        provider: Provider,
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        warning: str | None = None,
    ) -> Self:
        if warning:
            warnings.warn(warning, DeprecationWarning, stacklevel=2)
        return cls.from_response(
            completion,
            validation_context=validation_context,
            strict=strict,
            mode=mode,
            provider=provider,
        )

    @classmethod
    def parse_genai_structured_outputs(
        cls: type[ResponseSchema],
        completion: ChatCompletion,
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
    ) -> BaseModel:
        """Legacy GenAI structured parser (deprecated)."""
        return cls._parse_with_registry(
            completion,
            mode=Mode.JSON,
            provider=Provider.GENAI,
            validation_context=validation_context,
            strict=strict,
            warning=(
                "ResponseSchema.parse_genai_structured_outputs is deprecated. "
                "Use process_response(..., provider=Provider.GENAI, mode=Mode.JSON)."
            ),
        )

    @classmethod
    def parse_genai_tools(
        cls: type[ResponseSchema],
        completion: ChatCompletion,
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
    ) -> BaseModel:
        """Legacy GenAI tools parser (deprecated)."""
        return cls._parse_with_registry(
            completion,
            mode=Mode.TOOLS,
            provider=Provider.GENAI,
            validation_context=validation_context,
            strict=strict,
            warning=(
                "ResponseSchema.parse_genai_tools is deprecated. "
                "Use process_response(..., provider=Provider.GENAI, mode=Mode.TOOLS)."
            ),
        )

    @classmethod
    def parse_cohere_json_schema(
        cls: type[ResponseSchema],
        completion: ChatCompletion,
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
    ):
        """Legacy Cohere JSON schema parser (deprecated)."""
        return cls._parse_with_registry(
            completion,
            mode=Mode.JSON_SCHEMA,
            provider=Provider.COHERE,
            validation_context=validation_context,
            strict=strict,
            warning=(
                "ResponseSchema.parse_cohere_json_schema is deprecated. "
                "Use process_response(..., provider=Provider.COHERE, mode=Mode.JSON_SCHEMA)."
            ),
        )

    @classmethod
    def parse_anthropic_tools(
        cls: type[ResponseSchema],
        completion: ChatCompletion,
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
    ) -> BaseModel:
        """Legacy Anthropic tools parser (deprecated)."""
        return cls._parse_with_registry(
            completion,
            mode=Mode.ANTHROPIC_TOOLS,
            provider=Provider.ANTHROPIC,
            validation_context=validation_context,
            strict=strict,
            warning=(
                "ResponseSchema.parse_anthropic_tools is deprecated. "
                "Use process_response(..., provider=Provider.ANTHROPIC, mode=Mode.TOOLS) "
                "or ResponseSchema.from_response with core modes."
            ),
        )

    @classmethod
    def parse_anthropic_json(
        cls: type[ResponseSchema],
        completion: ChatCompletion,
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
    ) -> BaseModel:
        """Legacy Anthropic JSON parser (deprecated)."""
        return cls._parse_with_registry(
            completion,
            mode=Mode.ANTHROPIC_JSON,
            provider=Provider.ANTHROPIC,
            validation_context=validation_context,
            strict=strict,
            warning=(
                "ResponseSchema.parse_anthropic_json is deprecated. "
                "Use process_response(..., provider=Provider.ANTHROPIC, mode=Mode.JSON) "
                "or ResponseSchema.from_response with core modes."
            ),
        )

    @classmethod
    def parse_bedrock_json(
        cls: type[ResponseSchema],
        completion: Any,
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
    ) -> BaseModel:
        """Legacy Bedrock JSON parser (deprecated)."""
        return cls._parse_with_registry(
            completion,
            mode=Mode.MD_JSON,
            provider=Provider.BEDROCK,
            validation_context=validation_context,
            strict=strict,
            warning=(
                "ResponseSchema.parse_bedrock_json is deprecated. "
                "Use process_response(..., provider=Provider.BEDROCK, mode=Mode.MD_JSON)."
            ),
        )

    @classmethod
    def parse_bedrock_tools(
        cls: type[ResponseSchema],
        completion: Any,
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
    ) -> BaseModel:
        """Legacy Bedrock tools parser (deprecated)."""
        return cls._parse_with_registry(
            completion,
            mode=Mode.TOOLS,
            provider=Provider.BEDROCK,
            validation_context=validation_context,
            strict=strict,
            warning=(
                "ResponseSchema.parse_bedrock_tools is deprecated. "
                "Use process_response(..., provider=Provider.BEDROCK, mode=Mode.TOOLS)."
            ),
        )

    @classmethod
    def parse_gemini_json(
        cls: type[ResponseSchema],
        completion: Any,
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
    ) -> BaseModel:
        """Legacy Gemini JSON parser (deprecated)."""
        return cls._parse_with_registry(
            completion,
            mode=Mode.MD_JSON,
            provider=Provider.GEMINI,
            validation_context=validation_context,
            strict=strict,
            warning=(
                "ResponseSchema.parse_gemini_json is deprecated. "
                "Use process_response(..., provider=Provider.GEMINI, mode=Mode.MD_JSON)."
            ),
        )

    @classmethod
    def parse_gemini_tools(
        cls: type[ResponseSchema],
        completion: Any,
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
    ) -> BaseModel:
        """Legacy Gemini tools parser (deprecated)."""
        return cls._parse_with_registry(
            completion,
            mode=Mode.TOOLS,
            provider=Provider.GEMINI,
            validation_context=validation_context,
            strict=strict,
            warning=(
                "ResponseSchema.parse_gemini_tools is deprecated. "
                "Use process_response(..., provider=Provider.GEMINI, mode=Mode.TOOLS)."
            ),
        )

    @classmethod
    def parse_vertexai_tools(
        cls: type[ResponseSchema],
        completion: ChatCompletion,
        validation_context: dict[str, Any] | None = None,
    ) -> BaseModel:
        """Legacy VertexAI tools parser (deprecated)."""
        return cls._parse_with_registry(
            completion,
            mode=Mode.TOOLS,
            provider=Provider.VERTEXAI,
            validation_context=validation_context,
            strict=False,
            warning=(
                "ResponseSchema.parse_vertexai_tools is deprecated. "
                "Use process_response(..., provider=Provider.VERTEXAI, mode=Mode.TOOLS)."
            ),
        )

    @classmethod
    def parse_vertexai_json(
        cls: type[ResponseSchema],
        completion: ChatCompletion,
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
    ) -> BaseModel:
        """Legacy VertexAI JSON parser (deprecated)."""
        return cls._parse_with_registry(
            completion,
            mode=Mode.MD_JSON,
            provider=Provider.VERTEXAI,
            validation_context=validation_context,
            strict=strict,
            warning=(
                "ResponseSchema.parse_vertexai_json is deprecated. "
                "Use process_response(..., provider=Provider.VERTEXAI, mode=Mode.MD_JSON)."
            ),
        )

    @classmethod
    def parse_cohere_tools(
        cls: type[ResponseSchema],
        completion: ChatCompletion,
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
    ) -> BaseModel:
        """Legacy Cohere tools parser (deprecated)."""
        return cls._parse_with_registry(
            completion,
            mode=Mode.TOOLS,
            provider=Provider.COHERE,
            validation_context=validation_context,
            strict=strict,
            warning=(
                "ResponseSchema.parse_cohere_tools is deprecated. "
                "Use process_response(..., provider=Provider.COHERE, mode=Mode.TOOLS)."
            ),
        )

    @classmethod
    def parse_writer_tools(
        cls: type[ResponseSchema],
        completion: ChatCompletion,
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
    ) -> BaseModel:
        """Legacy Writer tools parser (deprecated)."""
        return cls._parse_with_registry(
            completion,
            mode=Mode.TOOLS,
            provider=Provider.WRITER,
            validation_context=validation_context,
            strict=strict,
            warning=(
                "ResponseSchema.parse_writer_tools is deprecated. "
                "Use process_response(..., provider=Provider.WRITER, mode=Mode.TOOLS)."
            ),
        )

    @classmethod
    def parse_writer_json(
        cls: type[ResponseSchema],
        completion: ChatCompletion,
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
    ) -> BaseModel:
        """Legacy Writer JSON parser (deprecated)."""
        return cls._parse_with_registry(
            completion,
            mode=Mode.MD_JSON,
            provider=Provider.WRITER,
            validation_context=validation_context,
            strict=strict,
            warning=(
                "ResponseSchema.parse_writer_json is deprecated. "
                "Use process_response(..., provider=Provider.WRITER, mode=Mode.MD_JSON)."
            ),
        )

    @classmethod
    def parse_functions(
        cls: type[ResponseSchema],
        completion: ChatCompletion,
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
    ) -> BaseModel:
        """Legacy OpenAI FUNCTIONS parser (deprecated)."""
        return cls._parse_with_registry(
            completion,
            mode=Mode.FUNCTIONS,
            provider=Provider.OPENAI,
            validation_context=validation_context,
            strict=strict,
            warning=(
                "ResponseSchema.parse_functions is deprecated. "
                "Use process_response(..., mode=Mode.TOOLS) or ResponseSchema.from_response."
            ),
        )

    @classmethod
    def parse_responses_tools(
        cls: type[ResponseSchema],
        completion: Any,
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
    ) -> BaseModel:
        """Legacy OpenAI Responses Tools parser (deprecated)."""
        return cls._parse_with_registry(
            completion,
            mode=Mode.RESPONSES_TOOLS,
            provider=Provider.OPENAI,
            validation_context=validation_context,
            strict=strict,
            warning=(
                "ResponseSchema.parse_responses_tools is deprecated. "
                "Use process_response(..., mode=Mode.RESPONSES_TOOLS) or ResponseSchema.from_response."
            ),
        )

    @classmethod
    def parse_tools(
        cls: type[ResponseSchema],
        completion: ChatCompletion,
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
    ) -> BaseModel:
        """Legacy OpenAI tools parser (deprecated)."""
        return cls._parse_with_registry(
            completion,
            mode=Mode.TOOLS,
            provider=Provider.OPENAI,
            validation_context=validation_context,
            strict=strict,
            warning=(
                "ResponseSchema.parse_tools is deprecated. "
                "Use process_response(..., mode=Mode.TOOLS) or ResponseSchema.from_response."
            ),
        )

    @classmethod
    def parse_mistral_structured_outputs(
        cls: type[ResponseSchema],
        completion: ChatCompletion,
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
    ) -> BaseModel:
        """Legacy Mistral structured-output parser (deprecated)."""
        return cls._parse_with_registry(
            completion,
            mode=Mode.JSON_SCHEMA,
            provider=Provider.MISTRAL,
            validation_context=validation_context,
            strict=strict,
            warning=(
                "ResponseSchema.parse_mistral_structured_outputs is deprecated. "
                "Use process_response(..., provider=Provider.MISTRAL, mode=Mode.JSON_SCHEMA)."
            ),
        )

    @classmethod
    def parse_json(
        cls: type[ResponseSchema],
        completion: ChatCompletion,
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
    ) -> BaseModel:
        """Legacy JSON parser (deprecated)."""
        return cls._parse_with_registry(
            completion,
            mode=Mode.JSON,
            provider=Provider.OPENAI,
            validation_context=validation_context,
            strict=strict,
            warning=(
                "ResponseSchema.parse_json is deprecated. "
                "Use process_response(..., mode=Mode.JSON) or ResponseSchema.from_response."
            ),
        )


# Cache for response_schema: avoids creating a new Pydantic class on every
# create() call.  Keyed weakly on the input model so dynamically-generated
# classes are cleaned up when the caller drops them.
_response_schema_cache: weakref.WeakKeyDictionary[type, type[BaseModel]] = (
    weakref.WeakKeyDictionary()
)


def response_schema(cls: type[Model]) -> type[Model]:
    """Wrap a Pydantic model class to add ResponseSchema behavior."""
    if not inspect.isclass(cls) or not issubclass(cls, BaseModel):
        got = cls.__name__ if inspect.isclass(cls) else type(cls).__name__
        raise TypeError(
            f"response_model must be a subclass of pydantic.BaseModel, got {got}"
        )

    cached = _response_schema_cache.get(cls)
    if cached is not None:
        return cast(type[Model], cached)

    # Create the wrapped model
    schema = cast(
        type[BaseModel],
        wraps(cls, updated=())(
            cast(
                Any,
                create_model(
                    cls.__name__ if hasattr(cls, "__name__") else str(cls),
                    __base__=(cls, ResponseSchema),
                ),
            )
        ),
    )

    _response_schema_cache[cls] = schema
    return cast(type[Model], schema)


# Backward compatibility aliases
openai_schema = response_schema
OpenAISchema = ResponseSchema
