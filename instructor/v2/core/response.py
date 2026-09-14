"""
This module serves as the central dispatcher for processing responses from various LLM providers
(OpenAI, Anthropic, Google, Cohere, etc.) and transforming them into structured Pydantic models.
It handles different response formats, streaming responses, validation, and error recovery.

The module supports 40+ different modes across providers, each with specific handling logic
for request formatting and response parsing. It also provides retry mechanisms (reask) for
handling validation errors gracefully.

Key Components:
    - Response processing functions for sync/async operations
    - Mode-based response model handlers for different providers
    - Error recovery and retry logic for validation failures
    - Support for streaming, partial, parallel, and iterable response models

Example:
    ```python
    from instructor.v2.core.response import process_response
    from instructor.v2.core.mode import Mode
    from pydantic import BaseModel

    class User(BaseModel):
        name: str
        age: int

    # Process an OpenAI response
    processed = process_response(
        response=openai_response,
        response_model=User,
        mode=Mode.TOOLS,
        stream=False
    )
    ```
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, TypeVar, TYPE_CHECKING, cast

from openai.types.chat import ChatCompletion
from pydantic import BaseModel
from typing_extensions import ParamSpec

from instructor.v2.core.errors import InstructorError

from instructor.v2.dsl.iterable import IterableBase
from instructor.v2.dsl.parallel import ParallelBase
from instructor.v2.dsl.partial import PartialBase
from instructor.v2.dsl.response_list import ListResponse
from instructor.v2.dsl.simple_type import AdapterBase

if TYPE_CHECKING:
    from instructor.v2.core.function_calls import ResponseSchema
from instructor.v2.core.messages import isolate_retry_kwargs
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import (
    Provider,
    normalize_mode_for_provider,
    provider_from_mode,
)
from instructor.v2.core.response_model import (
    is_typed_dict as is_typed_dict,
    prepare_response_model,
)
from instructor.v2.core.registry import mode_registry

logger = logging.getLogger("instructor")

T_Model = TypeVar("T_Model", bound=BaseModel)
T_Retval = TypeVar("T_Retval")
T_ParamSpec = ParamSpec("T_ParamSpec")
T = TypeVar("T")

_SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "access_token",
        "api_key",
        "api_secret",
        "authorization",
        "client_secret",
        "refresh_token",
        "token",
        "x_api_key",
        "x_goog_api_key",
    }
)


def _redact_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return a redacted copy of kwargs suitable for debug logging."""

    def redact(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: "[redacted]"
                if key.lower().replace("-", "_") in _SENSITIVE_KEYS
                else redact(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [redact(item) for item in value]
        if isinstance(value, tuple):
            return tuple(redact(item) for item in value)
        return value

    return redact(kwargs)


def _ensure_registry_loaded() -> None:
    """Ensure v2 handlers are imported so the registry is populated."""
    try:
        import importlib

        importlib.import_module("instructor.v2")
    except Exception:
        # Best-effort: allow downstream KeyError to surface if registry is empty.
        return


def _finalize_response(model: Any, response_model: Any, response: Any) -> Any:
    if isinstance(model, IterableBase):
        logger.debug(f"Returning takes from IterableBase")
        return ListResponse.from_list(  # type: ignore[return-value]
            list(cast(Any, model).tasks),
            raw_response=response,
        )
    if isinstance(model, list) and not isinstance(model, ListResponse):
        logger.debug("Wrapping list response with ListResponse")
        return ListResponse.from_list(model, raw_response=response)

    if isinstance(response_model, ParallelBase):
        logger.debug(f"Returning model from ParallelBase")
        object.__setattr__(model, "_raw_response", response)
        return model

    if isinstance(model, AdapterBase):
        logger.debug(f"Returning model from AdapterBase")
        return cast(Any, model).content

    if isinstance(model, BaseModel):
        object.__setattr__(model, "_raw_response", response)
    return model


async def process_response_async(
    response: ChatCompletion,
    *,
    response_model: type[T_Model | ResponseSchema | BaseModel] | None,
    stream: bool = False,
    validation_context: dict[str, Any] | None = None,
    strict: bool | None = None,
    mode: Mode = Mode.TOOLS,
    provider: Provider = Provider.OPENAI,
) -> Any:
    """Asynchronously process and transform LLM responses into structured models.

    This function is the async entry point for converting raw LLM responses into validated
    Pydantic models. It handles various response formats from different providers and
    supports special response types like streaming, partial objects, and parallel tool calls.

    Args:
        response (ChatCompletion or Similar API Response): The raw response from the LLM API. Despite the type hint,
            this can be responses from any supported provider (OpenAI, Anthropic, Google, etc.)
        response_model (type[T_Model | BaseModel] | None): The target Pydantic
            model to parse the response into. If None, returns the raw response unchanged.
            Can also be special DSL types like ParallelBase for parallel tool calls, or IterableBase and PartialBase for streaming.
        stream (bool): Whether this is a streaming response. Required for proper handling
            of IterableBase and PartialBase models. Defaults to False.
        validation_context (dict[str, Any] | None): Additional context passed to Pydantic
            validators during model validation. Useful for dynamic validation logic. The context
            is also used to format templated responses. Defaults to None.
        strict (bool | None): Whether to enforce strict JSON parsing. When True, the response
            must exactly match the model schema. When False, allows minor deviations.
        mode (Mode): The provider/format mode that determines how to parse the response.
            Examples: Mode.TOOLS (OpenAI), Mode.ANTHROPIC_JSON, Mode.GEMINI_TOOLS.
            Defaults to Mode.TOOLS.
        provider (Provider): The LLM provider used for handler lookup.

    Returns:
        T_Model | ChatCompletion: The processed response. Return type depends on inputs:
            - If response_model is None: returns raw response unchanged
            - If response_model is IterableBase with stream=True: returns list of models
            - If response_model is AdapterBase: returns the adapted content
            - Otherwise: returns instance of response_model with _raw_response attached

    Raises:
        ValidationError: If the response doesn't match the expected model schema
        IncompleteOutputException: If the response was truncated due to token limits
        ValueError: If an invalid mode is specified

    Note:
        The function automatically detects special response model types (Iterable, Partial,
        Parallel, Adapter) and applies appropriate processing logic for each.
    """

    logger.debug(
        f"Instructor Raw Response: {response}",
    )
    from instructor.v2.validation.async_validators import reject_async_validators

    reject_async_validators(response_model)
    if response_model is None:
        return response

    provider = provider_from_mode(mode, provider)
    mode = normalize_mode_for_provider(mode, provider)

    if (
        inspect.isclass(response_model)
        and issubclass(response_model, IterableBase)
        and not stream
        and not hasattr(response, "choices")
        and hasattr(response_model, "from_response")
    ):
        dynamic_response_model = cast(Any, response_model)
        model = dynamic_response_model.from_response(
            response,
            validation_context=validation_context,
            strict=strict,
            mode=mode,
        )
        return ListResponse.from_list(
            list(model.tasks),
            raw_response=response,
        )

    _ensure_registry_loaded()
    handlers = mode_registry.get_handlers(provider, mode)
    handler_obj = getattr(handlers.response_parser, "__self__", None)
    if handler_obj and hasattr(handler_obj, "mark_streaming_model"):
        handler_obj.mark_streaming_model(response_model, stream)

    model = handlers.response_parser(
        response=response,
        response_model=response_model,
        validation_context=validation_context,
        strict=strict,
        stream=stream,
        is_async=True,
    )

    if inspect.isasyncgen(model):
        return model
    if (
        stream
        and inspect.isclass(response_model)
        and issubclass(response_model, PartialBase)
    ):
        return model

    return _finalize_response(model, response_model, response)


def process_response(
    response: T_Model,
    *,
    response_model: type[ResponseSchema | BaseModel] | None = None,
    stream: bool,
    validation_context: dict[str, Any] | None = None,
    strict=None,
    mode: Mode = Mode.TOOLS,
    provider: Provider = Provider.OPENAI,
) -> Any:
    """Process and transform LLM responses into structured models (synchronous).

    This is the main entry point for converting raw LLM responses into validated Pydantic
    models. It acts as a dispatcher that handles various response formats from 40+ different
    provider modes and transforms them according to the specified response model type.

    Args:
        response (T_Model): The raw response from the LLM API. The actual type varies by
            provider (ChatCompletion for OpenAI, Message for Anthropic, etc.)
        response_model (type[ResponseSchema | BaseModel] | None): The target Pydantic model
            class to parse the response into. Special DSL types supported:
            - IterableBase: For streaming multiple objects from a single response
            - PartialBase: For incomplete/streaming partial objects
            - ParallelBase: For parallel tool/function calls
            - AdapterBase: For simple type adaptations (e.g., str, int)
            If None, returns the raw response unchanged.
        stream (bool): Whether this is a streaming response. Required to be True for
            proper handling of IterableBase and PartialBase models.
        validation_context (dict[str, Any] | None): Additional context passed to Pydantic
            validators. Useful for runtime validation logic based on external state.
        strict (bool | None): Controls JSON parsing strictness:
            - True: Enforce exact schema matching (no extra fields)
            - False/None: Allow minor deviations and extra fields
        mode (Mode): The provider/format mode that determines parsing strategy.
            Each mode corresponds to a specific provider and format combination:
            - Tool modes: TOOLS, ANTHROPIC_TOOLS, GEMINI_TOOLS, etc.
            - JSON modes: JSON, ANTHROPIC_JSON, VERTEXAI_JSON, etc.
            - Special modes: PARALLEL_TOOLS, MD_JSON, JSON_SCHEMA, etc.
        provider (Provider): The LLM provider used for handler lookup.

    Returns:
        T_Model | list[T_Model] | None: The processed response:
            - If response_model is None: Original response unchanged
            - If IterableBase: List of extracted model instances
            - If ParallelBase: Special parallel response object
            - If AdapterBase: The adapted simple type (str, int, etc.)
            - Otherwise: Single instance of response_model with _raw_response attached

    Raises:
        ValidationError: Response doesn't match the expected model schema
        IncompleteOutputException: Response truncated due to token limits
        ValueError: Invalid mode specified or mode not supported
        JSONDecodeError: Malformed JSON in response (for JSON modes)

    Note:
        The function preserves the raw response by attaching it to the parsed model
        as `_raw_response`. This allows access to metadata like token usage, model
        info, and other provider-specific fields after parsing.
    """
    logger.debug(
        f"Instructor Raw Response: {response}",
    )

    from instructor.v2.validation.async_validators import reject_async_validators

    reject_async_validators(response_model)
    if response_model is None:
        logger.debug("No response model, returning response as is")
        return response

    provider = provider_from_mode(mode, provider)
    mode = normalize_mode_for_provider(mode, provider)

    if (
        inspect.isclass(response_model)
        and issubclass(response_model, IterableBase)
        and not stream
        and not hasattr(response, "choices")
        and hasattr(response_model, "from_response")
    ):
        dynamic_response_model = cast(Any, response_model)
        model = dynamic_response_model.from_response(
            response,
            validation_context=validation_context,
            strict=strict,
            mode=mode,
        )
        return ListResponse.from_list(
            list(model.tasks),
            raw_response=response,
        )

    _ensure_registry_loaded()
    handlers = mode_registry.get_handlers(provider, mode)
    handler_obj = getattr(handlers.response_parser, "__self__", None)
    if handler_obj and hasattr(handler_obj, "mark_streaming_model"):
        handler_obj.mark_streaming_model(response_model, stream)

    model = handlers.response_parser(
        response=response,
        response_model=response_model,
        validation_context=validation_context,
        strict=strict,
        stream=stream,
        is_async=False,
    )

    if inspect.isgenerator(model):
        return model
    if (
        stream
        and inspect.isclass(response_model)
        and issubclass(response_model, PartialBase)
    ):
        return model

    return _finalize_response(model, response_model, response)


def handle_response_model(
    response_model: type[T] | None,
    mode: Mode = Mode.TOOLS,
    provider: Provider = Provider.OPENAI,
    **kwargs: Any,
) -> tuple[type[T] | None, dict[str, Any]]:
    """
    Handles the response model based on the specified mode and prepares the kwargs for the API call.
    This really should be named 'prepare_create_kwargs' as its job is to map the openai create kwargs
    to the correct format for the API call based on the mode.

    Args:
        response_model (type[T] | None): The response model to be used for parsing the API response.
        mode (Mode): The mode to use for handling the response model. Defaults to Mode.TOOLS.
        provider (Provider): The LLM provider used for handler lookup.
        **kwargs: Additional keyword arguments to be passed to the API call.

    Returns:
        tuple[type[T] | None, dict[str, Any]]: A tuple containing the processed response model and the updated kwargs.

    This function prepares the response model and modifies the kwargs based on the specified mode.
    It handles various modes like TOOLS, JSON, FUNCTIONS, etc., and applies the appropriate
    transformations to the response model and kwargs.
    """

    provider = provider_from_mode(mode, provider)
    mode = normalize_mode_for_provider(mode, provider)
    new_kwargs = kwargs.copy()
    autodetect_images = bool(new_kwargs.pop("autodetect_images", False))

    # Only prepare response_model if it's not None
    if response_model is not None:
        response_model = prepare_response_model(response_model)

    _ensure_registry_loaded()
    handlers = mode_registry.get_handlers(provider, mode)
    response_model, new_kwargs = handlers.request_handler(response_model, new_kwargs)

    # Handle message conversion for modes that don't already handle it
    if handlers.message_converter and "messages" in new_kwargs:
        new_kwargs["messages"] = handlers.message_converter(
            new_kwargs["messages"],
            autodetect_images=autodetect_images,
        )

    redacted_kwargs = _redact_kwargs(new_kwargs)
    logger.debug(
        f"Instructor Request: {mode.value=}, {response_model=}, {redacted_kwargs=}",
        extra={
            "mode": mode.value,
            "response_model": (
                response_model.__name__
                if response_model is not None and hasattr(response_model, "__name__")
                else str(response_model)
            ),
            "new_kwargs": redacted_kwargs,
        },
    )
    return response_model, new_kwargs


def handle_reask_kwargs(
    kwargs: dict[str, Any],
    mode: Mode,
    response: Any,
    exception: Exception,
    provider: Provider = Provider.OPENAI,
    failed_attempts: list[Any] | None = None,
) -> dict[str, Any]:
    """Compatibility dispatcher for provider-specific reask formatting.

    The retry loop itself lives in :mod:`instructor.v2.core.retry` and dispatches
    directly through the registry. This helper preserves the historical public API for
    callers that need to format one reask payload directly.

    The reask process involves:
    1. Analyzing the validation error and failed response
    2. Selecting the appropriate provider-specific reask handler
    3. Enriching the exception with retry history (failed_attempts)
    4. Formatting error feedback in the provider's expected message format
    5. Preserving original request parameters while adding retry context

    Args:
        kwargs (dict[str, Any]): The original request parameters that resulted in
            a validation error. Contains all parameters passed to the LLM API:
            - messages: conversation history
            - tools/functions: available function definitions
            - temperature, max_tokens: generation parameters
            - model, provider-specific settings
        mode (Mode): The provider/format mode that determines which reask handler
            to use. Each mode implements a specific strategy for formatting error
            feedback and retry messages. Examples:
            - Mode.TOOLS: OpenAI function calling
            - Mode.ANTHROPIC_TOOLS: Anthropic tool use
            - Mode.JSON: JSON-only responses
        provider (Provider): The LLM provider used for handler lookup.
        response (Any): The raw response from the LLM that failed validation.
            Type and structure varies by provider:
            - OpenAI: ChatCompletion with tool_calls or content
            - Anthropic: Message with tool_use blocks or text content
            - Google: GenerateContentResponse with function calls
            - Cohere: NonStreamedChatResponse with tool calls
        exception (Exception): The validation error that occurred, typically:
            - Pydantic ValidationError: field validation failures
            - JSONDecodeError: malformed JSON responses
            - Custom validation errors from response processors
            The exception will be enriched with failed_attempts data.
        failed_attempts (list[FailedAttempt] | None): Historical record of previous
            retry attempts for this request. Each FailedAttempt contains:
            - attempt_number: sequential attempt counter
            - exception: the validation error for that attempt
            - completion: the raw LLM response that failed
            Used to provide retry context and prevent repeated mistakes.

    Returns:
        dict[str, Any]: Modified kwargs for the retry request with:
            - Updated messages including error feedback
            - Original tool/function definitions preserved
            - Generation parameters maintained (temperature, etc.)
            - Provider-specific error formatting applied
            - Retry context embedded in appropriate message format

    Provider-Specific Reask Strategies:
        **OpenAI Modes:**
        - TOOLS/FUNCTIONS: Adds tool response messages with validation errors
        - JSON modes: Appends user message with correction instructions
        - Preserves function schemas and conversation context

        **Anthropic Modes:**
        - TOOLS: Creates tool_result blocks with error details
        - JSON: Adds user message with structured error feedback
        - Maintains conversation flow with proper message roles

        **Google/Gemini Modes:**
        - TOOLS: Formats as function response with error content
        - JSON: Appends user message with validation feedback

        **Other Providers (Cohere, Mistral, etc.):**
        - Provider-specific message formatting
        - Consistent error reporting patterns
        - Maintained conversation context

    Error Enrichment:
        The exception parameter is enriched with retry metadata:
        - exception.failed_attempts: list of previous failures
        - exception.retry_attempt_number: current attempt number
        This allows downstream handlers to access full retry context.

    Example:
        ```python
        # After a ValidationError occurs during retry attempt #2
        new_kwargs = handle_reask_kwargs(
            kwargs=original_request,
            mode=Mode.TOOLS,
            provider=Provider.OPENAI,
            response=failed_completion,
            exception=validation_error,  # Will be enriched with failed_attempts
            failed_attempts=[attempt1, attempt2]  # Previous failures
        )
        # new_kwargs now contains retry messages with error context
        ```

    Note:
        Provider-specific formatting still lives on each registered mode handler.
    """
    # Isolate the request lists a reask handler may mutate in place
    # (``messages``/``contents``/``chat_history``) so formatting one reask
    # payload cannot leak appended messages into the caller's own kwargs.
    kwargs_copy = isolate_retry_kwargs(kwargs)

    exception = InstructorError.from_exception(
        exception, failed_attempts=failed_attempts
    )

    provider = provider_from_mode(mode, provider)
    mode = normalize_mode_for_provider(mode, provider)
    _ensure_registry_loaded()
    handlers = mode_registry.get_handlers(provider, mode)
    return handlers.reask_handler(kwargs_copy, response, exception)
