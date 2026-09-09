"""v2 retry mechanism using registry handlers.

Custom retry logic for v2 that uses registry's reask and response_parser
instead of v1's process_response.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
)

from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.budget import (
    _budget_error as _budget_error,
    _validate_token_budget as _validate_token_budget,
)
from instructor.v2.core.errors import (
    AsyncValidationError,
    FailedAttempt,
    IncompleteOutputException,
    InstructorRetryException,
    ResponseParsingError,
    TokenBudgetError,
)
from instructor.v2.dsl.iterable import IterableBase
from instructor.v2.dsl.response_list import ListResponse
from instructor.v2.dsl.simple_type import AdapterBase
from instructor.v2.core.messages import extract_messages
from instructor.v2.core.usage import (
    _usage_snapshot as _usage_snapshot,
    _usage_total_tokens as _usage_total_tokens,
    has_compatible_usage,
    update_total_usage,
)
from instructor.v2.core.exceptions import RegistryValidationMixin
from instructor.v2.core.registry import mode_registry

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from instructor.v2.core.hooks import Hooks

logger = logging.getLogger("instructor.v2.retry")

T_Model = TypeVar("T_Model", bound=BaseModel)
_RETRYABLE_PARSE_ERRORS = (
    ValidationError,
    json.JSONDecodeError,
    AsyncValidationError,
    ResponseParsingError,
)


def _max_attempts(max_retries: int | Retrying | AsyncRetrying) -> int | None:
    return max(max_retries, 0) + 1 if isinstance(max_retries, int) else None


def _attempt_metadata(
    *,
    attempt_number: int,
    max_attempts: int | None,
    is_last_attempt: bool,
) -> dict[str, Any]:
    return {
        "attempt_number": attempt_number,
        "max_attempts": max_attempts,
        "is_last_attempt": is_last_attempt,
    }


def _finalize_parsed_response(
    parsed: Any,
    response: Any,
    total_usage: Any | None = None,
) -> Any:
    usage = _usage_snapshot(total_usage) if total_usage is not None else None
    if isinstance(parsed, IterableBase):
        parsed = [task for task in parsed.tasks]
    if isinstance(parsed, AdapterBase):
        return parsed.content
    if isinstance(parsed, list) and not isinstance(parsed, ListResponse):
        return ListResponse.from_list(
            parsed,
            raw_response=response,
            total_usage=usage,
        )
    if isinstance(parsed, ListResponse):
        parsed._raw_response = response
        parsed._total_usage = usage
        return parsed
    if isinstance(parsed, BaseModel):
        parsed._raw_response = response  # type: ignore[attr-defined]
        if usage is not None:
            parsed._total_usage = usage  # type: ignore[attr-defined]
    return parsed


def _initialize_usage(provider: Provider | Mode) -> Any:
    from openai.types.completion_usage import (
        CompletionTokensDetails,
        CompletionUsage,
        PromptTokensDetails,
    )

    total_usage: Any = CompletionUsage(
        completion_tokens=0,
        prompt_tokens=0,
        total_tokens=0,
        completion_tokens_details=CompletionTokensDetails(
            audio_tokens=0, reasoning_tokens=0
        ),
        prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0),
    )
    anthropic_modes = {
        Mode.ANTHROPIC_TOOLS,
        Mode.ANTHROPIC_REASONING_TOOLS,
        Mode.ANTHROPIC_JSON,
        Mode.ANTHROPIC_PARALLEL_TOOLS,
    }
    if provider is Provider.ANTHROPIC or provider in anthropic_modes:
        from instructor.v2.providers.anthropic.usage import initialize_usage

        total_usage = initialize_usage()
    return total_usage


def retry_sync_v2(
    func: Callable[..., Any],
    response_model: type[T_Model] | None,
    provider: Provider,
    mode: Mode,
    context: dict[str, Any] | None,
    max_retries: int | Retrying,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    strict: bool,
    hooks: Hooks | None = None,
    token_budget: int | None = None,
) -> T_Model:
    """Sync retry logic using v2 registry handlers.

    Args:
        func: API function to call
        response_model: Pydantic model to extract
        provider: Provider enum
        mode: Mode enum
        context: Validation context
        max_retries: Maximum retries after the initial attempt, or Retrying instance
        args: Positional args for func
        kwargs: Keyword args for func
        strict: Strict validation mode
        hooks: Optional hooks
        token_budget: Positive cumulative token budget for validation retries

    Returns:
        Validated Pydantic model instance

    Raises:
        InstructorRetryException: If max retries exceeded
        TokenBudgetExceeded: If a failed attempt exhausts the retry budget
        TokenUsageUnavailableError: If a retry budget cannot be enforced
    """
    from instructor.v2.validation.async_validators import reject_async_validators

    reject_async_validators(response_model)
    _validate_token_budget(
        token_budget,
        response_model=response_model,
        kwargs=kwargs,
    )
    if response_model is None:
        # No structured output, just call the API
        return func(*args, **kwargs)

    # Validate and get handlers from registry
    RegistryValidationMixin.validate_mode_registration(provider, mode)
    handlers = mode_registry.get_handlers(provider, mode)

    # Setup retrying
    if isinstance(max_retries, int):
        stop_condition = stop_after_attempt(max(max_retries, 0) + 1)
        timeout = kwargs.get("timeout")
        if isinstance(timeout, (int, float)):
            stop_condition = stop_condition | stop_after_delay(timeout)
        max_retries_instance = Retrying(
            stop=stop_condition,
            retry=retry_if_exception_type(_RETRYABLE_PARSE_ERRORS),
            reraise=True,
        )
    else:
        max_retries_instance = max_retries

    max_attempts = _max_attempts(max_retries)
    failed_attempts: list[FailedAttempt] = []
    last_exception: Exception | None = None
    last_attempt_number = 0
    total_usage = _initialize_usage(provider)
    usage_complete = True

    try:
        for attempt in max_retries_instance:
            with attempt:
                attempt_number = attempt.retry_state.attempt_number
                last_attempt_number = attempt_number
                # Call API
                if hooks:
                    hooks.emit_completion_arguments(**kwargs)

                try:
                    response = func(*args, **kwargs)
                except IncompleteOutputException:
                    raise
                except Exception as e:
                    logger.error(f"API call failed on attempt {attempt_number}: {e}")
                    if hooks:
                        hooks.emit_completion_error(
                            e,
                            **_attempt_metadata(
                                attempt_number=attempt_number,
                                max_attempts=max_attempts,
                                is_last_attempt=(
                                    max_attempts is not None
                                    and (
                                        not isinstance(e, ValidationError)
                                        or attempt_number >= max_attempts
                                    )
                                ),
                            ),
                        )
                    raise

                if hooks:
                    hooks.emit_completion_response(response)

                usage_available = has_compatible_usage(response, total_usage)
                usage_complete = usage_complete and usage_available
                update_total_usage(response=response, total_usage=total_usage)
                if hooks and usage_complete:
                    hooks.emit_completion_usage(
                        _usage_snapshot(total_usage),
                        attempt_number=attempt_number,
                    )

                # Parse response using registry
                try:
                    stream = kwargs.get("stream", False)
                    parsed = handlers.response_parser(
                        response=response,
                        response_model=response_model,
                        validation_context=context,
                        strict=strict,
                        stream=stream,
                        is_async=False,
                    )
                    logger.debug(
                        f"Successfully parsed response on attempt "
                        f"{attempt.retry_state.attempt_number}"
                    )
                    return _finalize_parsed_response(
                        parsed,
                        response,
                        total_usage=total_usage if usage_complete else None,
                    )

                except IncompleteOutputException:
                    raise
                except _RETRYABLE_PARSE_ERRORS as e:
                    logger.debug(f"Validation error on attempt {attempt_number}: {e}")
                    failed_attempts.append(
                        FailedAttempt(
                            attempt_number=attempt_number,
                            exception=e,
                            completion=response,
                        )
                    )
                    last_exception = e

                    budget_error = _budget_error(
                        token_budget=token_budget,
                        usage_available=usage_complete,
                        total_usage=total_usage,
                        attempt_number=attempt_number,
                        response=response,
                        kwargs=kwargs,
                        failed_attempts=failed_attempts,
                    )
                    if hooks:
                        hooks.emit_parse_error(
                            e,
                            **_attempt_metadata(
                                attempt_number=attempt_number,
                                max_attempts=max_attempts,
                                is_last_attempt=(
                                    max_attempts == attempt_number
                                    or budget_error is not None
                                ),
                            ),
                        )
                    if budget_error is not None:
                        if hooks:
                            hooks.emit_completion_last_attempt(
                                budget_error,
                                **_attempt_metadata(
                                    attempt_number=attempt_number,
                                    max_attempts=max_attempts,
                                    is_last_attempt=True,
                                ),
                            )
                        raise budget_error from e
                    # Prepare reask using registry
                    kwargs = handlers.reask_handler(
                        kwargs=kwargs,
                        response=response,
                        exception=e,
                    )

                    # Will retry with modified kwargs
                    raise

    except (IncompleteOutputException, TokenBudgetError):
        raise
    except Exception as e:
        # Max retries exceeded or non-validation error occurred
        last_exception = e

        logger.error(
            f"Max retries exceeded. Total attempts: {last_attempt_number}, "
            f"Last error: {last_exception}"
        )
        if hooks:
            hooks.emit_completion_last_attempt(
                last_exception,
                **_attempt_metadata(
                    attempt_number=last_attempt_number or len(failed_attempts),
                    max_attempts=max_attempts,
                    is_last_attempt=True,
                ),
            )

        raise InstructorRetryException(
            str(last_exception),
            last_completion=failed_attempts[-1].completion if failed_attempts else None,
            n_attempts=last_attempt_number,
            total_usage=total_usage,
            messages=extract_messages(kwargs),
            create_kwargs=kwargs,
            failed_attempts=failed_attempts,
        ) from last_exception

    # Should never reach here
    logger.error("Unexpected code path in retry_sync_v2")
    raise InstructorRetryException(
        str(last_exception) if last_exception else "Unknown error",
        last_completion=failed_attempts[-1].completion if failed_attempts else None,
        n_attempts=last_attempt_number,
        total_usage=total_usage,
        messages=extract_messages(kwargs),
        create_kwargs=kwargs,
        failed_attempts=failed_attempts,
    )


def retry_sync(
    func: Callable[..., Any],
    response_model: type[T_Model] | None,
    args: Any,
    kwargs: Any,
    context: dict[str, Any] | None = None,
    max_retries: int | Retrying = 1,
    strict: bool | None = None,
    mode: Mode = Mode.TOOLS,
    provider: Provider = Provider.OPENAI,
    hooks: Hooks | None = None,
    token_budget: int | None = None,
) -> T_Model | None:
    """Compatibility wrapper for the public retry API."""
    strict_value = True if strict is None else strict
    return retry_sync_v2(
        func=func,
        response_model=response_model,
        provider=provider,
        mode=mode,
        context=context,
        max_retries=max_retries,
        args=tuple(args) if isinstance(args, tuple) else args,
        kwargs=dict(kwargs),
        strict=strict_value,
        hooks=hooks,
        token_budget=token_budget,
    )


async def retry_async(
    func: Callable[..., Any],
    response_model: type[T_Model] | None,
    args: Any,
    kwargs: Any,
    context: dict[str, Any] | None = None,
    max_retries: int | AsyncRetrying = 1,
    strict: bool | None = None,
    mode: Mode = Mode.TOOLS,
    provider: Provider = Provider.OPENAI,
    hooks: Hooks | None = None,
    token_budget: int | None = None,
) -> T_Model | None:
    """Compatibility wrapper for the public retry API."""
    strict_value = True if strict is None else strict
    return await retry_async_v2(
        func=func,
        response_model=response_model,
        provider=provider,
        mode=mode,
        context=context,
        max_retries=max_retries,
        args=tuple(args) if isinstance(args, tuple) else args,
        kwargs=dict(kwargs),
        strict=strict_value,
        hooks=hooks,
        token_budget=token_budget,
    )


async def retry_async_v2(
    func: Callable[..., Awaitable[Any]],
    response_model: type[T_Model] | None,
    provider: Provider,
    mode: Mode,
    context: dict[str, Any] | None,
    max_retries: int | AsyncRetrying,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    strict: bool,
    hooks: Hooks | None = None,
    token_budget: int | None = None,
) -> T_Model:
    """Async retry logic using v2 registry handlers.

    Args:
        func: Async API function to call
        response_model: Pydantic model to extract
        provider: Provider enum
        mode: Mode enum
        context: Validation context
        max_retries: Maximum retries after the initial attempt, or AsyncRetrying instance
        args: Positional args for func
        kwargs: Keyword args for func
        strict: Strict validation mode
        hooks: Optional hooks
        token_budget: Positive cumulative token budget for validation retries

    Returns:
        Validated Pydantic model instance

    Raises:
        InstructorRetryException: If max retries exceeded
        TokenBudgetExceeded: If a failed attempt exhausts the retry budget
        TokenUsageUnavailableError: If a retry budget cannot be enforced
    """
    from instructor.v2.validation.async_validators import reject_async_validators

    reject_async_validators(response_model)
    _validate_token_budget(
        token_budget,
        response_model=response_model,
        kwargs=kwargs,
    )
    if response_model is None:
        # No structured output, just call the API
        return await func(*args, **kwargs)

    # Validate and get handlers from registry
    RegistryValidationMixin.validate_mode_registration(provider, mode)
    handlers = mode_registry.get_handlers(provider, mode)

    # Setup retrying
    if isinstance(max_retries, int):
        stop_condition = stop_after_attempt(max(max_retries, 0) + 1)
        timeout = kwargs.get("timeout")
        if isinstance(timeout, (int, float)):
            stop_condition = stop_condition | stop_after_delay(timeout)
        max_retries_instance = AsyncRetrying(
            stop=stop_condition,
            retry=retry_if_exception_type(_RETRYABLE_PARSE_ERRORS),
            reraise=True,
        )
    else:
        max_retries_instance = max_retries

    max_attempts = _max_attempts(max_retries)
    failed_attempts: list[FailedAttempt] = []
    last_exception: Exception | None = None
    last_attempt_number = 0
    total_usage = _initialize_usage(provider)
    usage_complete = True

    try:
        async for attempt in max_retries_instance:
            with attempt:
                attempt_number = attempt.retry_state.attempt_number
                last_attempt_number = attempt_number
                # Call API
                if hooks:
                    hooks.emit_completion_arguments(**kwargs)

                try:
                    response = await func(*args, **kwargs)
                except IncompleteOutputException:
                    raise
                except Exception as e:
                    logger.error(f"API call failed on attempt {attempt_number}: {e}")
                    if hooks:
                        hooks.emit_completion_error(
                            e,
                            **_attempt_metadata(
                                attempt_number=attempt_number,
                                max_attempts=max_attempts,
                                is_last_attempt=(
                                    max_attempts is not None
                                    and (
                                        not isinstance(e, ValidationError)
                                        or attempt_number >= max_attempts
                                    )
                                ),
                            ),
                        )
                    raise

                if hooks:
                    hooks.emit_completion_response(response)

                usage_available = has_compatible_usage(response, total_usage)
                usage_complete = usage_complete and usage_available
                update_total_usage(response=response, total_usage=total_usage)
                if hooks and usage_complete:
                    hooks.emit_completion_usage(
                        _usage_snapshot(total_usage),
                        attempt_number=attempt_number,
                    )

                # Parse response using registry
                try:
                    stream = kwargs.get("stream", False)
                    parsed = handlers.response_parser(
                        response=response,
                        response_model=response_model,
                        validation_context=context,
                        strict=strict,
                        stream=stream,
                        is_async=True,
                    )
                    logger.debug(
                        f"Successfully parsed response on attempt "
                        f"{attempt.retry_state.attempt_number}"
                    )
                    return _finalize_parsed_response(
                        parsed,
                        response,
                        total_usage=total_usage if usage_complete else None,
                    )

                except IncompleteOutputException:
                    raise
                except _RETRYABLE_PARSE_ERRORS as e:
                    logger.debug(f"Validation error on attempt {attempt_number}: {e}")
                    failed_attempts.append(
                        FailedAttempt(
                            attempt_number=attempt_number,
                            exception=e,
                            completion=response,
                        )
                    )
                    last_exception = e

                    budget_error = _budget_error(
                        token_budget=token_budget,
                        usage_available=usage_complete,
                        total_usage=total_usage,
                        attempt_number=attempt_number,
                        response=response,
                        kwargs=kwargs,
                        failed_attempts=failed_attempts,
                    )
                    if hooks:
                        hooks.emit_parse_error(
                            e,
                            **_attempt_metadata(
                                attempt_number=attempt_number,
                                max_attempts=max_attempts,
                                is_last_attempt=(
                                    max_attempts == attempt_number
                                    or budget_error is not None
                                ),
                            ),
                        )
                    if budget_error is not None:
                        if hooks:
                            hooks.emit_completion_last_attempt(
                                budget_error,
                                **_attempt_metadata(
                                    attempt_number=attempt_number,
                                    max_attempts=max_attempts,
                                    is_last_attempt=True,
                                ),
                            )
                        raise budget_error from e
                    # Prepare reask using registry
                    kwargs = handlers.reask_handler(
                        kwargs=kwargs,
                        response=response,
                        exception=e,
                    )

                    # Will retry with modified kwargs
                    raise

    except (IncompleteOutputException, TokenBudgetError):
        raise
    except Exception as e:
        # Max retries exceeded or non-validation error occurred
        last_exception = e

        logger.error(
            f"Max retries exceeded. Total attempts: {last_attempt_number}, "
            f"Last error: {last_exception}"
        )
        if hooks:
            hooks.emit_completion_last_attempt(
                last_exception,
                **_attempt_metadata(
                    attempt_number=last_attempt_number or len(failed_attempts),
                    max_attempts=max_attempts,
                    is_last_attempt=True,
                ),
            )

        raise InstructorRetryException(
            str(last_exception),
            last_completion=failed_attempts[-1].completion if failed_attempts else None,
            n_attempts=last_attempt_number,
            total_usage=total_usage,
            messages=extract_messages(kwargs),
            create_kwargs=kwargs,
            failed_attempts=failed_attempts,
        ) from last_exception

    # Should never reach here
    logger.error("Unexpected code path in retry_async_v2")
    raise InstructorRetryException(
        str(last_exception) if last_exception else "Unknown error",
        last_completion=failed_attempts[-1].completion if failed_attempts else None,
        n_attempts=last_attempt_number,
        total_usage=total_usage,
        messages=extract_messages(kwargs),
        create_kwargs=kwargs,
        failed_attempts=failed_attempts,
    )
