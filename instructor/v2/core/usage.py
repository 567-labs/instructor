"""Usage accumulation helpers owned by the v2 runtime."""

from __future__ import annotations

import copy
import logging
from numbers import Real
from typing import TYPE_CHECKING, Any, TypeVar, cast

from pydantic import BaseModel

if TYPE_CHECKING:
    from anthropic.types import Usage as AnthropicUsage
    from openai.types import CompletionUsage as OpenAIUsage

logger = logging.getLogger("instructor")
T_Response = TypeVar("T_Response")


def _usage_snapshot(total_usage: Any) -> Any:
    if isinstance(total_usage, BaseModel):
        return total_usage.model_copy(deep=True)
    return copy.deepcopy(total_usage)


def _usage_total_tokens(total_usage: Any) -> int | None:
    direct_total = getattr(total_usage, "total_tokens", None)
    if isinstance(direct_total, Real) and not isinstance(direct_total, bool):
        return int(direct_total)

    token_fields = (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    )
    values = [getattr(total_usage, field, None) for field in token_fields]
    numeric_values = [
        int(value)
        for value in values
        if isinstance(value, Real) and not isinstance(value, bool)
    ]
    return sum(numeric_values) if numeric_values else None


def _field_names(model: BaseModel | dict[str, Any]) -> set[str]:
    """Return keys, or declared and Pydantic extra field names, for usage data."""
    if isinstance(model, dict):
        return set(model)
    return set(type(model).model_fields) | set(model.model_extra or {})


def _all_field_names(*models: BaseModel | dict[str, Any]) -> set[str]:
    return {field_name for model in models for field_name in _field_names(model)}


def _get(model: BaseModel | dict[str, Any], field_name: str) -> Any:
    if isinstance(model, dict):
        return model.get(field_name)
    return getattr(model, field_name, None)


def _set(model: BaseModel | dict[str, Any], field_name: str, value: Any) -> None:
    if isinstance(model, dict):
        model[field_name] = value
    else:
        setattr(model, field_name, value)


def _is_numeric(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _zero_numeric_fields(model: Any) -> Any:
    for field_name in _field_names(model):
        value = _get(model, field_name)
        if isinstance(value, (BaseModel, dict)):
            _zero_numeric_fields(value)
        elif _is_numeric(value):
            _set(model, field_name, 0)
    return model


def _accumulate_models(
    response: BaseModel | dict[str, Any], total: BaseModel | dict[str, Any]
) -> None:
    """Add numeric usage, recursing into nested models and dicts such as cost details."""
    for field_name in _all_field_names(response, total):
        response_value = _get(response, field_name)
        total_value = _get(total, field_name)
        if isinstance(response_value, (BaseModel, dict)):
            if not isinstance(total_value, (BaseModel, dict)):
                total_value = _zero_numeric_fields(_usage_snapshot(response_value))
                _set(total, field_name, total_value)
            _accumulate_models(response_value, total_value)
            _set(response, field_name, _usage_snapshot(total_value))
        elif isinstance(total_value, (BaseModel, dict)):
            _set(response, field_name, _usage_snapshot(total_value))
        elif _is_numeric(response_value):
            response_number = cast(Real, response_value)
            total_number = cast(Real, total_value) if _is_numeric(total_value) else 0
            value = total_number + response_number
            _set(total, field_name, value)
            _set(response, field_name, value)
        elif _is_numeric(total_value):
            _set(response, field_name, total_value)
        elif response_value is not None:
            _set(total, field_name, response_value)
            _set(response, field_name, response_value)
        elif total_value is not None:
            _set(response, field_name, total_value)


def _response_usage_as_completion(response_usage: Any) -> Any:
    """Project an OpenAI Responses-API ``ResponseUsage`` onto a Chat ``CompletionUsage``.

    The Responses API reports ``input_tokens``/``output_tokens`` while the running
    total is a Chat ``CompletionUsage`` (``prompt_tokens``/``completion_tokens``).
    Projecting onto the shared shape lets us reuse :func:`_accumulate_models`
    instead of maintaining a parallel accumulation code path.
    """
    from openai.types.completion_usage import (
        CompletionTokensDetails,
        CompletionUsage,
        PromptTokensDetails,
    )

    in_details = getattr(response_usage, "input_tokens_details", None)
    out_details = getattr(response_usage, "output_tokens_details", None)
    cached_tokens = getattr(in_details, "cached_tokens", None) or 0
    reasoning_tokens = getattr(out_details, "reasoning_tokens", None) or 0

    return CompletionUsage(
        prompt_tokens=response_usage.input_tokens,
        completion_tokens=response_usage.output_tokens,
        total_tokens=response_usage.total_tokens,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=cached_tokens),
        completion_tokens_details=CompletionTokensDetails(
            reasoning_tokens=reasoning_tokens
        ),
    )


def _openai_usage_types() -> tuple[type, ...]:
    """Chat-shaped usage types that accumulate into an OpenAI ``CompletionUsage``."""
    from openai.types import CompletionUsage as _OpenAIUsage
    from openai.types.responses import ResponseUsage as _ResponseUsage

    try:
        from groq.types import CompletionUsage as _GroqUsage
    except ImportError:
        return (_OpenAIUsage, _ResponseUsage)
    return (_OpenAIUsage, _ResponseUsage, _GroqUsage)


def has_compatible_usage(response: object, total_usage: object) -> bool:
    """Return whether a response exposes usage supported by the accumulator."""
    from openai.types import CompletionUsage as _OpenAIUsage

    response_usage = getattr(response, "usage", None)
    # Responses API and Groq usage are compatible with the Chat-shaped running total.
    if isinstance(response_usage, _openai_usage_types()) and isinstance(
        total_usage, _OpenAIUsage
    ):
        return True

    try:
        from instructor.v2.providers.anthropic.usage import is_compatible_usage

        return is_compatible_usage(response_usage, total_usage)
    except ImportError:
        return False


def update_total_usage(
    response: T_Response | None,
    total_usage: OpenAIUsage | AnthropicUsage,
) -> T_Response | None:
    if response is None:
        return None

    from openai.types import CompletionUsage as _OpenAIUsage
    from openai.types.responses import ResponseUsage as _ResponseUsage

    response_usage = getattr(response, "usage", None)
    # Project Responses API usage onto the Chat-shaped total, then echo totals back.
    if isinstance(response_usage, _ResponseUsage) and isinstance(
        total_usage, _OpenAIUsage
    ):
        _accumulate_models(_response_usage_as_completion(response_usage), total_usage)
        response_usage.input_tokens = total_usage.prompt_tokens
        response_usage.output_tokens = total_usage.completion_tokens
        response_usage.total_tokens = total_usage.total_tokens
        if response_usage.input_tokens_details is not None:
            details = total_usage.prompt_tokens_details
            if details is not None:
                response_usage.input_tokens_details.cached_tokens = (
                    details.cached_tokens
                )
        if response_usage.output_tokens_details is not None:
            details = total_usage.completion_tokens_details
            if details is not None:
                response_usage.output_tokens_details.reasoning_tokens = (
                    details.reasoning_tokens
                )
        return response

    if isinstance(response_usage, _openai_usage_types()) and isinstance(
        total_usage, _OpenAIUsage
    ):
        _accumulate_models(cast(BaseModel, response_usage), total_usage)
        return response

    try:
        from instructor.v2.providers.anthropic.usage import (
            update_total_usage as update_anthropic_total_usage,
        )

        if update_anthropic_total_usage(response_usage, total_usage):
            return response
    except ImportError:
        pass

    logger.debug("No compatible response.usage found, token usage not updated.")
    return response
