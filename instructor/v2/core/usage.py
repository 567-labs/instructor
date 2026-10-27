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


def _field_names(model: BaseModel) -> set[str]:
    """Return declared and Pydantic extra field names for a model."""
    return set(type(model).model_fields) | set(model.model_extra or {})


def _all_field_names(*models: BaseModel) -> set[str]:
    return {field_name for model in models for field_name in _field_names(model)}


def _is_numeric(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _zero_numeric_fields(model: BaseModel) -> BaseModel:
    for field_name in _field_names(model):
        value = getattr(model, field_name, None)
        if isinstance(value, BaseModel):
            _zero_numeric_fields(value)
        elif _is_numeric(value):
            setattr(model, field_name, 0)
    return model


def _accumulate_models(response: BaseModel, total: BaseModel) -> None:
    for field_name in _all_field_names(response, total):
        response_value = getattr(response, field_name, None)
        total_value = getattr(total, field_name, None)
        if isinstance(response_value, BaseModel):
            if not isinstance(total_value, BaseModel):
                total_value = _zero_numeric_fields(response_value.model_copy(deep=True))
                setattr(total, field_name, total_value)
            _accumulate_models(response_value, total_value)
            setattr(response, field_name, total_value.model_copy(deep=True))
        elif isinstance(total_value, BaseModel):
            setattr(response, field_name, total_value.model_copy(deep=True))
        elif _is_numeric(response_value):
            response_number = cast(Real, response_value)
            total_number = cast(Real, total_value) if _is_numeric(total_value) else 0
            value = total_number + response_number
            setattr(total, field_name, value)
            setattr(response, field_name, value)
        elif _is_numeric(total_value):
            setattr(response, field_name, total_value)
        elif response_value is not None:
            setattr(total, field_name, response_value)
            setattr(response, field_name, response_value)
        elif total_value is not None:
            setattr(response, field_name, total_value)


def has_compatible_usage(response: object, total_usage: object) -> bool:
    """Return whether a response exposes usage supported by the accumulator."""
    response_usage = getattr(response, "usage", None)

    try:
        from openai.types import CompletionUsage as _OpenAIUsage

        if isinstance(response_usage, _OpenAIUsage) and isinstance(
            total_usage, _OpenAIUsage
        ):
            return True
    except ImportError:
        pass

    try:
        from anthropic.types import Usage as _AnthropicUsage

        return isinstance(response_usage, _AnthropicUsage) and isinstance(
            total_usage, _AnthropicUsage
        )
    except ImportError:
        return False


def update_total_usage(
    response: T_Response | None,
    total_usage: OpenAIUsage | AnthropicUsage,
) -> T_Response | None:
    if response is None:
        return None

    response_usage = getattr(response, "usage", None)

    try:
        from openai.types import CompletionUsage as _OpenAIUsage

        if isinstance(response_usage, _OpenAIUsage) and isinstance(
            total_usage, _OpenAIUsage
        ):
            _accumulate_models(response_usage, total_usage)
            return response
    except ImportError:
        pass

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
