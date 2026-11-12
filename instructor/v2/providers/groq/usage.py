"""Explicit native Groq token accounting; other fields describe the latest call."""

from __future__ import annotations

import copy
from typing import Any, cast

from pydantic import BaseModel

_TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def _usage_type() -> type[BaseModel]:
    try:
        from groq.types.completion_usage import CompletionUsage

        return CompletionUsage
    except ImportError:
        # Groq 0.4.2 exposes usage under the chat response module.
        from groq.types.chat import chat_completion

        return cast(type[BaseModel], cast(Any, chat_completion).Usage)


def initialize_usage() -> BaseModel:
    return _usage_type()(**dict.fromkeys(_TOKEN_FIELDS, 0))


def is_native_usage(value: object) -> bool:
    # Do not import an optional SDK when dispatching another provider's usage.
    if not type(value).__module__.startswith("groq."):
        return False
    return isinstance(value, _usage_type())


def has_compatible_usage(response: Any, total: Any) -> bool:
    usage = getattr(response, "usage", None)
    if not is_native_usage(usage) or not is_native_usage(total):
        return False
    if getattr(response, "usage_breakdown", None) is not None:
        return False
    values = [getattr(usage, name, None) for name in _TOKEN_FIELDS]
    return all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in values
    ) and values[2] == cast(int, values[0]) + cast(int, values[1])


def update_total_usage(response: Any, total: BaseModel) -> None:
    if not has_compatible_usage(response, total):
        return
    usage = response.usage
    counts = {
        name: getattr(total, name) + getattr(usage, name) for name in _TOKEN_FIELDS
    }
    # Fields other than P/C/T are latest-call metadata, including timing seconds
    # and cached/reasoning subsets. Copy extras through their mapping so names
    # such as "_trace" or "schema" are not redirected to Python attributes.
    for name in type(usage).model_fields:
        setattr(total, name, copy.deepcopy(getattr(usage, name)))
    total.__pydantic_extra__ = copy.deepcopy(usage.model_extra)
    for name, value in counts.items():
        setattr(total, name, value)
        setattr(usage, name, value)
