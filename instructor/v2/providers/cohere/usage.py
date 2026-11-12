"""Native Cohere V2 token accounting, separate from billing metadata."""

from __future__ import annotations

import math
from importlib import import_module
from typing import cast

from pydantic import BaseModel


def _usage_types() -> tuple[type[BaseModel], type[BaseModel]]:
    # ClientV2 first appeared in SDK 5.8; its native models lived under v2.
    try:
        module = import_module("cohere.types.usage")
        tokens_module = import_module("cohere.types.usage_tokens")
    except ImportError:
        module = import_module("cohere.v2.types.usage")
        tokens_module = import_module("cohere.v2.types.usage_tokens")
    return cast(type[BaseModel], module.Usage), cast(
        type[BaseModel], tokens_module.UsageTokens
    )


def initialize_usage() -> BaseModel:
    usage_type, tokens_type = _usage_types()
    return usage_type.model_validate(
        {"tokens": tokens_type.model_validate({"input_tokens": 0, "output_tokens": 0})}
    )


def token_counts(usage: object) -> tuple[int, int] | None:
    """Read explicitly present model-token counters, never billed units."""
    if not type(usage).__module__.startswith("cohere."):
        return None
    Usage, UsageTokens = _usage_types()

    if not isinstance(usage, Usage) or "tokens" not in usage.model_fields_set:
        return None
    tokens = getattr(usage, "tokens", None)
    if not isinstance(tokens, UsageTokens):
        return None
    values: list[int] = []
    for name in ("input_tokens", "output_tokens"):
        if name not in tokens.model_fields_set:
            return None
        value = getattr(tokens, name)
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            if value < 0:
                return None
        elif isinstance(value, float):
            if not math.isfinite(value) or value < 0 or not value.is_integer():
                return None
        else:
            return None
        values.append(int(value))
    return values[0], values[1]


def accumulate_usage(response: object, total: object) -> BaseModel | None:
    """Return a fresh tokens-only native total, or fail closed without mutation.

    SDK 5.8 native models are frozen. Returning a new object also leaves every
    raw response and previous accumulator snapshot intact on newer SDKs.
    """
    counts = token_counts(getattr(response, "usage", None))
    previous = token_counts(total)
    if counts is None or previous is None:
        return None
    sums = (previous[0] + counts[0], previous[1] + counts[1])
    # Native SDK fields are floats. Refuse overflow or rounding instead of
    # silently turning an exact integer counter into a different budget value.
    for value in sums:
        try:
            native = float(value)
        except OverflowError:
            return None
        if not math.isfinite(native) or int(native) != value:
            return None
    usage_type, tokens_type = _usage_types()
    result = usage_type.model_validate(
        {
            "tokens": tokens_type.model_validate(
                {"input_tokens": sums[0], "output_tokens": sums[1]}
            )
        }
    )
    return result if token_counts(result) == sums else None
