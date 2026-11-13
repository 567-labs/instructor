"""Anthropic-specific usage helpers."""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel

from instructor.v2.core.usage import _accumulate_models, _all_field_names


def initialize_usage() -> Any:
    """Create an empty Anthropic usage accumulator."""
    from anthropic.types import Usage as AnthropicUsage

    return AnthropicUsage(
        input_tokens=0,
        output_tokens=0,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )


def prepare_usage_accumulator(response_usage: Any, total_usage: Any) -> Any:
    """Use the native beta schema so serialization keeps beta-only details."""
    from anthropic.types import Usage
    from anthropic.types.beta import BetaUsage

    if (
        isinstance(response_usage, BetaUsage)
        and isinstance(total_usage, Usage)
        and total_usage.model_dump() == initialize_usage().model_dump()
    ):
        return BetaUsage.model_validate(
            total_usage.model_dump(mode="python", round_trip=True)
        )
    return total_usage


def has_compatible_usage(response_usage: Any, total_usage: Any) -> bool:
    """Recognize native totals only when top-level counters can budget retries."""
    from anthropic.types import Usage
    from anthropic.types.beta import BetaUsage

    if isinstance(response_usage, BetaUsage) and isinstance(total_usage, BetaUsage):
        # Compaction and future iteration kinds can contain billed tokens absent
        # from top-level usage. Do not silently underestimate their retry cost.
        return not getattr(response_usage, "iterations", None)
    return isinstance(response_usage, Usage) and isinstance(total_usage, Usage)


def update_total_usage(response_usage: Any, total_usage: Any) -> bool:
    """Accumulate Anthropic token usage into a running total when applicable."""
    from anthropic.types import Usage as AnthropicUsage

    from anthropic.types.beta import BetaUsage

    beta = isinstance(response_usage, BetaUsage) and isinstance(total_usage, BetaUsage)
    stable = isinstance(response_usage, AnthropicUsage) and isinstance(
        total_usage, AnthropicUsage
    )
    if not (beta or stable):
        return False
    if beta and getattr(response_usage, "iterations", None):
        # Keep unsupported iterative responses untouched, including their native
        # metadata. The retry loop keeps usage_complete false for this call.
        return False

    opaque = frozenset()
    if beta:
        additive = {
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "cache_creation",
            "server_tool_use",
            "output_tokens_details",
        }
        opaque = frozenset(_all_field_names(response_usage, total_usage) - additive)

    # Older Anthropic SDKs retain this newer documented field as an extra dict,
    # rather than a nested model. Only its known token counter is additive;
    # unknown metadata must keep the existing generic accumulator behavior.
    previous_details = getattr(total_usage, "output_tokens_details", None)
    current_details = getattr(response_usage, "output_tokens_details", None)
    thinking_total = None
    if isinstance(previous_details, dict) and isinstance(current_details, dict):
        previous_tokens = previous_details.get("thinking_tokens")
        current_tokens = current_details.get("thinking_tokens")
        if type(previous_tokens) is int and type(current_tokens) is int:
            thinking_total = previous_tokens + current_tokens

    _accumulate_models(response_usage, total_usage, exclude_fields=opaque)
    for field in opaque:
        # Declared metadata uses the native schema's attribute namespace.
        if field in type(response_usage).model_fields:
            setattr(total_usage, field, copy.deepcopy(getattr(response_usage, field)))
    if beta:
        # SDK extras can have underscore or method-colliding names. Attributes
        # do not faithfully expose those keys; preserve the actual extra map.
        total_usage.__pydantic_extra__ = copy.deepcopy(response_usage.model_extra or {})
    if thinking_total is not None:
        for usage in (total_usage, response_usage):
            usage.output_tokens_details = {
                **usage.output_tokens_details,
                "thinking_tokens": thinking_total,
            }
    if beta:
        # Old SDKs retain newer details as dictionaries; generic accumulation
        # may alias those on the first response. Keep beta totals isolated too.
        for field in additive:
            value = getattr(total_usage, field, None)
            if isinstance(value, (BaseModel, dict, list)):
                setattr(total_usage, field, copy.deepcopy(value))
    return True
