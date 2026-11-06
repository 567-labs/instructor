"""Anthropic-specific usage helpers."""

from __future__ import annotations

from typing import Any

from instructor.v2.core.usage import _accumulate_models


def initialize_usage() -> Any:
    """Create an empty Anthropic usage accumulator."""
    from anthropic.types import Usage as AnthropicUsage

    return AnthropicUsage(
        input_tokens=0,
        output_tokens=0,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )


def update_total_usage(response_usage: Any, total_usage: Any) -> bool:
    """Accumulate Anthropic token usage into a running total when applicable."""
    from anthropic.types import Usage as AnthropicUsage

    if not isinstance(response_usage, AnthropicUsage) or not isinstance(
        total_usage, AnthropicUsage
    ):
        return False

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

    _accumulate_models(response_usage, total_usage)
    if thinking_total is not None:
        for usage in (total_usage, response_usage):
            usage.output_tokens_details = {
                **usage.output_tokens_details,
                "thinking_tokens": thinking_total,
            }
    return True
