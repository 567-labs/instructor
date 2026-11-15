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


def is_compatible_usage(response_usage: Any, total_usage: Any) -> bool:
    """Return whether standard or beta Anthropic usage fits the running total."""
    from anthropic.types import Usage as AnthropicUsage
    from anthropic.types.beta import BetaUsage

    return isinstance(response_usage, (AnthropicUsage, BetaUsage)) and isinstance(
        total_usage, AnthropicUsage
    )


def update_total_usage(response_usage: Any, total_usage: Any) -> bool:
    """Accumulate Anthropic token usage into a running total when applicable."""
    if not is_compatible_usage(response_usage, total_usage):
        return False

    _accumulate_models(response_usage, total_usage)
    return True
