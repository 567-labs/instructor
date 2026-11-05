"""Anthropic-specific message templating helpers."""

from __future__ import annotations

from typing import Any, Callable

from instructor.v2.core.providers import Provider
from instructor.v2.core.templating import render_text_blocks


def process_message(
    message: dict[str, Any],
    context: dict[str, Any],
    apply_template: Callable[[str, dict[str, Any]], str],
) -> dict[str, Any]:
    """Apply templates to Anthropic message content."""
    if isinstance(message.get("content"), list):
        render_text_blocks(
            message["content"], context, apply_template, Provider.ANTHROPIC
        )
    return message
