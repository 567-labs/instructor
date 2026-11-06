"""OpenAI-shaped message templating helpers."""

from __future__ import annotations

from typing import Any, Callable


def process_message(
    message: dict[str, Any],
    context: dict[str, Any],
    apply_template: Callable[[str, dict[str, Any]], str],
) -> dict[str, Any]:
    """Apply templates to OpenAI-style message content."""
    if isinstance(message.get("content"), str):
        message["content"] = apply_template(message["content"], context)
    return message
