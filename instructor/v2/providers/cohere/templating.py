"""Cohere-specific message templating helpers."""

from __future__ import annotations

from typing import Any, Callable


def process_message(
    message: dict[str, Any],
    context: dict[str, Any],
    apply_template: Callable[[str, dict[str, Any]], str],
) -> dict[str, Any]:
    """Apply templates to Cohere message content."""
    if isinstance(message.get("message"), str):
        message["message"] = apply_template(message["message"], context)
    return message
