"""Gemini-specific message templating helpers."""

from __future__ import annotations

from typing import Any, Callable


def process_message(
    message: dict[str, Any],
    context: dict[str, Any],
    apply_template: Callable[[str, dict[str, Any]], str],
) -> dict[str, Any]:
    """Apply templates to Gemini message content."""
    if isinstance(message.get("parts"), list):
        message["parts"] = [
            apply_template(part, context) if isinstance(part, str) else part
            for part in message["parts"]
        ]
    return message
