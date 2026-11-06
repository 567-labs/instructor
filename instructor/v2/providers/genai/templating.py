"""Google GenAI-specific message templating helpers."""

from __future__ import annotations

from typing import Any, Callable


def process_message(
    message: Any,
    context: dict[str, Any],
    apply_template: Callable[[str, dict[str, Any]], str],
) -> Any:
    """Apply templates to GenAI Content parts."""
    from google.genai import types

    return types.Content(
        role=message.role,
        parts=[
            (
                part.model_copy(update={"text": apply_template(part.text, context)})
                if isinstance(getattr(part, "text", None), str)
                else part
            )
            for part in message.parts
        ],
    )
