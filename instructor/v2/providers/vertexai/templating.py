"""VertexAI-specific message templating helpers."""

from __future__ import annotations

from typing import Any, Callable


def process_message(
    message: Any,
    context: dict[str, Any],
    apply_template: Callable[[str, dict[str, Any]], str],
) -> Any:
    """Apply templates to VertexAI Content parts."""
    import vertexai.generative_models as gm

    return gm.Content(
        role=message.role,
        parts=[
            (
                gm.Part.from_text(apply_template(part.text, context))
                if isinstance(getattr(part, "text", None), str)
                else part
            )
            for part in message.parts
        ],
    )
