"""Bedrock Converse message templating helpers.

The Bedrock request handler rewrites messages into Converse shape before templating runs: the
system prompt moves to a top-level ``system=[{"text": ...}]`` list and every content part becomes
a block such as ``{"text": ...}``. These helpers template those shapes; blocks without a string
``text`` (images, documents, cache points) pass through untouched.
"""

from __future__ import annotations

from typing import Any, Callable

ApplyTemplate = Callable[[str, dict[str, Any]], str]


def template_block(
    block: Any, context: dict[str, Any], apply_template: ApplyTemplate
) -> Any:
    """Template one Converse content block; anything without a string ``text`` passes through."""
    if isinstance(block, dict) and isinstance(block.get("text"), str):
        templated = block.copy()
        templated["text"] = apply_template(block["text"], context)
        return templated
    return block


def process_message(
    message: dict[str, Any],
    context: dict[str, Any],
    apply_template: ApplyTemplate,
) -> dict[str, Any]:
    """Apply templates to a Bedrock message, whether or not it was already converted to Converse shape."""
    content = message.get("content")
    if isinstance(content, str):
        message["content"] = apply_template(content, context)
    elif isinstance(content, list):
        message["content"] = [
            template_block(block, context, apply_template) for block in content
        ]
    return message


def process_system(
    system: list[Any],
    context: dict[str, Any],
    apply_template: ApplyTemplate,
) -> list[Any]:
    """Apply templates to the Converse ``system`` list."""
    return [template_block(block, context, apply_template) for block in system]
