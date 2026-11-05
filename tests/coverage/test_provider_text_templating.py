from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.templating import handle_templating


@pytest.mark.parametrize(
    ("provider", "mode", "text_block", "opaque_block"),
    [
        (
            Provider.OPENAI,
            Mode.RESPONSES_TOOLS,
            {"type": "input_text", "text": "{{ name }}"},
            {"type": "input_image", "image_url": "{{ name }}"},
        ),
        (
            Provider.AZURE_OPENAI,
            Mode.RESPONSES_TOOLS,
            {"type": "input_text", "text": "{{ name }}"},
            {"type": "function_call", "arguments": "{{ name }}"},
        ),
        (
            Provider.ANTHROPIC,
            Mode.TOOLS,
            {"type": "text", "text": "{{ name }}"},
            {"type": "tool_result", "text": "{{ name }}"},
        ),
        (
            Provider.BEDROCK,
            Mode.JSON,
            {"text": "{{ name }}"},
            {"document": {"name": "{{ name }}"}},
        ),
    ],
)
def test_templates_render_only_provider_text_blocks(
    provider: Provider,
    mode: Mode,
    text_block: dict[str, Any],
    opaque_block: dict[str, Any],
) -> None:
    kwargs = {"messages": [{"role": "user", "content": [text_block, opaque_block]}]}
    original = deepcopy(kwargs)

    result = handle_templating(kwargs, mode, provider=provider, context={"name": "Ada"})

    assert result["messages"][0]["content"] == [
        {**text_block, "text": "Ada"},
        opaque_block,
    ]
    assert kwargs == original


@pytest.mark.parametrize(
    ("mode", "text_block", "other_block"),
    [
        (
            Mode.ANTHROPIC_TOOLS,
            {
                "type": "text",
                "text": "{{ name }}",
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "image", "text": "{{ name }}"},
        ),
        (
            Mode.BEDROCK_TOOLS,
            {"text": "{{ name }}"},
            {"cachePoint": {"type": "default"}},
        ),
    ],
)
def test_hoisted_system_blocks_render_without_messages(
    mode: Mode, text_block: dict[str, Any], other_block: dict[str, Any]
) -> None:
    kwargs = {"system": [text_block, other_block]}
    original = deepcopy(kwargs)

    result = handle_templating(kwargs, mode, context={"name": "Ada"})

    assert result["system"] == [{**text_block, "text": "Ada"}, other_block]
    assert kwargs == original


def test_bedrock_templates_render_after_request_preparation() -> None:
    from instructor.v2.core.registry import mode_registry

    kwargs = {
        "model": "example-model",
        "messages": [
            {"role": "system", "content": "Classify {{ topic }}"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Report: {{ text }}"},
                    {"cachePoint": {"type": "default"}},
                ],
            },
        ],
    }
    original = deepcopy(kwargs)
    handlers = mode_registry.get_handlers(Provider.BEDROCK, Mode.TOOLS)
    _, prepared = handlers.request_handler(response_model=None, kwargs=kwargs)
    result = handle_templating(
        prepared, Mode.BEDROCK_TOOLS, context={"topic": "airports", "text": "snow"}
    )

    assert result["system"] == [{"text": "Classify airports"}]
    assert result["messages"][0]["content"] == [
        {"text": "Report: snow"},
        {"cachePoint": {"type": "default"}},
    ]
    assert kwargs == original
