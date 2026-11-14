"""GenAI requests with ``response_model=None`` keep generation options."""

from __future__ import annotations

from typing import Any

import pytest

from instructor.v2.core.mode import Mode
from instructor.v2.providers.genai.handlers import (
    GenAIStructuredOutputsHandler,
    GenAIToolsHandler,
)

types = pytest.importorskip("google.genai.types")

HANDLERS = [
    GenAIToolsHandler(mode=Mode.TOOLS),
    GenAIStructuredOutputsHandler(mode=Mode.JSON),
]
MESSAGES = [{"role": "user", "content": "Say hello"}]


@pytest.mark.parametrize("handler", HANDLERS, ids=["tools", "json"])
def test_raw_request_maps_openai_options(handler: Any) -> None:
    _, request = handler.prepare_request(
        None,
        {
            "messages": MESSAGES,
            "max_tokens": 16,
            "temperature": 0,
            "top_p": 0.5,
            "n": 1,
            "stop": ["END"],
            "seed": 3,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.2,
        },
    )

    config = request["config"]
    assert isinstance(config, types.GenerateContentConfig)
    assert config.max_output_tokens == 16
    assert config.temperature == 0
    assert config.top_p == 0.5
    assert config.candidate_count == 1
    assert config.stop_sequences == ["END"]
    assert config.seed == 3
    assert config.presence_penalty == 0.1
    assert config.frequency_penalty == 0.2
    assert config.safety_settings is None
    assert config.tools is None
    assert config.response_schema is None
    for key in ("max_tokens", "temperature", "top_p", "stop", "seed"):
        assert key not in request


@pytest.mark.parametrize("handler", HANDLERS, ids=["tools", "json"])
def test_raw_request_reads_generation_config_and_explicit_config_wins(
    handler: Any,
) -> None:
    _, request = handler.prepare_request(
        None,
        {
            "messages": MESSAGES,
            "generation_config": {"max_tokens": 8, "temperature": 0.3},
            "temperature": 0.9,
            "config": types.GenerateContentConfig(max_output_tokens=32),
        },
    )

    config = request["config"]
    assert config.max_output_tokens == 32
    assert config.temperature == 0.9
    assert "generation_config" not in request


@pytest.mark.parametrize("handler", HANDLERS, ids=["tools", "json"])
def test_raw_request_forwards_thinking_and_safety_settings(handler: Any) -> None:
    safety = [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        )
    ]
    _, request = handler.prepare_request(
        None,
        {
            "messages": MESSAGES,
            "thinking_config": types.ThinkingConfig(thinking_budget=0),
            "safety_settings": safety,
        },
    )

    config = request["config"]
    assert config.thinking_config == types.ThinkingConfig(thinking_budget=0)
    assert config.safety_settings == safety
    assert "thinking_config" not in request
    assert "safety_settings" not in request


@pytest.mark.parametrize("handler", HANDLERS, ids=["tools", "json"])
def test_raw_request_converts_safety_settings_mapping(handler: Any) -> None:
    category = types.HarmCategory.HARM_CATEGORY_HARASSMENT
    threshold = types.HarmBlockThreshold.BLOCK_ONLY_HIGH
    _, request = handler.prepare_request(
        None, {"messages": MESSAGES, "safety_settings": {category: threshold}}
    )

    assert request["config"].safety_settings == [
        types.SafetySetting(category=category, threshold=threshold)
    ]
