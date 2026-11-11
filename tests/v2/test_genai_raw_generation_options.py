from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from instructor import Mode, Provider
from instructor.processing.response import handle_response_model

types = pytest.importorskip("google.genai.types")


@pytest.mark.parametrize("mode", [Mode.TOOLS, Mode.JSON])
@pytest.mark.parametrize(
    "nested", [False, True], ids=["top-level", "generation-config"]
)
def test_raw_requests_preserve_generation_options(mode: Mode, nested: bool) -> None:
    options = {
        "max_tokens": 16,
        "temperature": 0,
        "top_p": 0,
        "n": 1,
        "stop": ["END"],
        "seed": 0,
        "presence_penalty": 0,
        "frequency_penalty": 0,
    }
    kwargs: dict[str, Any] = {
        "messages": [{"role": "user", "content": "Say hello"}],
        **({"generation_config": options} if nested else options),
    }
    original = deepcopy(kwargs)

    response_model, request = handle_response_model(
        None, mode=mode, provider=Provider.GENAI, **kwargs
    )

    assert response_model is None
    config = request["config"]
    assert isinstance(config, types.GenerateContentConfig)
    assert config.max_output_tokens == 16
    assert config.temperature == 0
    assert config.top_p == 0
    assert config.candidate_count == 1
    assert config.stop_sequences == ["END"]
    assert config.seed == 0
    assert config.presence_penalty == 0
    assert config.frequency_penalty == 0
    assert config.safety_settings is None
    assert config.response_schema is None
    assert config.tools is None
    assert not (set(options) | {"generation_config"}) & request.keys()
    assert kwargs == original


@pytest.mark.parametrize("mode", [Mode.TOOLS, Mode.JSON])
@pytest.mark.parametrize("native_object", [False, True], ids=["dict", "sdk-config"])
def test_raw_requests_merge_options_without_losing_native_config(
    mode: Mode, native_object: bool
) -> None:
    native_config = {
        "temperature": 0.75,
        "max_output_tokens": 128,
        "top_p": 0.9,
        "labels": {"source": "raw-request"},
        "safety_settings": [
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"}
        ],
    }
    kwargs: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Say hello"},
        ],
        "config": (
            types.GenerateContentConfig(**native_config)
            if native_object
            else native_config
        ),
        "generation_config": {"temperature": 0.25, "max_tokens": 32},
        "temperature": 0,
        "max_tokens": 16,
    }
    original = deepcopy(kwargs)

    _, request = handle_response_model(
        None, mode=mode, provider=Provider.GENAI, **kwargs
    )

    config = request["config"]
    assert config.temperature == 0
    assert config.max_output_tokens == 16
    assert config.top_p == 0.9
    assert config.labels == {"source": "raw-request"}
    assert config.system_instruction.strip() == "Be concise."
    assert len(config.safety_settings) == 1
    assert config.safety_settings[0].threshold == "BLOCK_ONLY_HIGH"
    assert kwargs == original


@pytest.mark.parametrize("mode", [Mode.TOOLS, Mode.JSON])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"generation_config": None},
        {
            "generation_config": {"temperature": 0.25, "max_tokens": 32},
            "temperature": None,
            "max_tokens": None,
        },
    ],
    ids=["omitted", "null-config", "null-overrides"],
)
def test_raw_requests_without_options_preserve_provider_defaults(
    mode: Mode, options: dict[str, Any]
) -> None:
    _, request = handle_response_model(
        None,
        mode=mode,
        provider=Provider.GENAI,
        messages=[{"role": "user", "content": "Say hello"}],
        **options,
    )

    assert "config" not in request
