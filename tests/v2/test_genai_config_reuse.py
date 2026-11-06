from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from pydantic import BaseModel

from instructor.mode import Mode
from instructor.utils.providers import Provider
from instructor.v2.core.registry import mode_registry

pytest.importorskip("google.genai")


class Answer(BaseModel):
    value: int


@pytest.mark.parametrize("mode", [Mode.TOOLS, Mode.JSON])
@pytest.mark.parametrize("override_temperature", [False, True])
def test_reusing_generation_config_preserves_caller_options(
    mode: Mode, override_temperature: bool
) -> None:
    generation_config = {
        "max_tokens": 64,
        "temperature": 0.25,
        "top_p": 0.9,
        "seed": 42,
        "stop": ["END"],
    }
    original = deepcopy(generation_config)
    kwargs: dict[str, Any] = {
        "messages": [{"role": "user", "content": "The answer is 42."}],
        "generation_config": generation_config,
    }
    if override_temperature:
        kwargs["temperature"] = 0.75
    prepare = mode_registry.get_handlers(Provider.GENAI, mode).request_handler

    for _ in range(2):
        _, prepared = prepare(Answer, kwargs)
        config = prepared["config"]
        assert config.max_output_tokens == 64
        assert config.temperature == (0.75 if override_temperature else 0.25)
        assert config.top_p == 0.9
        assert config.seed == 42
        assert config.stop_sequences == ["END"]
        assert generation_config == original

    _, prepared = prepare(
        Answer,
        {
            "messages": [{"role": "user", "content": "The answer is 43."}],
            "generation_config": generation_config,
        },
    )
    assert prepared["config"].temperature == 0.25
