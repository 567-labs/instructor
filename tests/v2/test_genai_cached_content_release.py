from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from pydantic import BaseModel

from instructor.mode import Mode
from instructor.utils.providers import Provider
from instructor.v2.core.registry import mode_registry

types = pytest.importorskip("google.genai.types")


class Answer(BaseModel):
    value: int


@pytest.mark.parametrize("mode", [Mode.TOOLS, Mode.JSON])
@pytest.mark.parametrize("source", ["dict", "sdk", "top-level"])
@pytest.mark.parametrize("response_model", [Answer, None])
def test_cached_content_removes_conflicting_fields(
    mode: Mode, source: str, response_model: type[BaseModel] | None
) -> None:
    kwargs: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": "Use the cached instructions."},
            {"role": "user", "content": "Answer 42."},
        ],
    }
    cache = "cachedContents/release-test"
    if source == "dict":
        kwargs["config"] = {"cached_content": cache}
    elif source == "sdk":
        kwargs["config"] = types.GenerateContentConfig(cached_content=cache)
    else:
        kwargs["cached_content"] = cache
    original = deepcopy(kwargs)
    prepare = mode_registry.get_handlers(Provider.GENAI, mode).request_handler
    for _ in range(2):
        _, prepared = prepare(response_model, kwargs)
        config = prepared["config"]
        assert config.cached_content == cache
        assert config.system_instruction is None
        assert config.tools is None
        assert config.tool_config is None
        assert "cached_content" not in prepared
        if mode == Mode.JSON and response_model is not None:
            assert config.response_mime_type == "application/json"
            assert config.response_schema is not None
        assert kwargs == original


@pytest.mark.parametrize("mode", [Mode.TOOLS, Mode.JSON])
@pytest.mark.parametrize("response_model", [Answer, None])
def test_config_cached_content_takes_precedence(
    mode: Mode, response_model: type[BaseModel] | None
) -> None:
    kwargs = {
        "messages": [{"role": "user", "content": "Answer 42."}],
        "config": {"cached_content": "cachedContents/config"},
        "cached_content": "cachedContents/top-level",
    }
    original = deepcopy(kwargs)
    prepare = mode_registry.get_handlers(Provider.GENAI, mode).request_handler
    _, prepared = prepare(response_model, kwargs)
    assert prepared["config"].cached_content == "cachedContents/config"
    assert "cached_content" not in prepared
    assert kwargs == original
