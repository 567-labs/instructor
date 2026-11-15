"""GenAI request config merging and context-cache handling.

Covers #1966 (thinking_config inside config), #1953 (automatic_function_calling),
#1759 (labels from a dict config), #1964 (unions) and explicit context caching,
where the cache owns system_instruction, tools and tool_config.
"""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from instructor.mode import Mode
from instructor.utils.providers import Provider
from instructor.v2.core.registry import mode_registry

types = pytest.importorskip("google.genai.types")

from instructor.v2.providers.gemini import utils  # noqa: E402

LABELS = {"tenant": "acme", "cost-center": "123"}
MESSAGES = [
    {"role": "system", "content": "Use the cached instructions."},
    {"role": "user", "content": "extract a value"},
]


class Answer(BaseModel):
    value: int


def user_config(source: str, **fields: Any) -> Any:
    """Build the same request config as a dict, an SDK object or a bare object."""
    if source == "dict":
        return fields
    if source == "sdk":
        return types.GenerateContentConfig(**fields)
    # A bare object only has the attributes it was given.
    return SimpleNamespace(**{k: v for k, v in fields.items() if v is not None})


@pytest.mark.parametrize("source", ["dict", "sdk"])
def test_update_genai_kwargs_merges_config_fields(source: str) -> None:
    thinking = types.ThinkingConfig(thinking_budget=2048)
    calling = types.AutomaticFunctionCallingConfig(disable=True)
    config = user_config(
        source,
        thinking_config=thinking,
        automatic_function_calling=calling,
        labels={"key": "value"},
        cached_content="caches/abc123",
    )

    result = utils.update_genai_kwargs({"config": config}, {})

    assert result["thinking_config"] == thinking
    assert result["automatic_function_calling"] == calling
    assert result["labels"] == {"key": "value"}
    assert result["cached_content"] == "caches/abc123"


@pytest.mark.parametrize("source", ["dict", "sdk"])
def test_update_genai_kwargs_explicit_values_win_over_config(source: str) -> None:
    config = user_config(
        source,
        thinking_config=types.ThinkingConfig(thinking_budget=1024),
        labels={"config_key": "config_value"},
        cached_content="caches/from_config",
    )
    explicit = types.ThinkingConfig(thinking_budget=2048)
    base = {
        "labels": {"base_key": "base_value"},
        "cached_content": "caches/from_base",
        "system_instruction": "dropped",
        "tools": ["dropped"],
        "tool_config": "dropped",
    }

    result = utils.update_genai_kwargs(
        {"config": config, "thinking_config": explicit}, base
    )

    assert result["thinking_config"] is explicit
    assert result["labels"] == {"base_key": "base_value"}
    assert result["cached_content"] == "caches/from_base"
    assert {"system_instruction", "tools", "tool_config"}.isdisjoint(result)


def test_update_genai_kwargs_reads_top_level_options_and_partial_config() -> None:
    result = utils.update_genai_kwargs(
        {
            "generation_config": {"max_tokens": 100, "temperature": 0.7},
            "config": SimpleNamespace(automatic_function_calling=True),
            "cached_content": "caches/top-level",
        },
        {},
    )

    assert result["max_output_tokens"] == 100
    assert result["temperature"] == 0.7
    assert result["automatic_function_calling"] is True
    assert result["cached_content"] == "caches/top-level"
    assert "thinking_config" not in result


@pytest.mark.parametrize(
    "schema",
    [
        {"properties": {"name": {"type": "string"}}},
        {"properties": {"v": {"anyOf": [{"type": "string"}, {"type": "null"}]}}},
        {"properties": {"v": {"anyOf": [{"type": "integer"}, {"type": "string"}]}}},
        {
            "properties": {
                "v": {
                    "anyOf": [
                        {"type": "integer"},
                        {"type": "string"},
                        {"type": "boolean"},
                    ]
                }
            }
        },
    ],
    ids=["plain", "optional", "union", "three-way-union"],
)
def test_verify_no_unions_accepts_all_unions(schema: dict[str, Any]) -> None:
    assert utils.verify_no_unions(schema) is True


@pytest.mark.parametrize(
    "prepare",
    [utils.handle_genai_structured_outputs, utils.handle_genai_tools],
    ids=["structured", "tools"],
)
@pytest.mark.parametrize("source", ["dict", "sdk", "object"])
@pytest.mark.parametrize("cached", [False, True], ids=["uncached", "cached"])
def test_genai_helpers_merge_user_config(
    prepare: Any, source: str, cached: bool
) -> None:
    cache = "cachedContents/session-1" if cached else None
    config = user_config(
        source,
        cached_content=cache,
        thinking_config=None
        if cached and source == "object"
        else types.ThinkingConfig(thinking_budget=24),
        labels=LABELS,
    )

    model, request = prepare(Answer, {"messages": deepcopy(MESSAGES), "config": config})
    prepared = request["config"]

    assert model is Answer
    assert prepared.cached_content == cache
    assert prepared.labels == LABELS
    if cached and source == "object":
        assert prepared.thinking_config is None
    else:
        assert prepared.thinking_config.thinking_budget == 24
    assert request["contents"][0].parts[0].text == "extract a value"
    # The cache owns the system instruction, tools and tool config.
    assert (prepared.system_instruction is None) is cached
    if prepare is utils.handle_genai_tools:
        assert (prepared.tools is None) is cached
        assert (prepared.tool_config is None) is cached


@pytest.mark.parametrize("mode", [Mode.TOOLS, Mode.JSON])
@pytest.mark.parametrize("source", ["dict", "sdk", "top-level"])
@pytest.mark.parametrize("response_model", [Answer, None])
def test_cached_content_removes_conflicting_fields(
    mode: Mode, source: str, response_model: type[BaseModel] | None
) -> None:
    kwargs: dict[str, Any] = {"messages": deepcopy(MESSAGES)}
    cache = "cachedContents/release-test"
    if source == "top-level":
        kwargs["cached_content"] = cache
    else:
        kwargs["config"] = user_config(source, cached_content=cache)
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
