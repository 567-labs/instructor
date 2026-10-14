"""Unit tests for Anthropic v2 handlers.

These tests verify handler behavior without requiring API keys, by calling
the request-preparation logic directly and inspecting the resulting kwargs.
"""

from __future__ import annotations

from collections.abc import Iterable

import pytest
from pydantic import BaseModel

from instructor import Mode, Provider
from instructor.v2.core.registry import mode_registry


class Answer(BaseModel):
    """Simple answer model for testing."""

    answer: float


@pytest.fixture
def handler():
    """Get the Anthropic tools handler from registry."""
    return mode_registry.get_handlers(Provider.ANTHROPIC, Mode.TOOLS)


class TestAnthropicToolsHandlerToolChoice:
    """Regression tests for #2477: a forced single-tool tool_choice did not
    set disable_parallel_tool_use, so Anthropic could still emit multiple
    tool_use blocks for the same forced tool, which the single-model parser
    then rejected even though the model's response was otherwise valid."""

    def test_single_response_model_disables_parallel_tool_use(self, handler):
        kwargs = {
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 100,
        }

        _, result_kwargs = handler.request_handler(Answer, kwargs)

        assert result_kwargs["tool_choice"] == {
            "type": "tool",
            "name": "Answer",
            "disable_parallel_tool_use": True,
        }

    def test_parallel_tools_are_not_forced_to_disable_parallel_use(self, handler):
        # Iterable[T] is the parallel-tools case: the model is *expected* to
        # emit multiple tool_use blocks here, so tool_choice must stay "auto"
        # and must not gain disable_parallel_tool_use.
        kwargs = {
            "messages": [{"role": "user", "content": "List some answers"}],
            "max_tokens": 100,
        }

        _, result_kwargs = handler.request_handler(Iterable[Answer], kwargs)

        assert result_kwargs["tool_choice"] == {"type": "auto"}

    def test_explicit_tool_choice_is_not_overridden(self, handler):
        kwargs = {
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 100,
            "tool_choice": {"type": "auto"},
        }

        _, result_kwargs = handler.request_handler(Answer, kwargs)

        assert result_kwargs["tool_choice"] == {"type": "auto"}


class TestAnthropicToolsHandlerAdaptiveAndModelToolChoice:
    """Regression tests for #2792: adaptive thinking and Opus/Sonnet/Fable 5.x
    must not receive a forced tool_choice (API returns 400)."""

    def test_adaptive_thinking_uses_auto_tool_choice(self, handler):
        kwargs = {
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 1000,
            "model": "claude-sonnet-4-6",
            "thinking": {"type": "adaptive"},
        }

        _, result_kwargs = handler.request_handler(Answer, kwargs)

        assert result_kwargs["tool_choice"] == {"type": "auto"}
        assert result_kwargs["system"] == [
            {
                "type": "text",
                "text": "Return only the tool call and no additional text.",
            }
        ]

    def test_enabled_thinking_still_uses_auto_tool_choice(self, handler):
        kwargs = {
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 1000,
            "model": "claude-sonnet-4-5",
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        }

        _, result_kwargs = handler.request_handler(Answer, kwargs)

        assert result_kwargs["tool_choice"] == {"type": "auto"}

    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-5-5",
            "claude-sonnet-5-5",
            "claude-fable-5-1",
            "anthropic.claude-opus-5-5-20261001",
        ],
    )
    def test_models_without_forced_tool_choice_use_auto(self, handler, model):
        kwargs = {
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 1000,
            "model": model,
        }

        _, result_kwargs = handler.request_handler(Answer, kwargs)

        assert result_kwargs["tool_choice"] == {"type": "auto"}
        assert result_kwargs["system"] == [
            {
                "type": "text",
                "text": "Return only the tool call and no additional text.",
            }
        ]

    def test_older_claude_still_forces_single_tool(self, handler):
        kwargs = {
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 100,
            "model": "claude-sonnet-4-5",
        }

        _, result_kwargs = handler.request_handler(Answer, kwargs)

        assert result_kwargs["tool_choice"] == {
            "type": "tool",
            "name": "Answer",
            "disable_parallel_tool_use": True,
        }
