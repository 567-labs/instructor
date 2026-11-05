"""Provider isolation for the batch provider registry.

A missing optional SDK must disable only its own provider.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

import pytest

import instructor.batch.providers as batch_providers

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def restore_registry():
    yield
    importlib.reload(batch_providers)


def reload_without(monkeypatch: pytest.MonkeyPatch, missing: str) -> Any:
    """Reload the registry as if ``missing`` were not installed"""
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == missing:
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    return importlib.reload(batch_providers)


@pytest.mark.parametrize(
    "missing, disabled, still_available",
    [
        ("mistralai", "MistralProvider", ("OpenAIProvider", "AnthropicProvider")),
        ("anthropic", "AnthropicProvider", ("OpenAIProvider", "MistralProvider")),
        ("openai", "OpenAIProvider", ("AnthropicProvider", "MistralProvider")),
    ],
)
def test_missing_sdk_disables_only_its_own_provider(
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
    disabled: str,
    still_available: tuple[str, ...],
) -> None:
    module = reload_without(monkeypatch, missing)

    assert getattr(module, disabled) is None
    for name in still_available:
        assert getattr(module, name) is not None


@pytest.mark.parametrize(
    "missing, unavailable, message",
    [
        ("mistralai", "mistral", "Mistral is not installed"),
        ("anthropic", "anthropic", "Anthropic is not installed"),
        ("openai", "openai", "OpenAI is not installed"),
    ],
)
def test_get_provider_reports_only_the_missing_sdk(
    monkeypatch: pytest.MonkeyPatch, missing: str, unavailable: str, message: str
) -> None:
    module = reload_without(monkeypatch, missing)

    with pytest.raises(ValueError, match=message):
        module.get_provider(unavailable)

    for provider_name in {"openai", "anthropic", "mistral"} - {unavailable}:
        assert isinstance(module.get_provider(provider_name), module.BatchProvider)


def test_unsupported_provider_name_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported provider: gemini"):
        batch_providers.get_provider("gemini")


def test_mistral_provider_is_exported() -> None:
    assert "MistralProvider" in batch_providers.__all__
    assert batch_providers.MistralProvider is not None
