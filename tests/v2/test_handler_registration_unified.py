"""Registry lookups and OpenAI-compatible handler reuse.

Per-mode registration, handler completeness, and PARALLEL/RESPONSES support
checks live in test_client_unified.py.
"""

from __future__ import annotations

import pytest

from instructor import Mode, Provider
from instructor.v2.core.registry import mode_registry
from tests.v2.test_handlers_parametrized import (
    PROVIDER_HANDLER_MODES,
    _ensure_handlers_loaded,
)


@pytest.mark.parametrize(
    "provider",
    [pytest.param(p, id=p.value) for p in sorted(PROVIDER_HANDLER_MODES, key=str)],
)
def test_provider_in_mode_providers(provider: Provider) -> None:
    """Each provider is listed for every mode it supports."""
    _ensure_handlers_loaded(provider)
    for mode in PROVIDER_HANDLER_MODES[provider]:
        assert provider in mode_registry.get_providers_for_mode(mode), (
            f"{provider.value} should be in providers for {mode.value}"
        )


@pytest.mark.parametrize("mode", [Mode.TOOLS, Mode.MD_JSON])
@pytest.mark.parametrize(
    "provider",
    [Provider.GROQ, Provider.FIREWORKS, Provider.CEREBRAS],
    ids=lambda p: p.value,
)
def test_openai_compatible_providers_reuse_openai_handlers(
    provider: Provider, mode: Mode
) -> None:
    assert mode in PROVIDER_HANDLER_MODES[provider]
    assert mode_registry.is_registered(provider, mode)
    handlers = mode_registry.get_handlers(provider, mode)
    openai_handlers = mode_registry.get_handlers(Provider.OPENAI, mode)
    assert handlers.request_handler == openai_handlers.request_handler
    assert handlers.response_parser == openai_handlers.response_parser
