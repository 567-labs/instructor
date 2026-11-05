"""Unified parametrized tests for handler registration across all providers.

These tests verify handler registration, inheritance, and common patterns
without requiring API keys.
"""

from __future__ import annotations

import pytest

from instructor import Mode, Provider
from instructor.v2.core.registry import mode_registry

# Import handler loading utilities from existing test
from tests.v2.conftest import get_registered_provider_mode_pairs
from tests.v2.test_handlers_parametrized import (
    PROVIDER_HANDLER_MODES,
    _ensure_handlers_loaded,
)


def _get_provider_mode_params():
    """Generate (provider, mode) parameters for all registered modes."""
    pairs = get_registered_provider_mode_pairs()
    return [
        pytest.param(provider, mode, id=f"{provider.value}-{mode.value}")
        for provider, mode in pairs
    ]


def _get_provider_params():
    """Generate provider parameters."""
    providers = {provider for provider, _ in get_registered_provider_mode_pairs()}
    # Sort by value for deterministic ordering across pytest-xdist workers
    sorted_providers = sorted(providers, key=lambda p: p.value)
    return [pytest.param(provider, id=provider.value) for provider in sorted_providers]


# ============================================================================
# Handler Registration Tests
# ============================================================================


@pytest.mark.parametrize("provider,mode", _get_provider_mode_params())
def test_mode_is_registered(provider: Provider, mode: Mode) -> None:
    """Test that all expected modes are registered."""
    _ensure_handlers_loaded(provider)
    assert mode_registry.is_registered(provider, mode), (
        f"Mode {mode.value} should be registered for {provider.value}"
    )


@pytest.mark.parametrize("provider,mode", _get_provider_mode_params())
def test_handlers_have_all_methods(provider: Provider, mode: Mode) -> None:
    """Test that all handlers have required methods."""
    _ensure_handlers_loaded(provider)
    handlers = mode_registry.get_handlers(provider, mode)

    assert handlers.request_handler is not None, (
        f"request_handler should not be None for {provider.value}-{mode.value}"
    )
    assert handlers.reask_handler is not None, (
        f"reask_handler should not be None for {provider.value}-{mode.value}"
    )
    assert handlers.response_parser is not None, (
        f"response_parser should not be None for {provider.value}-{mode.value}"
    )


@pytest.mark.parametrize("provider", _get_provider_params())
def test_get_modes_for_provider(provider: Provider) -> None:
    """Test getting all modes for a provider."""
    _ensure_handlers_loaded(provider)
    expected_modes = set(PROVIDER_HANDLER_MODES.get(provider, []))
    registered_modes = set(mode_registry.get_modes_for_provider(provider))

    assert expected_modes.issubset(registered_modes), (
        f"Expected modes {expected_modes} should be subset of registered modes {registered_modes} for {provider.value}"
    )


@pytest.mark.parametrize("provider", _get_provider_params())
def test_provider_in_mode_providers(provider: Provider) -> None:
    """Test that provider is listed for its supported modes."""
    _ensure_handlers_loaded(provider)
    expected_modes = PROVIDER_HANDLER_MODES.get(provider, [])

    for mode in expected_modes:
        providers_for_mode = mode_registry.get_providers_for_mode(mode)
        assert provider in providers_for_mode, (
            f"{provider.value} should be in providers for {mode.value}"
        )


# ============================================================================
# Handler Inheritance Tests (OpenAI-compatible providers)
# ============================================================================


# Providers that inherit from OpenAI handlers
OPENAI_COMPATIBLE_PROVIDERS = [
    Provider.GROQ,
    Provider.FIREWORKS,
    Provider.CEREBRAS,
]


@pytest.mark.parametrize(
    "provider", [pytest.param(p, id=p.value) for p in OPENAI_COMPATIBLE_PROVIDERS]
)
def test_tools_handler_inherits_from_openai(provider: Provider) -> None:
    """Test that OpenAI-compatible providers use OpenAI handlers."""
    if Mode.TOOLS not in PROVIDER_HANDLER_MODES.get(provider, []):
        pytest.skip(f"{provider.value} does not support TOOLS mode")

    from instructor.v2.core.registry import mode_registry

    # For groq, fireworks, and cerebras, handlers are registered directly via OpenAI handlers
    # (they're in OPENAI_COMPAT_PROVIDERS list), so they use the same handler class
    if provider in (Provider.GROQ, Provider.FIREWORKS, Provider.CEREBRAS):
        # Verify handlers are registered
        assert mode_registry.is_registered(provider, Mode.TOOLS)
        # Get the handler and verify it's the OpenAI handler
        handlers = mode_registry.get_handlers(provider, Mode.TOOLS)
        # The handler functions should be the same as OpenAI's
        openai_handlers = mode_registry.get_handlers(Provider.OPENAI, Mode.TOOLS)
        assert handlers.request_handler == openai_handlers.request_handler
        assert handlers.response_parser == openai_handlers.response_parser
    else:
        # For other providers that might have separate handler classes, skip this test
        # as they may have their own implementations
        pytest.skip(f"{provider.value} may have separate handler implementation")


@pytest.mark.parametrize(
    "provider", [pytest.param(p, id=p.value) for p in OPENAI_COMPATIBLE_PROVIDERS]
)
def test_md_json_handler_inherits_from_openai(provider: Provider) -> None:
    """Test that OpenAI-compatible providers use OpenAI MD_JSON handlers."""
    if Mode.MD_JSON not in PROVIDER_HANDLER_MODES.get(provider, []):
        pytest.skip(f"{provider.value} does not support MD_JSON mode")

    from instructor.v2.core.registry import mode_registry

    # For groq, fireworks, and cerebras, handlers are registered directly via OpenAI handlers
    # (they're in OPENAI_COMPAT_PROVIDERS list), so they use the same handler class
    if provider in (Provider.GROQ, Provider.FIREWORKS, Provider.CEREBRAS):
        # Verify handlers are registered
        assert mode_registry.is_registered(provider, Mode.MD_JSON)
        # Get the handler and verify it's the OpenAI handler
        handlers = mode_registry.get_handlers(provider, Mode.MD_JSON)
        # The handler functions should be the same as OpenAI's
        openai_handlers = mode_registry.get_handlers(Provider.OPENAI, Mode.MD_JSON)
        assert handlers.request_handler == openai_handlers.request_handler
        assert handlers.response_parser == openai_handlers.response_parser
    else:
        # For other providers that might have separate handler classes, skip this test
        # as they may have their own implementations
        pytest.skip(f"{provider.value} may have separate handler implementation")


# ============================================================================
# Common Unsupported Mode Tests
# ============================================================================


@pytest.mark.parametrize("provider", _get_provider_params())
def test_parallel_tools_not_supported_unless_listed(provider: Provider) -> None:
    """Test that PARALLEL_TOOLS is not supported unless in PROVIDER_HANDLER_MODES."""
    _ensure_handlers_loaded(provider)
    expected_modes = PROVIDER_HANDLER_MODES.get(provider, [])
    is_expected = Mode.PARALLEL_TOOLS in expected_modes
    is_registered = mode_registry.is_registered(provider, Mode.PARALLEL_TOOLS)

    assert is_expected == is_registered, (
        f"PARALLEL_TOOLS registration mismatch for {provider.value}: "
        f"expected={is_expected}, registered={is_registered}"
    )


@pytest.mark.parametrize("provider", _get_provider_params())
def test_responses_tools_not_supported_unless_listed(provider: Provider) -> None:
    """Test that RESPONSES_TOOLS is not supported unless in PROVIDER_HANDLER_MODES."""
    _ensure_handlers_loaded(provider)
    expected_modes = PROVIDER_HANDLER_MODES.get(provider, [])
    is_expected = Mode.RESPONSES_TOOLS in expected_modes
    is_registered = mode_registry.is_registered(provider, Mode.RESPONSES_TOOLS)

    assert is_expected == is_registered, (
        f"RESPONSES_TOOLS registration mismatch for {provider.value}: "
        f"expected={is_expected}, registered={is_registered}"
    )
