"""Unified parametrized tests for all provider client factories.

These tests verify client factory behavior (mode normalization, registry, errors, imports)
across all providers without requiring API keys.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from instructor import Mode, Provider
from instructor.v2.core.registry import mode_registry, normalize_mode
from tests.v2.provider_matrix import legacy_config_dicts

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_HANDLER_MODULE_PATHS: dict[Provider, Path] = {
    Provider.OPENAI: _PROJECT_ROOT / "instructor/v2/providers/openai/handlers.py",
    Provider.ANYSCALE: _PROJECT_ROOT / "instructor/v2/providers/openai/handlers.py",
    Provider.TOGETHER: _PROJECT_ROOT / "instructor/v2/providers/openai/handlers.py",
    Provider.DATABRICKS: _PROJECT_ROOT / "instructor/v2/providers/openai/handlers.py",
    Provider.DEEPSEEK: _PROJECT_ROOT / "instructor/v2/providers/openai/handlers.py",
    Provider.ANTHROPIC: _PROJECT_ROOT / "instructor/v2/providers/anthropic/handlers.py",
    Provider.GENAI: _PROJECT_ROOT / "instructor/v2/providers/genai/handlers.py",
    Provider.GEMINI: _PROJECT_ROOT / "instructor/v2/providers/gemini/handlers.py",
    Provider.COHERE: _PROJECT_ROOT / "instructor/v2/providers/cohere/handlers.py",
    Provider.OPENROUTER: _PROJECT_ROOT
    / "instructor/v2/providers/openrouter/handlers.py",
    Provider.HUBRIS: _PROJECT_ROOT / "instructor/v2/providers/hubris/handlers.py",
    Provider.PERPLEXITY: _PROJECT_ROOT
    / "instructor/v2/providers/perplexity/handlers.py",
    Provider.XAI: _PROJECT_ROOT / "instructor/v2/providers/xai/handlers.py",
    Provider.GROQ: _PROJECT_ROOT / "instructor/v2/providers/openai/handlers.py",
    Provider.MISTRAL: _PROJECT_ROOT / "instructor/v2/providers/mistral/handlers.py",
    Provider.FIREWORKS: _PROJECT_ROOT / "instructor/v2/providers/openai/handlers.py",
    Provider.CEREBRAS: _PROJECT_ROOT / "instructor/v2/providers/openai/handlers.py",
    Provider.WRITER: _PROJECT_ROOT / "instructor/v2/providers/writer/handlers.py",
    Provider.BEDROCK: _PROJECT_ROOT / "instructor/v2/providers/bedrock/handlers.py",
    Provider.VERTEXAI: _PROJECT_ROOT / "instructor/v2/providers/vertexai/handlers.py",
}
_HANDLERS_LOADED: set[Provider] = set()


def _clear_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "ALL_PROXY",
        "all_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
    ):
        monkeypatch.delenv(key, raising=False)


def _ensure_handlers_loaded(provider: Provider) -> None:
    if provider in _HANDLERS_LOADED:
        return
    provider_modes = PROVIDER_CLIENT_CONFIGS.get(provider, {}).get(
        "supported_modes", []
    )
    if provider_modes and all(
        mode_registry.is_registered(provider, mode) for mode in provider_modes
    ):
        _HANDLERS_LOADED.add(provider)
        return
    handler_path = _HANDLER_MODULE_PATHS.get(provider)
    if handler_path is None or not handler_path.exists():
        return
    spec = importlib.util.spec_from_file_location(
        f"tests.v2.handlers_{provider.value}",
        handler_path,
    )
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _HANDLERS_LOADED.add(provider)


PROVIDER_CLIENT_CONFIGS: dict[Provider, dict[str, Any]] = legacy_config_dicts()


def _dependency_missing(module: str) -> bool:
    """Check if a dependency module is missing."""
    try:
        return importlib.util.find_spec(module.split(".")[0]) is None
    except ModuleNotFoundError:
        return True


def _is_expected_missing_dependency(provider: Provider, exc: ImportError) -> bool:
    """Return True when an import failed because the provider SDK is unavailable."""
    sdk_module = PROVIDER_CLIENT_CONFIGS[provider]["sdk_module"]
    expected_root = str(sdk_module).split(".")[0]
    missing_name = getattr(exc, "name", None)
    if missing_name:
        return missing_name.split(".")[0] == expected_root

    return f"No module named '{expected_root}'" in str(exc)


def _get_provider_params():
    """Generate provider parameters for parametrized tests."""
    return [
        pytest.param(provider, id=provider.value)
        for provider in PROVIDER_CLIENT_CONFIGS.keys()
    ]


def _get_provider_mode_params():
    """Generate (provider, mode) parameters for supported modes."""
    params = []
    for provider, config in PROVIDER_CLIENT_CONFIGS.items():
        for mode in config["supported_modes"]:
            params.append(
                pytest.param(provider, mode, id=f"{provider.value}-{mode.value}")
            )
    return params


def _get_provider_unsupported_mode_params():
    """Generate (provider, mode) parameters for unsupported modes."""
    params = []
    for provider, config in PROVIDER_CLIENT_CONFIGS.items():
        for mode in config["unsupported_modes"]:
            params.append(
                pytest.param(provider, mode, id=f"{provider.value}-{mode.value}")
            )
    return params


def _get_provider_legacy_mode_params():
    """Generate (provider, legacy_mode) parameters."""
    params = []
    for provider, config in PROVIDER_CLIENT_CONFIGS.items():
        for legacy_mode in config["legacy_modes"].keys():
            params.append(
                pytest.param(
                    provider,
                    legacy_mode,
                    id=f"{provider.value}-{legacy_mode.value}",
                )
            )
    return params


# ============================================================================
# Mode Registry Tests
# ============================================================================


@pytest.mark.parametrize("provider,mode", _get_provider_mode_params())
def test_supported_mode_is_registered(provider: Provider, mode: Mode) -> None:
    """Test that all supported modes are registered in the registry."""
    _ensure_handlers_loaded(provider)
    assert mode_registry.is_registered(provider, mode), (
        f"Mode {mode.value} should be registered for {provider.value}"
    )


@pytest.mark.parametrize("provider,mode", _get_provider_unsupported_mode_params())
def test_unsupported_mode_not_registered(provider: Provider, mode: Mode) -> None:
    """Test that unsupported modes are NOT registered."""
    assert not mode_registry.is_registered(provider, mode), (
        f"Mode {mode.value} should NOT be registered for {provider.value}"
    )


@pytest.mark.parametrize("provider", _get_provider_params())
def test_get_modes_for_provider(provider: Provider) -> None:
    """Test getting all modes for a provider."""
    _ensure_handlers_loaded(provider)
    config = PROVIDER_CLIENT_CONFIGS[provider]
    registered_modes = mode_registry.get_modes_for_provider(provider)

    # All supported modes should be registered
    for mode in config["supported_modes"]:
        assert mode in registered_modes, (
            f"Mode {mode.value} should be in registered modes for {provider.value}"
        )

    # Unsupported modes should not be registered
    for mode in config["unsupported_modes"]:
        assert mode not in registered_modes, (
            f"Mode {mode.value} should NOT be in registered modes for {provider.value}"
        )


@pytest.mark.parametrize("provider,mode", _get_provider_mode_params())
def test_handlers_have_all_methods(provider: Provider, mode: Mode) -> None:
    """Test that all handlers have required methods."""
    _ensure_handlers_loaded(provider)
    handlers = mode_registry.get_handlers(provider, mode)

    assert handlers.request_handler is not None
    assert handlers.reask_handler is not None
    assert handlers.response_parser is not None


# ============================================================================
# Mode Normalization Tests
# ============================================================================


@pytest.mark.parametrize("provider,mode", _get_provider_mode_params())
def test_generic_mode_passes_through(provider: Provider, mode: Mode) -> None:
    """Test that generic modes pass through unchanged."""
    result = normalize_mode(provider, mode)
    assert result == mode, (
        f"Generic mode {mode.value} should pass through unchanged for {provider.value}"
    )


@pytest.mark.parametrize("provider,legacy_mode", _get_provider_legacy_mode_params())
def test_legacy_mode_normalizes_to_registered_mode(
    provider: Provider, legacy_mode: Mode
) -> None:
    """Legacy provider-specific modes normalize to registered v2 modes."""
    result = normalize_mode(provider, legacy_mode)
    assert result != legacy_mode
    assert mode_registry.is_registered(provider, legacy_mode), (
        f"Legacy mode {legacy_mode.value} should remain accepted for {provider.value}"
    )


# ============================================================================
# Import Tests
# ============================================================================


@pytest.mark.parametrize("provider", _get_provider_params())
def test_from_function_importable(provider: Provider) -> None:
    """Test that from_* function is importable from instructor.v2."""
    config = PROVIDER_CLIENT_CONFIGS[provider]
    from_function = config["from_function"]

    # Import from instructor.v2
    module = __import__("instructor.v2", fromlist=[from_function])
    func = getattr(module, from_function, None)

    # Should be None if SDK not installed, or a callable if installed
    assert func is None or callable(func), (
        f"{from_function} should be None or callable, got {type(func)}"
    )


@pytest.mark.parametrize("provider", _get_provider_params())
def test_handlers_importable(provider: Provider) -> None:
    """Test that handlers are importable."""
    handler_path = _HANDLER_MODULE_PATHS.get(provider)
    assert handler_path is not None and handler_path.exists(), (
        f"Missing handler module path for {provider.value}"
    )

    _ensure_handlers_loaded(provider)

    assert any(
        mode_registry.is_registered(provider, mode)
        for mode in PROVIDER_CLIENT_CONFIGS[provider]["supported_modes"]
    ), f"No registered handlers found for {provider.value}"


# ============================================================================
# Error Handling Tests
# ============================================================================


@pytest.mark.parametrize("provider,mode", _get_provider_unsupported_mode_params())
def test_unsupported_mode_raises_error(provider: Provider, mode: Mode) -> None:
    """Test that getting handlers for unsupported mode raises KeyError."""
    with pytest.raises(KeyError):
        mode_registry.get_handlers(provider, mode)


@pytest.mark.parametrize("provider", _get_provider_params())
def test_parallel_tools_not_supported_unless_registered(provider: Provider) -> None:
    """Test that PARALLEL_TOOLS is not supported unless registered."""
    config = PROVIDER_CLIENT_CONFIGS[provider]
    is_supported = Mode.PARALLEL_TOOLS in config["supported_modes"]
    is_registered = mode_registry.is_registered(provider, Mode.PARALLEL_TOOLS)

    assert is_supported == is_registered, (
        f"PARALLEL_TOOLS support mismatch for {provider.value}: "
        f"supported={is_supported}, registered={is_registered}"
    )


@pytest.mark.parametrize("provider", _get_provider_params())
def test_responses_tools_not_supported_unless_registered(provider: Provider) -> None:
    """Test that RESPONSES_TOOLS is not supported unless registered."""
    config = PROVIDER_CLIENT_CONFIGS[provider]
    is_supported = Mode.RESPONSES_TOOLS in config["supported_modes"]
    is_registered = mode_registry.is_registered(provider, Mode.RESPONSES_TOOLS)

    assert is_supported == is_registered, (
        f"RESPONSES_TOOLS support mismatch for {provider.value}: "
        f"supported={is_supported}, registered={is_registered}"
    )


# ============================================================================
# SDK Availability Tests
# ============================================================================


@pytest.mark.parametrize("provider", _get_provider_params())
def test_from_function_raises_without_sdk(provider: Provider) -> None:
    """Test that from_* function raises error when SDK not installed."""
    config = PROVIDER_CLIENT_CONFIGS[provider]
    sdk_module = config["sdk_module"]
    from_function = config["from_function"]

    if not _dependency_missing(sdk_module):
        pytest.skip(
            f"{sdk_module} is installed"  # ty: ignore[too-many-positional-arguments]
        )

    # Try to import the from_* function from the provider's client module
    try:
        client_module_path = f"instructor.v2.providers.{provider.value}.client"
        client_module = __import__(client_module_path, fromlist=[from_function])
        from_function_obj = getattr(client_module, from_function, None)

        if from_function_obj is None:
            pytest.skip(
                f"{from_function} not found in client module"  # ty: ignore[too-many-positional-arguments]
            )

        from instructor.core.exceptions import ClientError

        expected_message = config.get(
            "missing_sdk_message",
            f"{sdk_module.split('.')[0]} is not installed",
        )
        with pytest.raises(ClientError, match=expected_message):
            from_function_obj("not a client")  # type: ignore[call-arg]
    except (ImportError, ModuleNotFoundError) as exc:
        if _is_expected_missing_dependency(provider, exc):
            pytest.skip(
                f"{sdk_module} import path is unavailable in this environment"  # ty: ignore[too-many-positional-arguments]
            )
        raise


# ============================================================================
# String-Based Initialization Tests
# ============================================================================


# OpenAI-compatible providers that support string-based initialization
_OPENAI_COMPAT_PROVIDERS = [
    Provider.ANYSCALE,
    Provider.TOGETHER,
    Provider.DATABRICKS,
    Provider.DEEPSEEK,
]


@pytest.mark.parametrize(
    "provider",
    [pytest.param(p, id=p.value) for p in _OPENAI_COMPAT_PROVIDERS],
)
def test_string_based_initialization_delegates_to_from_provider(
    provider: Provider,
) -> None:
    """Test that string-based initialization delegates to from_provider."""
    config = PROVIDER_CLIENT_CONFIGS[provider]
    from_function = config["from_function"]

    # Import the from_* function
    module = __import__("instructor.v2", fromlist=[from_function])
    func = getattr(module, from_function, None)

    if func is None:
        pytest.skip(
            f"{from_function} not available (SDK may not be installed)"  # ty: ignore[too-many-positional-arguments]
        )

    # Mock from_provider to verify it's called
    from unittest.mock import patch

    with patch("instructor.from_provider") as mock_from_provider:
        # Call with string (model name)
        func("test-model", mode=Mode.TOOLS)

        # Verify from_provider was called with correct provider prefix
        mock_from_provider.assert_called_once()
        call_args = mock_from_provider.call_args
        assert call_args[0][0] == f"{provider.value}/test-model"
        assert call_args[1]["mode"] == Mode.TOOLS


@pytest.mark.parametrize(
    "provider",
    [pytest.param(p, id=p.value) for p in _OPENAI_COMPAT_PROVIDERS],
)
def test_string_based_initialization_with_async_client(provider: Provider) -> None:
    """Test that string-based initialization supports async_client parameter."""
    config = PROVIDER_CLIENT_CONFIGS[provider]
    from_function = config["from_function"]

    # Import the from_* function
    module = __import__("instructor.v2", fromlist=[from_function])
    func = getattr(module, from_function, None)

    if func is None:
        pytest.skip(
            f"{from_function} not available (SDK may not be installed)"  # ty: ignore[too-many-positional-arguments]
        )

    # Mock from_provider to verify it's called
    from unittest.mock import patch

    with patch("instructor.from_provider") as mock_from_provider:
        # Call with string and async_client=True
        func("test-model", mode=Mode.TOOLS, async_client=True)

        # Verify from_provider was called with async_client=True
        mock_from_provider.assert_called_once()
        call_args = mock_from_provider.call_args
        assert call_args[0][0] == f"{provider.value}/test-model"
        assert call_args[1]["mode"] == Mode.TOOLS
        assert call_args[1]["async_client"] is True


@pytest.mark.parametrize(
    "provider",
    [pytest.param(p, id=p.value) for p in _OPENAI_COMPAT_PROVIDERS],
)
def test_string_based_initialization_forwards_kwargs(provider: Provider) -> None:
    """Test that string-based initialization forwards all kwargs to from_provider."""
    config = PROVIDER_CLIENT_CONFIGS[provider]
    from_function = config["from_function"]

    # Import the from_* function
    module = __import__("instructor.v2", fromlist=[from_function])
    func = getattr(module, from_function, None)

    if func is None:
        pytest.skip(
            f"{from_function} not available (SDK may not be installed)"  # ty: ignore[too-many-positional-arguments]
        )

    # Mock from_provider to verify it's called
    from unittest.mock import patch

    with patch("instructor.from_provider") as mock_from_provider:
        # Call with string and additional kwargs
        func(
            "test-model",
            mode=Mode.TOOLS,
            api_key="test-key",
            base_url="https://test.example.com",
            timeout=30,
        )

        # Verify from_provider was called with all kwargs
        mock_from_provider.assert_called_once()
        call_args = mock_from_provider.call_args
        assert call_args[0][0] == f"{provider.value}/test-model"
        assert call_args[1]["mode"] == Mode.TOOLS
        assert call_args[1]["api_key"] == "test-key"
        assert call_args[1]["base_url"] == "https://test.example.com"
        assert call_args[1]["timeout"] == 30


@pytest.mark.parametrize(
    "provider",
    [pytest.param(p, id=p.value) for p in _OPENAI_COMPAT_PROVIDERS],
)
def test_client_based_initialization_still_works(
    provider: Provider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that client-based initialization still works (backward compatibility)."""
    from unittest.mock import patch

    config = PROVIDER_CLIENT_CONFIGS[provider]
    from_function = config["from_function"]
    sdk_module = config["sdk_module"]

    # Skip if SDK not installed
    if _dependency_missing(sdk_module):
        pytest.skip(
            f"{sdk_module} not installed"  # ty: ignore[too-many-positional-arguments]
        )

    # Import the from_* function
    module = __import__("instructor.v2", fromlist=[from_function])
    func = getattr(module, from_function, None)

    if func is None:
        pytest.skip(
            f"{from_function} not available"  # ty: ignore[too-many-positional-arguments]
        )

    # Import OpenAI client
    try:
        import openai
    except ImportError:
        pytest.skip(
            "openai package not installed"  # ty: ignore[too-many-positional-arguments]
        )

    _clear_proxy_env(monkeypatch)

    # Create a mock OpenAI client
    client = openai.OpenAI(api_key="test-key")

    # Call with client (should use _from_openai_compat, not from_provider)
    with patch(
        "instructor.v2.providers.openai.client._from_openai_compat"
    ) as mock_compat:
        mock_compat.return_value = "mock_instructor"
        result = func(client, mode=Mode.TOOLS)

        # Verify _from_openai_compat was called (not from_provider)
        mock_compat.assert_called_once()
        call_args = mock_compat.call_args
        assert call_args[0][0] == client
        assert call_args[1]["provider"] == provider
        assert call_args[1]["mode"] == Mode.TOOLS
