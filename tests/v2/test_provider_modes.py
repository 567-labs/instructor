"""
Comprehensive parametrized tests for all provider modes.

Tests all registered modes for each provider with actual API calls to ensure complete coverage.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any, Literal, Union

import importlib.util

import pytest
from pydantic import BaseModel
from openai.types.chat import ChatCompletionMessageParam

import instructor
from instructor.core.exceptions import InstructorRetryException
from instructor import Mode
from instructor.v2 import Provider, mode_registry
from tests.v2.provider_matrix import HANDLER_MODULE_PATHS, legacy_config_dicts

# Ensure handlers are loaded by dynamically importing them
_HANDLERS_LOADED: set[Provider] = set()


def _is_expected_missing_dependency(provider: Provider, exc: ImportError) -> bool:
    """Return True when handler loading failed only because an optional SDK is absent."""
    sdk_module = PROVIDER_CONFIGS.get(provider, {}).get("sdk_module")
    if not sdk_module:
        return False

    expected_root = str(sdk_module).split(".")[0]
    missing_name = getattr(exc, "name", None)
    if missing_name:
        return missing_name.split(".")[0] == expected_root

    return f"No module named '{expected_root}'" in str(exc)


def _ensure_handlers_loaded(provider: Provider) -> None:
    """Dynamically load handler module to ensure handlers are registered."""
    if provider in _HANDLERS_LOADED:
        return
    handler_path = HANDLER_MODULE_PATHS.get(provider)
    if handler_path is None:
        return
    if not handler_path.exists():
        return
    try:
        spec = importlib.util.spec_from_file_location(
            f"tests.v2.handlers_{provider.value}",
            handler_path,
        )
        if spec is None or spec.loader is None:
            return
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _HANDLERS_LOADED.add(provider)
    except (ImportError, ModuleNotFoundError) as exc:
        if _is_expected_missing_dependency(provider, exc):
            pytest.skip(
                f"{provider.value} handlers require optional dependency "
                f"{PROVIDER_CONFIGS[provider]['sdk_module']}"
            )
        raise


class Answer(BaseModel):
    """Simple answer model."""

    answer: float


class Weather(BaseModel):
    """Weather tool."""

    location: str
    units: Literal["imperial", "metric"]


class GoogleSearch(BaseModel):
    """Search tool."""

    query: str


_PROVIDER_CLIENT_CONFIGS = legacy_config_dicts()
PROVIDER_CONFIGS = {
    provider: {
        "provider_string": config["provider_string"],
        "sdk_module": config["sdk_module"],
        "modes": config["supported_modes"],
        "basic_modes": config["basic_modes"],
        "async_modes": config["async_modes"],
    }
    for provider, config in _PROVIDER_CLIENT_CONFIGS.items()
    if config["provider_string"] is not None and config["basic_modes"]
}


def _get_all_mode_params():
    """Generate (provider, mode) tuples for all registered modes."""
    params = []
    for provider, config in PROVIDER_CONFIGS.items():
        for mode in config["modes"]:
            params.append((provider, mode))
    return params


@pytest.mark.parametrize("provider,mode", _get_all_mode_params())
def test_mode_is_registered(provider: Provider, mode: Mode):
    """Verify each mode is registered in the v2 registry."""
    _ensure_handlers_loaded(provider)

    sdk = PROVIDER_CONFIGS[provider]["sdk_module"]
    if sdk and importlib.util.find_spec(sdk.split(".")[0]) is None:
        pytest.skip(f"{sdk} is not installed")
    assert mode_registry.is_registered(provider, mode)

    handlers = mode_registry.get_handlers(provider, mode)
    assert handlers.request_handler is not None
    assert handlers.reask_handler is not None
    assert handlers.response_parser is not None


def _skip_on_provider_quota(provider: Provider, exc: Exception) -> None:
    """Skip tests when provider quota limits prevent execution."""
    if os.environ.get("CI") or os.environ.get("INSTRUCTOR_STRICT_PROVIDER_TESTS"):
        return
    if (
        provider == Provider.GENAI
        and isinstance(exc, InstructorRetryException)
        and "RESOURCE_EXHAUSTED" in str(exc)
    ):
        # fmt: off
        pytest.skip("GenAI quota exhausted for this environment")
        # fmt: on
    if (
        provider == Provider.OPENAI
        and isinstance(exc, InstructorRetryException)
        and "Connection error" in str(exc)
    ):
        # fmt: off
        pytest.skip("OpenAI connectivity is unavailable in this environment")
        # fmt: on


@pytest.mark.parametrize(
    ("provider", "mode", "asynchronous"),
    [
        pytest.param(
            provider, mode, asynchronous, id=f"{provider.value}-{mode.name}-{direction}"
        )
        for provider, config in PROVIDER_CONFIGS.items()
        for direction, asynchronous, modes in (
            ("sync", False, config["basic_modes"]),
            ("async", True, config["async_modes"]),
        )
        for mode in modes
    ],
)
@pytest.mark.asyncio
@pytest.mark.requires_api_key
async def test_mode_extraction(provider: Provider, mode: Mode, asynchronous: bool):
    model = PROVIDER_CONFIGS[provider]["provider_string"]
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "What is 2 + 2? Reply with a number."}
    ]
    token_limit = (
        "max_completion_tokens" if provider == Provider.OPENAI else "max_tokens"
    )
    options: dict[str, Any] = {token_limit: 1000}
    try:
        if asynchronous:
            client = instructor.from_provider(model, mode=mode, async_client=True)
            response = await client.chat.completions.create(
                response_model=Answer, messages=messages, **options
            )
        else:
            client = instructor.from_provider(model, mode=mode)
            response = client.chat.completions.create(
                response_model=Answer, messages=messages, **options
            )
    except InstructorRetryException as exc:
        _skip_on_provider_quota(provider, exc)
        raise

    assert isinstance(response, Answer)
    assert response.answer == 4.0


@pytest.mark.provider(Provider.ANTHROPIC)
@pytest.mark.requires_api_key
def test_anthropic_parallel_tools_extraction():
    """Test PARALLEL_TOOLS mode extraction (Anthropic-specific)."""
    client = instructor.from_provider(
        "anthropic/claude-haiku-5-5",
        mode=Mode.PARALLEL_TOOLS,
    )
    response = client.chat.completions.create(
        response_model=Iterable[Union[Weather, GoogleSearch]],
        messages=[
            {
                "role": "system",
                "content": "You must always use tools. Use them simultaneously when appropriate.",
            },
            {
                "role": "user",
                "content": "Get weather for San Francisco and search for Python tutorials.",
            },
        ],
        max_tokens=1000,
    )

    result = list(response)
    assert len(result) >= 1
    assert all(isinstance(r, (Weather, GoogleSearch)) for r in result)


def test_anthropic_reasoning_tools_normalizes_in_v2():
    """Legacy reasoning mode remains accepted through Anthropic tools mode."""
    assert mode_registry.is_registered(
        Provider.ANTHROPIC,
        Mode.ANTHROPIC_REASONING_TOOLS,
    )
