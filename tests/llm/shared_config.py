"""
Shared configuration for multi-provider tests.

This module provides common test configuration for running the same tests
across multiple providers (OpenAI, Anthropic, Google, Cohere, xAI, Mistral,
Cerebras, Fireworks, Writer, Perplexity).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import instructor
import pytest


if TYPE_CHECKING:
    from _pytest.mark import ParameterSet


GOOGLE_GENAI_MODEL = os.getenv("GOOGLE_GENAI_MODEL", "")

# Provider configurations: (model_string, mode, required_env_var, required_package)
PROVIDER_CONFIGS = [
    (
        "openai/gpt-6-luna",
        instructor.Mode.TOOLS,
        "OPENAI_API_KEY",
        "openai",
    ),
    (
        "anthropic/claude-haiku-5-5",
        instructor.Mode.ANTHROPIC_TOOLS,
        "ANTHROPIC_API_KEY",
        "anthropic",
    ),
    (
        GOOGLE_GENAI_MODEL,
        instructor.Mode.GENAI_STRUCTURED_OUTPUTS,
        "GOOGLE_API_KEY",
        "google.genai",
    ),
    (
        "cohere/command-a-03-2025",
        instructor.Mode.COHERE_TOOLS,
        "COHERE_API_KEY",
        "cohere",
    ),
    (
        "xai/grok-3-mini",
        instructor.Mode.XAI_TOOLS,
        "XAI_API_KEY",
        "xai_sdk",
    ),
    (
        "mistral/ministral-8b-latest",
        instructor.Mode.MISTRAL_TOOLS,
        "MISTRAL_API_KEY",
        "mistralai",
    ),
    (
        "cerebras/llama3.1-70b",
        instructor.Mode.CEREBRAS_TOOLS,
        "CEREBRAS_API_KEY",
        "cerebras",
    ),
    (
        "fireworks/accounts/fireworks/models/llama-v3p3-70b-instruct",
        instructor.Mode.FIREWORKS_TOOLS,
        "FIREWORKS_API_KEY",
        "fireworks",
    ),
    (
        "writer/palmyra-x5",
        instructor.Mode.WRITER_TOOLS,
        "WRITER_API_KEY",
        "writerai",
    ),
    (
        "perplexity/sonar-pro",
        instructor.Mode.PERPLEXITY_JSON,
        "PERPLEXITY_API_KEY",
        "openai",  # Perplexity transports over OpenAI-compatible API
    ),
]


# These additional routes participate in the shared Python/Rust cases.
_PAIRED_ROUTES = [
    ("openai/gpt-6-luna", instructor.Mode.RESPONSES_TOOLS, "OPENAI_API_KEY", "openai"),
    (
        "openrouter/google/gemini-3.8-flash",
        instructor.Mode.TOOLS,
        "OPENROUTER_API_KEY",
        "openai",
    ),
]
PAIRED_PROVIDERS = {"openai", "anthropic", "google", "cohere", "mistral", "openrouter"}


def _paired_google_model() -> str:
    configured = os.getenv("GOOGLE_GENAI_MODEL", "").strip() or "gemini-3.8-flash"
    name = configured.removeprefix("google/").removeprefix("gemini/")
    return f"google/{name}"


def _missing_requirement(env_var: str, package: str) -> str | None:
    if not os.getenv(env_var):
        return f"{env_var} is not configured"
    try:
        __import__(package)
    except ImportError:
        return f"{package} package not installed"
    return None


def get_available_providers() -> list[tuple[str, instructor.Mode]]:
    """Legacy core cases run for providers with a selected model, key and SDK."""
    return [
        (model, mode)
        for model, mode, key, package in PROVIDER_CONFIGS
        if model and not _missing_requirement(key, package)
    ]


def paired_parameters(*, only: bool) -> list[ParameterSet]:
    """Give the shared cases stable node IDs and explicit missing-key skips."""
    params = []
    for model, mode, key, package in [*PROVIDER_CONFIGS, *_PAIRED_ROUTES]:
        if key == "GOOGLE_API_KEY":
            model = _paired_google_model()
        paired = model.partition("/")[0] in PAIRED_PROVIDERS
        if only and not paired:
            continue
        reason = _missing_requirement(key, package)
        if not paired and reason:
            continue
        route = (
            "openai-responses"
            if mode == instructor.Mode.RESPONSES_TOOLS
            else model.split("/")[0]
        )
        marks = (pytest.mark.skip(reason=reason),) if reason else ()
        params.append(pytest.param((model, mode), id=route, marks=marks))
    return params


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "provider_config" not in metafunc.fixturenames:
        return
    mark = metafunc.definition.get_closest_marker("paired")
    if mark is not None:
        configs = paired_parameters(only=mark.kwargs.get("only", False))
    else:
        configs = [
            pytest.param((model, mode), id=model.split("/")[0])
            for model, mode in get_available_providers()
        ]
    # Pytest marks an empty parameter set skipped. A module can still contain
    # paired cases available through a separately configured gateway.
    metafunc.parametrize("provider_config", configs)


def pytest_configure(config):
    """Register custom markers for provider-specific tests."""
    config.addinivalue_line(
        "markers", "paired: live contract shared with instructor-rs"
    )
    config.addinivalue_line("markers", "openai: mark test as requiring OpenAI provider")
    config.addinivalue_line(
        "markers", "anthropic: mark test as requiring Anthropic provider"
    )
    config.addinivalue_line("markers", "google: mark test as requiring Google provider")
    config.addinivalue_line("markers", "cohere: mark test as requiring Cohere provider")
    config.addinivalue_line("markers", "xai: mark test as requiring xAI provider")
    config.addinivalue_line(
        "markers", "mistral: mark test as requiring Mistral provider"
    )
    config.addinivalue_line(
        "markers", "cerebras: mark test as requiring Cerebras provider"
    )
    config.addinivalue_line(
        "markers", "fireworks: mark test as requiring Fireworks provider"
    )
    config.addinivalue_line("markers", "writer: mark test as requiring Writer provider")
    config.addinivalue_line(
        "markers", "perplexity: mark test as requiring Perplexity provider"
    )


# Convenience function to skip if specific provider not available
def skip_if_provider_unavailable(provider_name: str):
    """
    Skip test if specific provider is not available.

    Args:
        provider_name: One of "openai", "anthropic", "google", "cohere", "xai",
                       "mistral", "cerebras", "fireworks", "writer", "perplexity"
    """
    config_map = {
        "openai": ("OPENAI_API_KEY", "openai"),
        "anthropic": ("ANTHROPIC_API_KEY", "anthropic"),
        "google": ("GOOGLE_API_KEY", "google.genai"),
        "cohere": ("COHERE_API_KEY", "cohere"),
        "xai": ("XAI_API_KEY", "xai_sdk"),
        "mistral": ("MISTRAL_API_KEY", "mistralai"),
        "cerebras": ("CEREBRAS_API_KEY", "cerebras"),
        "fireworks": ("FIREWORKS_API_KEY", "fireworks"),
        "writer": ("WRITER_API_KEY", "writerai"),
        "perplexity": ("PERPLEXITY_API_KEY", "openai"),
    }

    if provider_name not in config_map:
        pytest.skip(f"Unknown provider: {provider_name}")

    env_var, package = config_map[provider_name]

    if not os.getenv(env_var):
        pytest.skip(f"{env_var} not set")

    try:
        __import__(package)
    except ImportError:
        pytest.skip(f"{package} package not installed")
