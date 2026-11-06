"""
Shared configuration for multi-provider tests.

This module provides common test configuration for running the same tests
across multiple providers (OpenAI, Anthropic, Google, Cohere, xAI, Mistral,
Cerebras, Fireworks, Writer, Perplexity).
"""

import os

import instructor
import pytest


GOOGLE_GENAI_MODEL = os.getenv("GOOGLE_GENAI_MODEL", "")

# Provider configurations: (model_string, mode, required_env_var, required_package)
PROVIDER_CONFIGS = [
    (
        "openai/gpt-4.1-mini",
        instructor.Mode.TOOLS,
        "OPENAI_API_KEY",
        "openai",
    ),
    (
        "anthropic/claude-haiku-4-5-20251001",
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


def get_available_providers() -> list[tuple[str, instructor.Mode]]:
    """
    Get list of available providers based on API keys and installed packages.

    Returns:
        List of tuples (model_string, mode) for available providers
    """
    available = []

    for model, mode, env_var, package in PROVIDER_CONFIGS:
        if not model:
            continue
        # Check if API key is set
        if not os.getenv(env_var):
            continue

        # Check if package is installed
        try:
            parts = package.split(".")
            if len(parts) > 1:
                __import__(parts[0])
                # For nested imports like google.genai
                __import__(package)
            else:
                __import__(package)
            available.append((model, mode))
        except ImportError:
            continue

    return available


def pytest_generate_tests(metafunc):
    """
    Pytest hook to generate parametrized tests for available providers.

    This is used in test files that have 'provider_config' as a parameter.
    """
    if "provider_config" in metafunc.fixturenames:
        available = get_available_providers()
        if not available:
            pytest.skip("No providers available (missing API keys or packages)")

        # Generate test IDs like "openai" "anthropic" "google"
        ids = [model.split("/")[0] for model, _ in available]
        metafunc.parametrize("provider_config", available, ids=ids)


def pytest_configure(config):
    """Register custom markers for provider-specific tests."""
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
