"""
Provider capability definitions for test skipping.

This module defines which capabilities each provider supports, allowing tests
to skip when a provider doesn't support a required feature.
"""

from typing import Literal
import instructor

# Capability types
Capability = Literal[
    "streaming",
    "partial_streaming",
    "iterable_streaming",
    "list_extraction",
    "nested_models",
    "validation",
    "response_model_none",
    "create_with_completion",
    "union_types",
    "enum_types",
    "union_streaming",
]

# Provider capabilities mapping
# Format: provider_name -> set of supported capabilities
PROVIDER_CAPABILITIES: dict[str, set[Capability]] = {
    "openai": {
        "streaming",
        "partial_streaming",
        "iterable_streaming",
        "list_extraction",
        "nested_models",
        "validation",
        "response_model_none",
        "create_with_completion",
    },
    "anthropic": {
        "streaming",
        "partial_streaming",
        "iterable_streaming",
        "list_extraction",
        "nested_models",
        "validation",
        "response_model_none",
        "create_with_completion",
    },
    "google": {
        "streaming",
        "partial_streaming",
        "iterable_streaming",
        "list_extraction",
        "nested_models",
        "validation",
        "response_model_none",
        "create_with_completion",
        # Note: Gemini doesn't support Union types or Enum types, only Optional
        # Also doesn't support union streaming
    },
    "cohere": {
        "streaming",
        "partial_streaming",
        "iterable_streaming",
        "list_extraction",
        "nested_models",
        "validation",
        "response_model_none",
        "create_with_completion",
    },
    "xai": {
        "streaming",
        "partial_streaming",
        "iterable_streaming",
        # list_extraction may have issues with tool_calls
        "nested_models",
        "validation",
        "response_model_none",
        "create_with_completion",
    },
    "mistral": {
        "streaming",
        "partial_streaming",
        "iterable_streaming",
        "list_extraction",
        "nested_models",
        "validation",
        "create_with_completion",
    },
    "cerebras": {
        "streaming",
        "partial_streaming",
        "iterable_streaming",
        "list_extraction",
        "nested_models",
        "validation",
        "create_with_completion",
    },
    "fireworks": {
        "streaming",
        "partial_streaming",
        "iterable_streaming",
        "list_extraction",
        "nested_models",
        "validation",
        "create_with_completion",
    },
    "writer": {
        "streaming",
        "partial_streaming",
        "iterable_streaming",
        "list_extraction",
        "nested_models",
        "validation",
        "create_with_completion",
    },
    "perplexity": {
        # Limited streaming support
        "list_extraction",
        "nested_models",
        "validation",
        "create_with_completion",
    },
}


def get_provider_name(model_string: str) -> str:
    """Extract provider name from model string (e.g., 'openai/gpt-4' -> 'openai')."""
    return model_string.split("/")[0]


def provider_supports(
    provider_config: tuple[str, instructor.Mode], capability: Capability
) -> bool:
    """
    Check if a provider supports a specific capability.

    Args:
        provider_config: Tuple of (model_string, mode)
        capability: The capability to check

    Returns:
        True if the provider supports the capability, False otherwise
    """
    model_string, _ = provider_config
    provider_name = get_provider_name(model_string)
    capabilities = PROVIDER_CAPABILITIES.get(provider_name, set())
    return capability in capabilities


def skip_if_unsupported(
    provider_config: tuple[str, instructor.Mode], capability: Capability
):
    """
    Skip test if provider doesn't support the capability.

    Args:
        provider_config: Tuple of (model_string, mode)
        capability: The capability required for the test
    """
    import pytest

    if not provider_supports(provider_config, capability):
        model_string, mode = provider_config
        provider_name = get_provider_name(model_string)
        pytest.skip(
            f"{provider_name} does not support {capability} "
            f"(model: {model_string}, mode: {mode})"
        )
