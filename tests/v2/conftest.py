# conftest.py
import os
import pytest
import importlib.util

from instructor import Mode, Provider
from instructor.v2.core.registry import mode_registry


# Mapping of providers to their API key environment variables and package names
PROVIDER_API_KEYS = {
    Provider.ANTHROPIC: ("ANTHROPIC_API_KEY", "anthropic"),
    Provider.GENAI: ("GOOGLE_API_KEY", "google.genai"),
    Provider.COHERE: ("COHERE_API_KEY", "cohere"),
    Provider.OPENAI: ("OPENAI_API_KEY", "openai"),
    Provider.MISTRAL: ("MISTRAL_API_KEY", "mistralai"),
    Provider.GROQ: ("GROQ_API_KEY", "groq"),
    Provider.XAI: ("XAI_API_KEY", "xai_sdk"),
    Provider.FIREWORKS: ("FIREWORKS_API_KEY", "fireworks"),
    Provider.CEREBRAS: ("CEREBRAS_API_KEY", "cerebras"),
    Provider.WRITER: ("WRITER_API_KEY", "writerai"),
    Provider.PERPLEXITY: ("PERPLEXITY_API_KEY", "openai"),
    Provider.BEDROCK: ("AWS_ACCESS_KEY_ID", "boto3"),
    Provider.VERTEXAI: ("GOOGLE_APPLICATION_CREDENTIALS", "google.cloud.aiplatform"),
    Provider.AZURE_OPENAI: ("AZURE_OPENAI_API_KEY", "openai"),
}


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_api_key: mark test as requiring provider API key"
    )
    config.addinivalue_line(
        "markers", "provider(provider): specify provider for API key checks"
    )
    config.addinivalue_line("markers", "asyncio: mark test as requiring pytest-asyncio")


def get_registered_provider_mode_pairs() -> list[tuple[Provider, Mode]]:
    """Return all registered (provider, mode) pairs from the v2 registry."""
    pairs = mode_registry.list_modes()
    # Return empty list if no modes registered - tests using this should handle empty case
    return pairs


@pytest.fixture(autouse=True)
def check_api_key_requirement(request):
    """Skip tests marked with 'requires_api_key' if API key is not set.

    Automatically detects the provider from test parameters and checks
    for the appropriate API key.
    """
    if not request.node.get_closest_marker("requires_api_key"):
        return

    # Try to get provider from test parameters
    provider = None
    provider_marker = request.node.get_closest_marker("provider")
    if provider_marker and provider_marker.args:
        provider = provider_marker.args[0]
    if hasattr(request, "param"):
        provider = request.param
    elif (
        hasattr(request.node, "callspec") and "provider" in request.node.callspec.params
    ):
        provider = request.node.callspec.params["provider"]

    if provider is None:
        # Fallback: check if any provider API key is set
        for _prov, (env_var, _pkg) in PROVIDER_API_KEYS.items():
            if os.getenv(env_var):
                return  # At least one API key is set
        pytest.skip(
            "No provider API key environment variable is set"  # ty: ignore[too-many-positional-arguments]
        )
        return

    # Check for specific provider
    if provider in PROVIDER_API_KEYS:
        env_var, package = PROVIDER_API_KEYS[provider]

        if not os.getenv(env_var):
            pytest.skip(
                f"{env_var} environment variable not set"  # ty: ignore[too-many-positional-arguments]
            )

        if importlib.util.find_spec(package.split(".")[0]) is None:
            pytest.skip(
                f"{package} package is not installed"  # ty: ignore[too-many-positional-arguments]
            )

    if request.node.get_closest_marker("asyncio"):
        if importlib.util.find_spec("pytest_asyncio") is None:
            pytest.skip(
                "pytest-asyncio is not installed"  # ty: ignore[too-many-positional-arguments]
            )
