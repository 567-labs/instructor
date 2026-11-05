"""Azure OpenAI / AI Foundry v2 provider."""

try:
    from instructor.v2.providers.azure.client import from_azure
except ImportError:
    from_azure = None  # type: ignore

__all__ = ["from_azure"]
