"""v2 Cohere provider."""

try:
    from instructor.v2.providers.cohere.client import from_cohere
except ImportError:
    from_cohere = None

__all__ = ["from_cohere"]
