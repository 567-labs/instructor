"""v2 Cerebras provider."""

try:
    from instructor.v2.providers.cerebras.client import from_cerebras
except ImportError:
    from_cerebras = None

__all__ = ["from_cerebras"]
