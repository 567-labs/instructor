"""v2 OpenAI provider."""

try:
    from instructor.v2.providers.openai.client import from_openai
except ImportError:
    from_openai = None

__all__ = ["from_openai"]
