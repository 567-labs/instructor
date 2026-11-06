"""v2 Fireworks provider.

Fireworks uses an OpenAI-compatible API, so the handlers inherit from OpenAI.
Supports TOOLS and MD_JSON modes.
"""

try:
    from instructor.v2.providers.fireworks.client import from_fireworks
except ImportError:
    from_fireworks = None

__all__ = ["from_fireworks"]
