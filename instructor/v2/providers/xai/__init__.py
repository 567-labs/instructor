"""v2 xAI provider.

Provides Instructor integration for xAI's Grok models using the v2 registry system.
"""

try:
    from instructor.v2.providers.xai.client import from_xai
except ImportError:
    from_xai = None

__all__ = ["from_xai"]
