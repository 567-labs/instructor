"""v2 Mistral provider.

Provides Instructor integration with Mistral AI using the v2 registry system.
Supports TOOLS, JSON_SCHEMA, and MD_JSON modes.
"""

try:
    from instructor.v2.providers.mistral.client import from_mistral
except ImportError:
    from_mistral = None

__all__ = ["from_mistral"]
