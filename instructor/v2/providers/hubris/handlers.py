"""Hubris v2 mode handlers.

Hubris speaks the OpenAI wire format without provider-specific quirks, so it is
listed in the OpenAI-compatible provider groups and reuses those handlers for
TOOLS, JSON_SCHEMA, MD_JSON and PARALLEL_TOOLS. Importing the module here is
what performs that registration.
"""

from __future__ import annotations

from instructor.v2.providers.openai import handlers as _openai_handlers  # noqa: F401

__all__: list[str] = []
