"""Gemini v2 provider handlers and client."""

from __future__ import annotations

from typing import Any

from .handlers import GeminiJSONHandler, GeminiToolsHandler

__all__ = ["GeminiJSONHandler", "GeminiToolsHandler", "from_gemini"]


def __getattr__(name: str) -> Any:
    if name == "from_gemini":
        from .client import from_gemini

        return from_gemini
    raise AttributeError(name)
