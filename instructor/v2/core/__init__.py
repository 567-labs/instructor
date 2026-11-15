"""Core v2 infrastructure with lazy public exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "Provider",
    "Mode",
    "mode_registry",
    "ModeRegistry",
    "ModeHandlers",
    "RequestHandler",
    "ReaskHandler",
    "ResponseParser",
    "normalize_mode",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "Provider": ("instructor.v2.core.providers", "Provider"),
    "Mode": ("instructor.v2.core.mode", "Mode"),
    "mode_registry": ("instructor.v2.core.registry", "mode_registry"),
    "ModeRegistry": ("instructor.v2.core.registry", "ModeRegistry"),
    "ModeHandlers": ("instructor.v2.core.registry", "ModeHandlers"),
    "RequestHandler": ("instructor.v2.core.protocols", "RequestHandler"),
    "ReaskHandler": ("instructor.v2.core.protocols", "ReaskHandler"),
    "ResponseParser": ("instructor.v2.core.protocols", "ResponseParser"),
    "normalize_mode": ("instructor.v2.core.registry", "normalize_mode"),
}


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr_name = _LAZY_ATTRS[name]
    value = getattr(import_module(module_path), attr_name)
    globals()[name] = value
    return value
