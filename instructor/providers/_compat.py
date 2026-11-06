"""Compatibility helpers for legacy provider import paths."""

from __future__ import annotations

from collections.abc import Iterable
from importlib import import_module
from typing import Any


def resolve_provider_attr(provider: str, modules: Iterable[str], name: str) -> Any:
    """Resolve a legacy provider attribute from the v2 provider package."""
    for module_name in modules:
        module = import_module(f"instructor.v2.providers.{provider}.{module_name}")
        try:
            return getattr(module, name)
        except AttributeError:
            continue
    raise AttributeError(
        f"module 'instructor.providers.{provider}' has no attribute {name!r}"
    )


def make_getattr(provider: str, modules: Iterable[str]):
    module_names = tuple(modules)

    def __getattr__(name: str) -> Any:
        return resolve_provider_attr(provider, module_names, name)

    return __getattr__
