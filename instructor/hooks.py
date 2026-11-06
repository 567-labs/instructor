"""Backwards compatibility module for instructor.hooks.

This module provides lazy imports to maintain backwards compatibility.
"""

import warnings


def __getattr__(name: str):
    """Lazy import to provide backward compatibility for hooks imports."""
    warnings.warn(
        f"Importing from 'instructor.hooks' is deprecated and will be removed in v2.0.0. "
        f"Please update your imports to use 'instructor.core.hooks.{name}' instead:\n"
        "  from instructor.core.hooks import Hooks, HookName",
        DeprecationWarning,
        stacklevel=2,
    )

    from .core import hooks as core_hooks

    # Try to get the attribute from the core.hooks module
    if hasattr(core_hooks, name):
        return getattr(core_hooks, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
