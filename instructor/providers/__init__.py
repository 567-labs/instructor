"""Legacy provider import paths backed by v2 provider implementations."""

from __future__ import annotations

from instructor.v2.core.provider_specs import PUBLIC_FACTORY_ATTRS

__all__ = list(PUBLIC_FACTORY_ATTRS)


def __getattr__(name: str):
    if name not in PUBLIC_FACTORY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr_name = PUBLIC_FACTORY_ATTRS[name]
    module = __import__(module_path, fromlist=[attr_name])
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
