"""Shared provider capability matrix for v2 tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from instructor import Provider
from instructor.v2.core.provider_specs import PROVIDER_SPECS


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
HANDLER_MODULE_PATHS = {
    provider: _PROJECT_ROOT.joinpath(*spec.handler_module.split(".")).with_suffix(".py")
    for provider, spec in PROVIDER_SPECS.items()
    if spec.handler_module
}


TEST_PROVIDER_SPECS = {
    provider: spec
    for provider, spec in PROVIDER_SPECS.items()
    if spec.handler_module is not None and spec.from_function is not None
}


PROVIDER_HANDLER_MODES = {
    provider: spec.supported_modes for provider, spec in PROVIDER_SPECS.items()
}


def legacy_config_dicts() -> dict[Provider, dict[str, Any]]:
    """Expose the old dict shape while tests migrate to ProviderSpec."""
    return {
        provider: {
            "provider_string": spec.provider_string,
            "supported_modes": list(spec.supported_modes),
            "unsupported_modes": list(spec.unsupported_modes),
            "legacy_modes": spec.legacy_modes,
            "from_function": spec.from_function,
            "sdk_module": spec.sdk_module,
            "basic_modes": list(spec.basic_modes),
            "async_modes": list(spec.async_modes),
            "missing_sdk_message": spec.missing_sdk_message,
        }
        for provider, spec in TEST_PROVIDER_SPECS.items()
    }
