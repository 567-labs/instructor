# conftest.py
import os
import pytest
import importlib.util


if not os.getenv("ANTHROPIC_API_KEY"):
    pytest.skip(
        "ANTHROPIC_API_KEY environment variable not set",
        allow_module_level=True,
    )

if (
    importlib.util.find_spec("anthropic") is None
):  # pragma: no cover - optional dependency
    pytest.skip("anthropic package is not installed", allow_module_level=True)
