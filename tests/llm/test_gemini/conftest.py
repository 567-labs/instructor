import os
import pytest

if not os.getenv("GOOGLE_API_KEY"):
    pytest.skip("GOOGLE_API_KEY environment variable not set", allow_module_level=True)

if not os.getenv("GOOGLE_GENAI_MODEL"):
    pytest.skip(
        "GOOGLE_GENAI_MODEL environment variable not set",
        allow_module_level=True,
    )

try:
    from google import genai  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("google-genai package is not installed", allow_module_level=True)
