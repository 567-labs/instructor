import os
import pytest

if not os.getenv("WRITER_API_KEY"):
    pytest.skip("WRITER_API_KEY environment variable not set", allow_module_level=True)

try:
    import writerai  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("writer-sdk package is not installed", allow_module_level=True)


@pytest.fixture(scope="session", autouse=True)
def configure_writer():
    pass
