from pathlib import Path

import pytest
from dotenv import load_dotenv

# Support .env for local development
load_dotenv()


def pytest_configure(config: pytest.Config) -> None:
    # LiteLLM imports must use the package's bundled pricing data in tests.
    environment = pytest.MonkeyPatch()
    environment.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    config.add_cleanup(environment.undo)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    live_tests = Path(__file__).resolve().parent / "llm"
    for item in items:
        if (
            item.path.is_relative_to(live_tests)
            or item.get_closest_marker("requires_api_key")
        ) and item.get_closest_marker("llm") is None:
            item.add_marker(pytest.mark.llm)
