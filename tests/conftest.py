from pathlib import Path

import pytest
from dotenv import load_dotenv

# Support .env for local development
load_dotenv()


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    live_tests = Path(__file__).resolve().parent / "llm"
    for item in items:
        if (
            item.path.is_relative_to(live_tests)
            and item.get_closest_marker("llm") is None
        ):
            item.add_marker(pytest.mark.llm)
