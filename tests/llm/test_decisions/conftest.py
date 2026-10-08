from __future__ import annotations

import os

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-decisions-live",
        action="store_true",
        default=False,
        help="Run paid TypeSafe/OpenRouter/OpenAI decisions contracts using configured keys.",
    )


@pytest.fixture(
    params=[
        ("typesafe", "jev-latest", "TYPESAFE_API_KEY", "TYPESAFE_DECISIONS_MODEL"),
        (
            "openrouter",
            "typesafe/jev-1.13",
            "OPENROUTER_API_KEY",
            "OPENROUTER_DECISIONS_MODEL",
        ),
        ("openai", "gpt-6-luna", "OPENAI_API_KEY", "OPENAI_DECISIONS_MODEL"),
    ],
    ids=["typesafe", "openrouter", "openai"],
)
def decisions_provider(request: pytest.FixtureRequest) -> tuple[str, str]:
    if not request.config.getoption("run_decisions_live"):
        reason = "Live decisions calls require --run-decisions-live"
        pytest.skip(reason)  # ty: ignore[too-many-positional-arguments]
    provider, default_model, key_name, model_name = request.param
    if not os.getenv(key_name):
        reason = f"Live {provider} contract requires {key_name}"
        pytest.skip(reason)  # ty: ignore[too-many-positional-arguments]
    return provider, os.getenv(model_name, default_model)
