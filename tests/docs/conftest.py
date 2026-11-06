from __future__ import annotations

from pathlib import Path

import pytest
from pytest_examples import CodeExample, EvalExample


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("docs")
    group.addoption(
        "--run-doc-examples",
        action="store_true",
        help="Execute doc code examples (requires network access and API keys).",
    )


@pytest.fixture(name="eval_example")
def eval_example(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    _examples_to_update: list[CodeExample],
):
    eval_ex = EvalExample(tmp_path=tmp_path, pytest_request=request)
    run_live = bool(
        request.config.getoption("run_doc_examples", default=False)
        or request.config.getoption("update_examples", default=False)
    )
    if not run_live:

        def _skip_run(_example: CodeExample) -> None:
            return None

        eval_ex.run = _skip_run  # type: ignore[assignment]
        eval_ex.run_print_update = _skip_run  # type: ignore[assignment]

    yield eval_ex

    if request.config.getoption("update_examples", default=False):
        _examples_to_update.extend(eval_ex.to_update)
