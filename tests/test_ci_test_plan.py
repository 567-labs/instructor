import json
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest

from scripts import ci_test_plan as ci


class PlanCase(NamedTuple):
    paths: tuple[str, ...]
    shards: tuple[str, ...] = ()
    live: tuple[str, ...] = ()
    providers: tuple[str, ...] = ()


CASES = {
    "docs": PlanCase(("docs/guide.md", "README.md", "tests/docs/test_guide.py")),
    "core-unit": PlanCase(("tests/test_hooks.py",), ("core",)),
    "two-unit-groups": PlanCase(
        ("tests/v2/test_parser.py", "tests/coverage/test_batch.py"), ("v2", "focused")
    ),
    "mixed-auto-client": PlanCase(
        ("tests/providers/test_auto_client.py",), ("core",), ("auto-client-test",)
    ),
    "one-live-provider": PlanCase(
        ("tests/llm/test_anthropic/test_tools.py",), providers=("Anthropic",)
    ),
    "shared-live-tests": PlanCase(
        ("tests/llm/test_core_providers/test_client.py",),
        live=tuple(ci.LIVE_JOBS - {"auto-client-test"}),
    ),
    "google-runtime": PlanCase(
        ("instructor/v2/providers/genai/client.py",),
        tuple(ci.SHARDS),
        ("core-google", "auto-client-test"),
        ("Gemini", "Google GenAI", "Vertex AI"),
    ),
}


@pytest.mark.parametrize("config", CASES.values(), ids=CASES.keys())
def test_selects_affected_suites(config: PlanCase) -> None:
    assert ci.select_tests(list(config.paths)) == ci.TestPlan(
        set(config.shards), set(config.live), set(config.providers)
    )


@pytest.mark.parametrize(
    "path",
    [
        "instructor/v2/providers/openai/client.py",
        "tests/conftest.py",
        "uv.lock",
        "scripts/new_tool.py",
    ],
    ids=[
        "shared-openai-gateways",
        "shared-fixture",
        "dependencies",
        "unrecognized-path",
    ],
)
def test_shared_inputs_run_all_suites(path: str) -> None:
    assert ci.select_tests([path]) == ci.full_plan()


@pytest.mark.parametrize(
    ("source", "consumer"),
    [
        ("tests/test_source.py", "tests/conftest.py"),
        ("tests/test_source.py", "tests/v2/test_consumer.py"),
        ("tests/llm/test_openai/provider_case.py", "tests/test_shared_runtime.py"),
    ],
    ids=["shared-fixture", "another-shard", "live-helper-used-by-unit"],
)
def test_cross_suite_import_disables_reuse(
    tmp_path: Path, source: str, consumer: str
) -> None:
    path = tmp_path / consumer
    path.parent.mkdir(parents=True)
    path.write_text(
        f"from {source.removesuffix('.py').replace('/', '.')} import sample"
    )
    assert ci.shared_test_dependency([source], tmp_path)


@pytest.mark.parametrize(
    ("head", "event", "conclusion", "expected"),
    [
        ("base", "push", "success", "42"),
        ("different", "push", "success", ""),
        ("base", "push", "failure", ""),
        ("base", "pull_request", "success", ""),
    ],
    ids=[
        "verified-base",
        "different-commit",
        "failed-coverage",
        "pull-request-artifact",
    ],
)
def test_baseline_requires_success_on_the_exact_base(
    monkeypatch, head, event, conclusion, expected
):
    def api(command, **_kwargs):
        response = (
            {
                "jobs": [
                    {"name": "Full Coverage (Python 3.11)", "conclusion": conclusion}
                ]
            }
            if "/jobs?" in command[-1]
            else {
                "workflow_runs": [
                    {
                        "id": 42,
                        "head_sha": head,
                        "event": event,
                        "head_branch": "main",
                        "head_repository": {"full_name": "owner/repo"},
                    }
                ]
            }
        )
        return SimpleNamespace(stdout=json.dumps(response))

    monkeypatch.setattr(ci.subprocess, "run", api)
    assert ci.verified_baseline("owner/repo", "base") == expected
