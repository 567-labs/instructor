"""Select GitHub Actions test jobs from the files changed by a pull request."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from scripts.run_unit_tests import UNIT_GROUPS, group_for_path


class Provider(NamedTuple):
    name: str
    env_key: str
    test_path: str
    family: str


PROVIDERS = (
    Provider("OpenAI", "OPENAI_API_KEY", "tests/llm/test_openai", "openai"),
    Provider("Anthropic", "ANTHROPIC_API_KEY", "tests/llm/test_anthropic", "anthropic"),
    Provider("Gemini", "GOOGLE_API_KEY", "tests/llm/test_gemini", "google"),
    Provider("Google GenAI", "GOOGLE_API_KEY", "tests/llm/test_genai", "google"),
    Provider("Vertex AI", "GOOGLE_API_KEY", "tests/llm/test_vertexai", "google"),
    Provider("Writer", "WRITER_API_KEY", "tests/llm/test_writer", "other"),
)
SHARDS = set(UNIT_GROUPS)
LIVE_JOBS = {
    "core-openai",
    "core-anthropic",
    "core-google",
    "core-other",
    "auto-client-test",
}
VERSIONS = ["3.10", "3.11", "3.12", "3.13"]
PROVIDER_FAMILY = {
    "anthropic": "anthropic",
    "gemini": "google",
    "genai": "google",
    "vertexai": "google",
    "cerebras": "other",
    "cohere": "other",
    "fireworks": "other",
    "mistral": "other",
    "perplexity": "other",
    "writer": "other",
    "xai": "other",
}


@dataclass
class TestPlan:
    shards: set[str]
    live_jobs: set[str]
    providers: set[str]


def full_plan() -> TestPlan:
    return TestPlan(SHARDS.copy(), LIVE_JOBS.copy(), {p.name for p in PROVIDERS})


def select_tests(paths: list[str]) -> TestPlan:
    plan = TestPlan(set(), set(), set())
    for path in paths:
        if (
            path.startswith(("docs/", "tests/docs/"))
            or path == "mkdocs.yml"
            or ("/" not in path and path.endswith(".md"))
        ):
            continue
        provider = next(
            (p for p in PROVIDERS if path.startswith(p.test_path + "/")), None
        )
        if provider:
            plan.providers.add(provider.name)
            continue
        if path.startswith("tests/llm/test_decisions/"):
            plan.live_jobs.add("core-openai")
            continue
        if path == "tests/llm/test_current_doc_models.py":
            plan.live_jobs.add("core-other")
            continue
        if path.startswith("tests/llm/test_core_providers/"):
            plan.live_jobs.update(LIVE_JOBS - {"auto-client-test"})
            continue
        if path.startswith("tests/") and not path.startswith("tests/llm/"):
            if not re.fullmatch(r"test_.*\.py|.*_test\.py", Path(path).name):
                return full_plan()
            plan.shards.add(group_for_path(path))
            if path == "tests/providers/test_auto_client.py":
                plan.live_jobs.add("auto-client-test")
            continue
        match = re.match(
            r"^instructor/(?:(?:v2/)?providers/([^/]+)/|batch/providers/([^/.]+)\.py$)",
            path,
        )
        family = PROVIDER_FAMILY.get((match[1] or match[2]) if match else "")
        if not family:
            return full_plan()
        plan.shards.update(SHARDS)
        plan.live_jobs.update(("core-" + family, "auto-client-test"))
        plan.providers.update(p.name for p in PROVIDERS if p.family == family)
    return plan


def shared_test_dependency(paths: list[str], root: Path) -> bool:
    def area(path: str) -> str:
        if path.startswith("tests/llm/"):
            return "/".join(path.split("/")[:3])
        if path.startswith("tests/docs/"):
            return "docs"
        return group_for_path(path)

    owners: dict[str, set[str]] = {}
    for path in paths:
        if path.startswith("tests/") and path.endswith(".py"):
            owners.setdefault(Path(path).stem, set()).add(area(path))
    if not owners:
        return False
    pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, owners)) + r")\b")
    for source in (root / "tests").rglob("*.py"):
        relative = source.relative_to(root).as_posix()
        for match in pattern.finditer(source.read_text(encoding="utf-8")):
            if (
                not re.fullmatch(r"test_.*\.py|.*_test\.py", source.name)
                or area(relative) not in owners[match[0]]
            ):
                return True
    return False


def verified_baseline(repository: str, sha: str) -> str:
    def read(endpoint: str):
        result = subprocess.run(
            ["gh", "api", "repos/" + repository + endpoint],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return json.loads(result.stdout)

    try:
        runs = read(
            f"/actions/workflows/test.yml/runs?event=push&branch=main&head_sha={sha}&status=completed&per_page=1"
        )["workflow_runs"]
        if not runs:
            return ""
        run = runs[0]
        if (
            run["head_sha"] != sha
            or run["event"] != "push"
            or run["head_branch"] != "main"
            or run["head_repository"]["full_name"] != repository
        ):
            return ""
        run_id = str(run["id"])
        jobs = read(f"/actions/runs/{run_id}/jobs?per_page=100")["jobs"]
        if any(
            j["name"] == "Full Coverage (Python 3.11)" and j["conclusion"] == "success"
            for j in jobs
        ):
            return run_id
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
        pass
    return ""


def usable_coverage(path: Path) -> bool:
    try:
        with closing(
            sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        ) as database:
            return (
                database.execute("PRAGMA quick_check").fetchone() == ("ok",)
                and database.execute(
                    "SELECT value FROM meta WHERE key = 'has_arcs'"
                ).fetchone()
                == ("1",)
                and database.execute(
                    "SELECT 1 FROM arc JOIN file ON file.id = arc.file_id LIMIT 1"
                ).fetchone()
                is not None
            )
    except sqlite3.Error:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-coverage", type=Path)
    parser.add_argument("--changed-paths", nargs="+")
    args = parser.parse_args()
    if args.verify_coverage is not None:
        raise SystemExit(int(not usable_coverage(args.verify_coverage)))

    base = os.environ.get("CI_BASE_SHA", "")
    try:
        pull_request = (
            os.environ["GITHUB_EVENT_NAME"] == "pull_request"
            and re.fullmatch(r"[0-9a-f]{40}", base)
            and subprocess.check_output(
                ["git", "rev-parse", "HEAD^1"], text=True
            ).strip()
            == base
        )
    except (OSError, subprocess.SubprocessError):
        pull_request = False
    if args.changed_paths:
        changed = (
            not pull_request
            or subprocess.run(
                ["git", "diff", "--quiet", base, "HEAD", "--", *args.changed_paths]
            ).returncode
            != 0
        )
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as out:
            print(f"changed={str(changed).lower()}", file=out)
        return

    plan = full_plan()
    baseline = ""
    if pull_request:
        try:
            raw = subprocess.check_output(
                ["git", "diff", "--no-renames", "--name-only", "-z", base, "HEAD"]
            )
            paths = [os.fsdecode(p) for p in raw.split(b"\0") if p]
            plan = select_tests(paths)
            if plan != full_plan() and shared_test_dependency(paths, Path.cwd()):
                plan = full_plan()
            elif plan.shards and plan.shards != SHARDS:
                baseline = verified_baseline(os.environ["GITHUB_REPOSITORY"], base)
        except (OSError, subprocess.SubprocessError, UnicodeError):
            plan = full_plan()

    reuse = sorted(SHARDS - plan.shards) if baseline else []
    outputs = {
        "unit": str(bool(plan.shards)).lower(),
        "reuse": json.dumps(reuse),
        "baseline": baseline,
        "versions": json.dumps(VERSIONS if "focused" in plan.shards else ["3.11"]),
        "live_jobs": json.dumps(sorted(plan.live_jobs)),
        "providers": json.dumps(
            [p._asdict() for p in PROVIDERS if p.name in plan.providers]
        ),
    }
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as out:
        for key, value in outputs.items():
            print(f"{key}={value}", file=out)
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as out:
        fresh = SHARDS - set(reuse) if plan.shards else set()
        print("Unit groups to run:", ", ".join(sorted(fresh)) or "none", file=out)
        if reuse:
            print(
                f"Eligible for verified coverage reuse from main run {baseline}: {', '.join(reuse)}. A missing artifact runs its group.",
                file=out,
            )
        print(
            "Live jobs selected:",
            ", ".join(sorted(plan.live_jobs | plan.providers)) or "none",
            file=out,
        )


if __name__ == "__main__":
    main()
