# AGENT.md

## Commands
- Install: `uv pip install -e ".[dev]"` or `poetry install --with dev`
- Run tests: `uv run pytest tests/`
- Run single test: `uv run pytest tests/path_to_test.py::test_name`
- Skip LLM tests: `uv run pytest tests/ -k 'not llm and not openai'`
- Temp deps for a run: `uv run --with <pkg>[==version] <command>` (example: `uv run --with pytest-asyncio --with anthropic pytest tests/...`)
- Type check: `uv run ty check`
- Lint: `uv run ruff check instructor examples tests`
- Format: `uv run ruff format instructor examples tests`
- Build docs: `uv run mkdocs serve` (local) or `./build_mkdocs.sh` (production)
- Waiting: use `sleep <seconds>` for explicit pauses (e.g., CI waits) or to let external processes finish

## Architecture
- **Core**: `instructor/` - Pydantic-based structured outputs for LLMs
- **Base classes**: `Instructor` and `AsyncInstructor` in `client.py`
- **Providers**: Client files (`client_*.py`) for OpenAI, Anthropic, Gemini, Cohere, etc.
- **Factory pattern**: `from_provider()` for automatic provider detection
- **DSL**: `dsl/` directory with Partial, Iterable, Maybe, Citation extensions
- **Key modules**: `patch.py` (patching), `process_response.py` (parsing), `function_calls.py` (schemas)
- **GenAI request ownership**: `_clone_kwargs` copies nested `generation_config` before either mode handler merges and translates sampling options. Keep caller-owned configurations reusable; regression tests are in `tests/v2/test_genai_config_reuse.py`.

## Code Style
- **Typing**: Strict type annotations, use `BaseModel` for structured outputs
- **Imports**: Standard lib → third-party → local
- **Formatting**: Ruff with Black conventions
- **Error handling**: Custom exceptions from `exceptions.py`, Pydantic validation
- **Naming**: `snake_case` functions/variables, `PascalCase` classes
- **No mocking**: Tests use real API calls
- **Client creation**: Always use `instructor.from_provider("provider_name/model_name")` instead of provider-specific methods like `from_openai()`, `from_anthropic()`, etc.

## Pull Request (PR) Formatting

Use **Conventional Commits** formatting for PR titles. Treat the PR title as the message we would use for a squash merge commit.

### PR Title Format

Use:

`<type>(<scope>): <short summary>`

Rules:
- Keep it under ~70 characters when you can.
- Use the imperative mood (for example, “add”, “fix”, “update”).
- Do not end with a period.
- If it includes a breaking change, add `!` after the type or scope (for example, `feat(api)!:`).

Good examples:
- `fix(openai): handle empty tool_calls in streaming`
- `feat(retry): add backoff for JSON parse failures`
- `docs(agents): add conventional commit PR title guidelines`
- `test(schema): cover nested union edge cases`
- `ci(ruff): enforce formatting in pre-commit`

Common types:
- `feat`: new feature
- `fix`: bug fix
- `docs`: documentation-only changes
- `refactor`: code change that is not a fix or feature
- `perf`: performance improvement
- `test`: add or update tests
- `build`: build system or dependency changes
- `ci`: CI pipeline changes
- `chore`: maintenance work

Suggested scopes (pick the closest match):
- Providers: `openai`, `anthropic`, `gemini`, `vertexai`, `bedrock`, `mistral`, `groq`, `writer`
- Core: `core`, `patch`, `process_response`, `function_calls`, `retry`, `dsl`
- Repo: `docs`, `examples`, `tests`, `ci`, `build`

### PR Description Guidelines

Keep PR descriptions short and easy to review:
- **What**: What changed, in 1–3 sentences.
- **Why**: Why this change is needed (link issues when possible).
- **Changes**: 3–7 bullet points with the main edits.
- **Testing**: What you ran (or why you did not run anything).

If the PR was authored by Cursor, include:
- `This PR was written by [Cursor](https://cursor.com)`

### Changelog Requirement

**Every PR that changes behavior must update `CHANGELOG.md`.**

Add an entry under the `## [Unreleased]` section (or the current in-progress version):

```
- **Area**: Short description of the change ([#PR_NUMBER](url))
```

Group entries under: `Security`, `Fixed`, `Added`, `Changed`, `Deprecated`, `Removed`, `Tests / CI`.

Do not add changelog entries for docs-only or example-only changes unless they fix something user-visible.

## Release Process

Follow the [release checklist](CONTRIBUTING.md#github-release-and-pypi-publication-are-separate).
Keep the version in `pyproject.toml`, `instructor/__init__.py`, and `uv.lock` in
sync and prepare a dated changelog section. Merge the reviewed changes into
`main` once CI passes.

Only publish with maintainer approval. Dispatch **Release Readiness** from
`main` with `expected_version=X.Y.Z` and `publish=true`. Let it create the tag
and GitHub release with the tested artifacts. Then dispatch **Upload Python
Package** from `main` with `tag=vX.Y.Z` and verify the version on PyPI.

Version bump rules (based on commits since last tag):
- `feat!:` / `fix!:` / `BREAKING` → major
- `feat:` → minor
- `fix:` / `chore:` / everything else → patch
