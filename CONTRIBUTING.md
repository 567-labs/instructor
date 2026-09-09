# Contributing to Instructor

Thank you for considering contributing to Instructor! This document provides guidelines and instructions to help you contribute effectively.

## Table of Contents

- [Contributing to Instructor](#contributing-to-instructor)
  - [Table of Contents](#table-of-contents)
  - [Code of Conduct](#code-of-conduct)
  - [Getting Started](#getting-started)
    - [Environment Setup](#environment-setup)
    - [Development Workflow](#development-workflow)
    - [Dependency Management](#dependency-management)
      - [Using UV](#using-uv)
      - [Using Poetry](#using-poetry)
    - [Working with Optional Dependencies](#working-with-optional-dependencies)
  - [How to Contribute](#how-to-contribute)
    - [Reporting Bugs](#reporting-bugs)
    - [Feature Requests](#feature-requests)
    - [Pull Requests](#pull-requests)
    - [Writing Documentation](#writing-documentation)
    - [Contributing to Evals](#contributing-to-evals)
  - [Code Style Guidelines](#code-style-guidelines)
    - [Conventional Comments](#conventional-comments)
    - [Conventional Commits](#conventional-commits)
      - [Types](#types)
      - [Examples](#examples)
  - [Testing](#testing)
  - [Branch and Release Process](#branch-and-release-process)
  - [Using Cursor for PR Creation](#using-cursor-for-pr-creation)
  - [License](#license)

## Code of Conduct

By participating in this project, you agree to abide by our code of conduct: treat everyone with respect, be constructive in your communication, and focus on the technical aspects of the contributions.

## Getting Started

### Environment Setup

1. **Fork the Repository**: Click the "Fork" button at the top right of the [repository page](https://github.com/567-labs/instructor).

2. **Clone Your Fork**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/instructor.git
   cd instructor
   ```

3. **Set up Remote**:
   ```bash
   git remote add upstream https://github.com/567-labs/instructor.git
   ```

4. **Install UV** (recommended):
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows PowerShell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

5. **Install Dependencies**:
   ```bash
   # Using uv (recommended)
   uv sync --extra dev --extra docs --extra test-docs
   
   # Using poetry
   poetry install --with dev,docs,test-docs
   
   # For specific providers, add the provider name as an extra
   # Example: uv sync --extra dev --extra docs --extra test-docs --extra anthropic
   ```

6. **Set up Pre-commit**:
   ```bash
   uv run pre-commit install
   ```

### Development Workflow

1. **Create a Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes and Commit**:
   ```bash
   git add .
   git commit -m "Your descriptive commit message"
   ```

3. **Keep Your Branch Updated**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

4. **Push Changes**:
   ```bash
   git push origin feature/your-feature-name
   ```

### Dependency Management

We support both UV and Poetry for dependency management. Choose the tool that works best for you:

#### Using UV

UV is a fast Python package installer and resolver. It's recommended for day-to-day development in Instructor.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project and development dependencies
uv sync --extra dev --extra docs

# Add a project dependency and update pyproject.toml plus uv.lock
uv add new-package
```

Key UV commands:
- `uv sync` - Install the project and synchronize the environment with `uv.lock`
- `uv sync --extra dev` - Install with a selected optional extra
- `uv add package-name` - Add a project dependency and update the lockfile
- `uv pip install package-name` - Install only into the current environment without changing project metadata
- `uv pip compile pyproject.toml -o requirements.txt` - Regenerate the committed requirements export
- `uv lock --check` - Verify that `uv.lock` matches `pyproject.toml`
- `uv self update` - Update UV to the latest version

#### Using Poetry

Poetry provides more comprehensive dependency management and packaging.

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies including development deps
poetry install --with dev,docs

# Add a new dependency
poetry add package-name

# Add a new development dependency
poetry add --group dev package-name
```

Key Poetry commands:
- `poetry shell` - Activate the virtual environment
- `poetry run python -m pytest` - Run commands within the virtual environment
- `poetry update` - Update dependencies to their latest versions

### Working with Optional Dependencies

Instructor uses optional dependencies to support different LLM providers. Provider-specific utilities live under `instructor/utils`. When adding integration for a new provider:

1. **Update pyproject.toml**: Add your provider's dependencies to both `[project.optional-dependencies]` and `[dependency-groups]`:

   ```toml
   [project.optional-dependencies]
   # Add your provider here
   my-provider = ["my-provider-sdk>=1.0.0,<2.0.0"]
   
   [dependency-groups]
   # Also add to dependency groups
   my-provider = ["my-provider-sdk>=1.0.0,<2.0.0"]
   ```

2. **Create Provider Client**: Implement your provider client in `instructor/clients/client_myprovider.py`

3. **Add Tests**: Create tests in `tests/llm/test_myprovider/`

4. **Document Installation**: Update the documentation to include installation instructions:
   ```
   # Install with your provider support
   uv add "instructor[my-provider]"
   # or
   poetry add "instructor[my-provider]"
   ```

5. **Create Provider Utilities and Handlers**:
   - Add a new module at `instructor/utils/myprovider.py`
   - Implement `reask` functions for validation errors and `handle_*` functions
     for formatting requests
   - Define `MYPROVIDER_HANDLERS` mapping `Mode` values to these functions

6. **Register the Provider**:
   - Add a value in `instructor/utils/providers.py` to the `Provider` enum
   - Extend `get_provider` with detection logic for your base URL

7. **Update `process_response.py`**:
   - Import your handler functions and include them in the `mode_handlers`
     dictionary so the library can route requests to your provider
   - `process_response.py` relies on these handlers to format arguments and
     parse results for each `Mode`

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue on [our issue tracker](https://github.com/567-labs/instructor/issues) with:

1. A clear, descriptive title
2. A detailed description including:
   - The `response_model` you are using
   - The `messages` you are using
   - The `model` you are using
   - Steps to reproduce the bug
   - The expected behavior and what went wrong
   - Your environment (Python version, OS, package versions)

### Feature Requests

For feature requests, please create an issue describing:

1. The problem your feature would solve
2. How your solution would work
3. Alternatives you've considered
4. Examples of how the feature would be used

### Pull Requests

1. **Create a Pull Request** from your fork to the main repository.
2. **Fill out the PR template** with details about your changes.
3. **Address review feedback** and make requested changes.
4. **Wait for CI checks** to pass.
5. Once approved, a maintainer will merge your PR.

### Writing Documentation

Documentation improvements are always welcome! Follow these guidelines:

1. Documentation is written in Markdown format in the `docs/` directory
2. When creating new markdown files, add them to `mkdocs.yml` under the appropriate section
3. Follow the existing hierarchy and structure
4. Use a grade 10 reading level (simple, clear language)
5. Include working code examples
6. Add links to related documentation

### Contributing to Evals

We encourage contributions to our evaluation tests:

1. Explore existing evals in the [evals directory](https://github.com/567-labs/instructor/tree/main/tests/llm)
2. Contribute new evals as pytest tests
3. Evals should test specific capabilities or edge cases of the library or models
4. Follow the existing patterns for structuring eval tests

## Code Style Guidelines

We use automated tools to maintain consistent code style:

- **Ruff**: For linting and formatting
- **ty**: For type checking
- **Black**: For code formatting (enforced by Ruff)

General guidelines:

- **Typing**: Use strict typing with annotations for all functions and variables
- **Imports**: Standard lib → third-party → local imports
- **Models**: Define structured outputs as Pydantic BaseModel subclasses
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Error Handling**: Use custom exceptions from exceptions.py, validate with Pydantic
- **Comments**: Docstrings for public functions, inline comments for complex logic

### Conventional Comments

We use conventional comments in code reviews and commit messages. This helps make feedback clearer and more actionable:

```
<label>: <subject>

<description>
```

Labels include:
- **praise:** highlights something positive
- **suggestion:** proposes a change or improvement
- **question:** asks for clarification
- **nitpick:** minor, trivial feedback that can be ignored
- **issue:** points out a specific problem that needs to be fixed
- **todo:** notes something to be addressed later
- **fix:** resolves an issue
- **refactor:** suggests reorganizing code without changing behavior
- **test:** suggests adding or improving tests

Examples:
```
suggestion: consider using Pydantic's validator for this check
This would ensure validation happens automatically when the model is created.

question: why is this approach used instead of async processing?
I'm wondering if there would be performance benefits.

fix: correct the type hint for the client parameter
The client should accept OpenAI instances, not strings.
```

For more details, see the [Conventional Comments specification](https://conventionalcomments.org/).

### Conventional Commits

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification for commit messages. This helps us generate changelogs and understand the changes at a glance.

The commit message should be structured as follows:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

#### Types

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Changes that do not affect the meaning of the code (white-space, formatting, etc)
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **perf**: A code change that improves performance
- **test**: Adding missing tests or correcting existing tests
- **build**: Changes that affect the build system or external dependencies
- **ci**: Changes to our CI configuration files and scripts

#### Examples

```
feat(openai): add support for response_format parameter

fix(anthropic): correct tool calling format in Claude client

docs: improve installation instructions for various providers

test(evals): add evaluation for recursive schema handling
```

Breaking changes should be indicated by adding `!` after the type/scope:

```
feat(api)!: change parameter order in from_openai factory function
```

Including a scope is recommended when changes affect a specific part of the codebase (e.g., a specific provider, feature, or component).

## Testing

Run tests using pytest:

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/path_to_test.py::test_name

# Skip LLM tests (faster for local development)
uv run pytest tests/ -k 'not llm and not openai'

# Generate coverage report
uv run coverage run -m pytest tests/ -k "not docs"
uv run coverage report
```

## Branch and Release Process

- `main` branch is the development branch
- Releases are tagged with version numbers
- We follow [Semantic Versioning](https://semver.org/)

### Installed-package compatibility

Run `PACKAGE_PYTHON=3.11 tests/packaging/check_package.sh` with uv installed.
This builds an sdist, builds its wheel, and installs the wheel in a temporary
environment outside the checkout. It checks dependency consistency, the typing
marker, public imports and selected legacy aliases, model/schema helpers, real
OpenAI sync/async client construction without requests, and CLI help with provider
credentials removed. File and job clients are constructed only when an operation
needs them.
Temporary environments are removed on exit; resolved versions appear in the log.

The package compatibility workflow has five bounded combinations:

| Python | SDK selection | Extras |
| --- | --- | --- |
| 3.9 | OpenAI 2.0.0 and Pydantic 2.8.0 (declared floors) | None |
| 3.9 | Latest compatible | None |
| 3.11 | Latest compatible | None |
| 3.14 | Latest compatible | None |
| 3.11 | Latest compatible | google-genai |

Use `PACKAGE_SDK=minimum` for the floor check, or `PACKAGE_EXTRAS=google-genai`
for the optional import check. Latest means a fresh resolution within the package
requirements and the selected Python version, not the lockfile or necessarily the
newest upstream release. The minimum row pins only OpenAI and Pydantic; their
transitive dependencies, including the matching pydantic-core, resolve normally.
Pydantic 2.8 predates Python 3.14, so these are deliberately not a Cartesian matrix.

This is packaging evidence, not provider behavioral certification. Other optional
extras and their minimum SDK versions, intermediate Python versions, Windows and
Linux/macOS differences are not exhaustively covered here. CI runs on Linux;
local runs describe their own platform. No provider requests are made. The existing
all-extras and provider tests supply separate evidence; declared support is unchanged.

### GitHub release and PyPI publication are separate

`scheduled-release.yml` creates a GitHub release only when explicitly dispatched
with `publish=true`, using `github.token`. [GitHub suppresses downstream workflow
runs for release events created with that token](https://docs.github.com/en/actions/concepts/security/github_token).
Therefore it does **not** trigger `python-publish.yml`, which listens only for
`release: published`. A successful GitHub release job is not evidence of a PyPI
upload; check both workflow runs and registry state.

The handoff needs a separately reviewed change that preserves publication approval,
tested artifact identity, and duplicate-publication protection. Do not create test
tags/releases or broaden permissions to exercise it. Until then, an approved manual
release process must explicitly account for event authentication and the tested
attached artifacts.

## Using Cursor for PR Creation

Cursor (https://cursor.sh) is a code editor powered by AI that can help you create PRs efficiently. We encourage using Cursor for Instructor development:

1. **Install Cursor**: Download from [cursor.sh](https://cursor.sh/)

2. **Create a Branch**: Start a new branch for your feature using Cursor's Git integration

3. **Use Cursor Rules**: We have Cursor rules that help with standards:
   - `new-features-planning`: Use when implementing new features
   - `simple-language`: Follow when writing documentation
   - `documentation-sync`: Reference when making code changes to keep docs in sync

4. **Generate Code with AI**: Use Cursor's AI assistance to generate code that follows our style

5. **Auto-Create PRs**: Use Cursor's PR creation feature with our template:
   ```
   # Create PR using gh CLI
   gh pr create -t "Your PR Title" -b "Description of changes" -r jxnl,ivanleomk
   ```

6. **Include Attribution**: Add `This PR was written by [Cursor](https://cursor.sh)` to your PR description

For more details, see our Cursor rules in `.cursor/rules/`.

## License

By contributing to Instructor, you agree that your contributions will be licensed under the project's MIT License. 
