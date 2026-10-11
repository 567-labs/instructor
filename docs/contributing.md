---
title: Contribute to Instructor: Evals, Issues, and Pull Requests
description: Join us in enhancing the Instructor library with evals, report issues, and submit pull requests on GitHub. Collaborate and contribute!
---

# Contributing to Instructor

We welcome contributions to Instructor! This page covers the different ways you can help improve the library.

## Ways to Contribute

### Evaluation Tests (Evals)

Evals help us monitor the quality of both the OpenAI models and the Instructor library. To contribute:

1. **Explore Existing Evals**: Check out [our evals directory](https://github.com/567-labs/instructor/tree/main/tests/llm)
2. **Create a New Eval**: Add new pytest tests that evaluate specific capabilities or edge cases
3. **Follow the Pattern**: Structure your eval similar to existing ones
4. **Submit a PR**: We'll review and incorporate your eval

Evals are run weekly, and results are tracked to monitor performance over time.

### Reporting Issues

If you encounter a bug or problem, please [file an issue on GitHub](https://github.com/567-labs/instructor/issues) with:

1. A clear, descriptive title
2. Detailed information including:
   - The `response_model` you're using
   - The `messages` you sent
   - The `model` you're using
   - Steps to reproduce the issue
   - Expected vs. actual behavior
   - Your environment details (Python version, OS, package versions)

### Contributing Code

We welcome pull requests! Here's the process:

1. **For Small Changes**: Feel free to submit a PR directly
2. **For Larger Changes**: [Start with an issue](https://github.com/567-labs/instructor/issues) to discuss approach
3. **Looking for Ideas?** Check issues labeled [help wanted](https://github.com/567-labs/instructor/labels/help%20wanted) or [good first issue](https://github.com/567-labs/instructor/labels/good%20first%20issue)

## Setting Up Your Development Environment

### Using UV (Recommended)

UV is a fast Python package installer and resolver that makes development easier.

1. **Install UV** (official method):
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows PowerShell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Install Project in Development Mode**:
   ```bash
   # Clone the repository
   git clone https://github.com/YOUR-USERNAME/instructor.git
   cd instructor

   # Install with development dependencies
   uv sync --extra dev --extra docs
   ```

3. **Adding New Dependencies**:
   ```bash
   # Add a project dependency and update pyproject.toml plus uv.lock
   uv add some-package

   # Install only into the current environment without changing project metadata
   uv pip install some-package
   ```

4. **Common UV Commands**:
   ```bash
   # Update UV itself
   uv self update

   # Verify the lockfile and regenerate the committed requirements export
   uv lock --check
   uv pip compile pyproject.toml -o requirements.txt
   ```

### Using Poetry

Poetry provides comprehensive dependency management and packaging.

1. **Install Poetry**:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Install Dependencies**:
   ```bash
   # Clone the repository
   git clone https://github.com/YOUR-USERNAME/instructor.git
   cd instructor

   # Install with development dependencies
   poetry install --with dev,docs
   ```

3. **Working with Poetry**:
   ```bash
   # Activate virtual environment
   poetry shell

   # Run a command in the virtual environment
   poetry run pytest

   # Add a dependency
   poetry add package-name

   # Add a development dependency
   poetry add --group dev package-name
   ```

## Adding Support for New LLM Providers

Instructor uses optional dependencies to support different LLM providers. Provider clients and handlers live under `instructor/v2/providers/`. To add a new provider:

1. **Add Dependencies to `pyproject.toml`**: Add the SDK to both optional dependencies and dependency groups:

   ```toml
   [project.optional-dependencies]
   my-provider = ["my-provider-sdk>=1.0.0,<2.0.0"]

   [dependency-groups]
   my-provider = ["my-provider-sdk>=1.0.0,<2.0.0"]
   ```

2. **Add Provider Identity and Client**:
   - Add a value to `Provider` in `instructor/v2/core/providers.py`; extend `get_provider` if the provider can be detected from its base URL
   - Implement `from_myprovider` in `instructor/v2/providers/myprovider/client.py`, with sync/async support as applicable
   - Follow an existing provider package, such as `instructor/v2/providers/perplexity/`

3. **Implement Mode Handlers**:
   - Add `handlers.py` in the provider package for request preparation, validation retries and response parsing
   - Register handler classes with `register_mode_handler` from `instructor/v2/core/decorators.py`
   - Use the supported modes in `instructor/v2/core/mode.py`; do not assume every provider supports every mode

4. **Declare Provider Capabilities**:
   - Add an entry to `PROVIDER_SPECS` in `instructor/v2/core/provider_specs.py` with the SDK module, aliases, client factory, handler module and supported modes
   - `HANDLER_SPECS` is derived from this manifest and used by `instructor/v2/core/registry.py` to load handlers lazily

5. **Wire Public Factories and `from_provider`**:
   - The manifest supplies `PUBLIC_FACTORY_ATTRS` for `instructor/v2/__init__.py`; also update the explicit exports in `instructor/__init__.py`, keeping optional SDK imports lazy
   - Add the provider builder to `_PROVIDER_BUILDERS` in `instructor/v2/auto_client.py` for `from_provider` support. Its supported aliases come from `ALIAS_TO_PROVIDER`, but the builder mapping must be updated separately

6. **Add Real-API Tests and CI Coverage**:
   - Add the provider/model, supported mode, credential environment variable and SDK import path to `PROVIDER_CONFIGS` in `tests/llm/shared_config.py`. See `tests/llm/test_core_providers/README.md` for the shared extraction, streaming and validation/retry tests
   - Add tests in `tests/llm/test_myprovider/` for provider-specific behavior not covered by the shared suite. Use real API calls, not mocked provider responses
   - Update the credentialed jobs in `.github/workflows/test.yml`. For shared tests, update the appropriate job (such as `core-other`), including secret exports, `PROVIDER_TEST_REQUIREMENTS`, credential skip/run conditions and the initial/retry `pytest -k` filters
   - For a provider-specific test directory, add a `provider-tests` matrix entry with `name`, `env_key` and `test_path`. Coordinate repository secrets with maintainers; tests skipped for missing credentials or not collected do not establish provider coverage

7. **Document Installation**:

   ```bash
   uv add "instructor[my-provider]"
   # or with poetry
   poetry add "instructor[my-provider]"
   ```

8. **Write Integration Documentation**:
   - Add a page in `docs/integrations/` with installation, a complete example and supported modes
   - Add the page to `mkdocs.yml` navigation

## Development Workflow

1. **Fork the Repository**: Create your own fork of the project
2. **Clone and Set Up**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/instructor.git
   cd instructor
   git remote add upstream https://github.com/567-labs/instructor.git
   ```
3. **Create a Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make Changes, Test, and Commit**:
   ```bash
   # Run tests
   uv run pytest tests/ -k 'not llm and not openai'  # Skip LLM tests for faster local dev

   # Commit changes
   git add .
   git commit -m "Your descriptive commit message"
   ```
5. **Keep Updated and Push**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   git push origin feature/your-feature-name
   ```
6. **Create a Pull Request**: Submit your PR with a clear description of changes

## Utility Scripts

The `scripts/` directory contains utility scripts that help maintain code quality and documentation. These scripts are integrated into pre-commit hooks and can also be run manually.

### Available Scripts

#### `make_clean.py` - Markdown File Cleaner
Cleans markdown files by removing special whitespace characters and replacing em dashes with regular dashes.

```bash
# Clean all markdown files
python scripts/make_clean.py

# Preview changes without modifying files
python scripts/make_clean.py --dry-run
```

#### `check_blog_excerpts.py` - Blog Post Excerpt Validator
Ensures all blog posts contain the `<!-- more -->` tag for proper excerpt handling.

```bash
# Check all blog posts
python scripts/check_blog_excerpts.py
```

#### `make_sitemap.py` - Enhanced Documentation Sitemap Generator
Generates an enhanced sitemap (`sitemap.yaml`) with AI-powered content analysis and cross-link suggestions.

```bash
# Generate sitemap with default settings
python scripts/make_sitemap.py

# Customize settings
python scripts/make_sitemap.py \
  --root-dir docs \
  --output-file sitemap.yaml \
  --max-concurrency 10
```

**Requirements for sitemap generation**:
- OpenAI API key (set as `OPENAI_API_KEY` environment variable)
- Additional dependencies: `openai`, `typer`, `rich`, `tenacity`, `pyyaml`

### Pre-commit Integration

These scripts run automatically during the commit process:

- **Markdown cleaning**: Runs on commits with markdown files in `docs/`
- **Blog excerpt validation**: Runs on commits with blog post files

### Manual Usage

You can run scripts manually for testing or one-time operations:

```bash
# Test markdown cleaning
python scripts/make_clean.py --dry-run

# Check blog excerpts
python scripts/check_blog_excerpts.py

# Generate fresh sitemap
python scripts/make_sitemap.py
```

For detailed documentation on each script, see the `scripts/README.md` file in the project repository.

## Using Cursor to Build PRs

[Cursor](https://cursor.sh) is an AI-powered code editor that can help you contribute to Instructor.

1. **Getting Started with Cursor**:
   - Download Cursor from [cursor.sh](https://cursor.sh)
   - Open the Instructor project in Cursor
   - Cursor will automatically detect our rules in `.cursor/rules/`

2. **Using Cursor Rules**:
   - `new-features-planning`: Helps plan and structure new features
   - `simple-language`: Guidelines for writing clear documentation
   - `documentation-sync`: Ensures documentation stays in sync with code changes

3. **Creating PRs with Cursor**:
   - Use Cursor's Git integration to create a new branch
   - Make your changes with AI assistance
   - Create a PR with:
     ```bash
     # Use GitHub CLI to create the PR
     gh pr create -t "Your feature title" -b "Description of your changes" -r jxnl,ivanleomk
     ```
   - Add `This PR was written by [Cursor](https://cursor.sh)` to your PR description

4. **Benefits of Using Cursor**:
   - AI helps generate code that follows our style guidelines
   - Simplifies PR creation process
   - Helps maintain documentation standards

## Code Style Guidelines

We use the following tools to maintain code quality:

- **Ruff**: For linting and formatting
- **ty**: For type checking
- **Pre-commit**: For automatic checks before committing

```bash
# Install pre-commit hooks
uv run pre-commit install
```

Key style guidelines:
- Use strict typing
- Follow import order: standard lib → third-party → local
- Use snake_case for functions/variables, PascalCase for classes
- Write comprehensive docstrings for public API functions

### Conventional Comments

When reviewing code or writing commit messages, we use conventional comments to make feedback clearer:

```
<label>: <subject>

<description>
```

Common labels:
- **praise:** highlights something positive
- **suggestion:** proposes a change or improvement
- **question:** asks for clarification
- **issue:** points out a problem that needs fixing
- **todo:** notes something to be addressed later
- **fix:** resolves an issue

Examples:

```
suggestion: use a validator for this field
This would ensure the value is always properly formatted.

question: why not use async processing here?
I'm curious if this would improve performance.

fix: correct the parameter type
It should be an OpenAI client instance, not a string.
```

This format helps everyone understand the purpose and importance of each comment. Visit [conventionalcomments.org](https://conventionalcomments.org/) to learn more.

### Conventional Commits

We use conventional commit messages to make our project history clear and generate automated changelogs. A conventional commit has this structure:

```
<type>[optional scope]: <description>

[optional body]

[optional footer]
```

#### Common Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Formatting changes
- **refactor**: Code change that neither fixes a bug nor adds a feature
- **test**: Adding or fixing tests
- **chore**: Maintenance tasks

#### Examples

```
feat(openai): add streaming response support

fix(anthropic): resolve tool calling response format

docs: update installation instructions

test(evals): add new recursive schema test cases
```

For breaking changes, add an exclamation mark before the colon:

```
feat(api)!: change return type of from_openai function
```

Using conventional commits helps automatically generate release notes and makes the project history easier to navigate.

For more details, see the [Conventional Commits specification](https://www.conventionalcommits.org/).

## Documentation Contributions

Documentation improvements are highly valued:

1. **Docs Structure**: All documentation is in Markdown in the `docs/` directory
2. **Adding New Pages**: When adding a new page, include it in `mkdocs.yml` in the right section
3. **Local Preview**: Run `mkdocs serve` to preview changes locally
4. **Style Guidelines**:
   - Write at a grade 10 reading level (simple, clear language)
   - Include working code examples
   - Add links to related documentation
   - Use consistent formatting
   - Make sure each code example is complete with imports

Example of a good documentation code block:

```python
# Complete example with imports
import instructor
from pydantic import BaseModel
# Define your model
class Person(BaseModel):
    name: str
    age: int

# Create the patched client
client = instructor.from_provider("openai/gpt-5-nano")

# Use the model
person = client.create(
    model="gpt-5.4-mini",
    response_model=Person,
    messages=[
        {"role": "user", "content": "Extract: John Doe is 25 years old"}
    ]
)

print(person.name)  # "John Doe"
print(person.age)   # 25
```

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

<a href="https://github.com/567-labs/instructor/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=567-labs/instructor" />
</a>

## Documentation Resources

When working on documentation, these resources may be helpful:

- **mkdocs serve**: Preview documentation locally. Install dependencies from `requirements-doc.txt` first.

- **hl_lines in Code Blocks**: Highlight specific lines in a code block to draw attention:
  ````markdown
  ```python hl_lines="2 3"
  def example():
      # This line is highlighted
      # This line is also highlighted
      return "normal line"
  ```
  ````

- **Admonitions**: Create styled callout boxes for important information:
  ```markdown
  !!! note "Optional Title"
      This is a note admonition.

  !!! warning
      This is a warning.
  ```

For more documentation features, check the [MkDocs Material documentation](https://squidfunk.github.io/mkdocs-material/).

Thank you for your contributions to Instructor!
