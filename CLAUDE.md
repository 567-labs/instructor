# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Instructor Development Guide

## Commands
- Install deps: `uv pip install -e ".[dev,anthropic]"` or `poetry install --with dev,anthropic`
- Run tests: `uv run pytest tests/ -n auto`
- Run specific test: `uv run pytest tests/path_to_test.py::test_name`
- Skip LLM tests: `uv run pytest tests/ -k 'not llm and not openai'`
- Type check: `uv run ty check`
- Lint: `uv run ruff check instructor examples tests`
- Format: `uv run ruff format instructor examples tests`
- Generate coverage: `uv run coverage run -m pytest tests/ -k "not docs"` then `uv run coverage report`
- Build documentation: `uv run mkdocs serve` (for local preview) or `./build_mkdocs.sh` (for production)
- Waiting: use `sleep <seconds>` for explicit pauses (e.g., CI waits) or to let external processes finish

## Installation & Setup
- Fork the repository and clone your fork
- Install UV: `pip install uv`
- Create virtual environment: `uv venv`
- Install dependencies: `uv pip install -e ".[dev]"`
- Install pre-commit: `uv run pre-commit install`
- Run tests to verify: `uv run pytest tests/ -k "not openai"`

## Code Style Guidelines
- **Typing**: Use strict typing with annotations for all functions and variables
- **Imports**: Standard lib → third-party → local imports
- **Formatting**: Follow Black's formatting conventions (enforced by Ruff)
- **Models**: Define structured outputs as Pydantic BaseModel subclasses
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Error Handling**: Use custom exceptions from exceptions.py, validate with Pydantic
- **Comments**: Docstrings for public functions, inline comments for complex logic

## Conventional Commits
- **Format**: `type(scope): description`
- **Types**: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert
- **Examples**:
  - `feat(anthropic): add support for Claude 3.5`
  - `fix(openai): correct response parsing for streaming`
  - `docs(README): update installation instructions`
  - `test(gemini): add validation tests for JSON mode`

## Core Architecture
- **Base Classes**: `Instructor` and `AsyncInstructor` in client.py are the foundation
- **Factory Pattern**: Provider-specific factory functions (`from_openai`, `from_anthropic`, etc.)
- **Unified Access**: `from_provider()` function in auto_client.py for automatic provider detection
- **Mode System**: `Mode` enum categorizes different provider capabilities (tools vs JSON output)
- **Patching Mechanism**: Uses Python's dynamic nature to patch provider clients for structured outputs
- **Response Processing**: Transforms raw API responses into validated Pydantic models
- **DSL Components**: Special types like Partial, Iterable, Maybe extend the core functionality

## Provider Architecture
- **Supported Providers**: OpenAI, Anthropic, Gemini, Cohere, Mistral, Groq, VertexAI, Fireworks, Cerebras, Writer, Databricks, Anyscale, Together, LiteLLM, Bedrock, Perplexity
- **Provider Implementation**: Each provider has a dedicated client file (e.g., `client_anthropic.py`) with factory functions
- **Modes**: Different providers support specific modes (`Mode` enum): `ANTHROPIC_TOOLS`, `GEMINI_JSON`, etc.
- **Common Pattern**: Factory functions (e.g., `from_anthropic`) take a native client and return patched `Instructor` instances
- **Provider Testing**: Tests in `tests/llm/` directory, define Pydantic models, make API calls, verify structured outputs
- **Provider Detection**: `get_provider` function analyzes base URL to detect which provider is being used

## Key Components
- **process_response.py**: Handles parsing and converting LLM outputs to Pydantic models
- **patch.py**: Contains the core patching logic for modifying provider clients
- **function_calls.py**: Handles generating function/tool schemas from Pydantic models
- **hooks.py**: Provides event hooks for intercepting various stages of the LLM request/response cycle
- **dsl/**: Domain-specific language extensions for specialized model types
- **retry.py**: Implements retry logic for handling validation failures
- **validators.py**: Custom validation mechanisms for structured outputs

## Testing Guidelines
- Tests are organized by provider under `tests/llm/`
- Each provider has its own conftest.py with fixtures
- Standard tests cover: basic extraction, streaming, validation, retries
- Evaluation tests in `tests/llm/test_provider/evals/` assess model capabilities
- Use parametrized tests when testing similar functionality across variants
- **IMPORTANT**: No mocking in tests - tests make real API calls

## Documentation Guidelines
- Every provider needs documentation in `docs/integrations/` following standard format
- Provider docs should include: installation, basic example, modes supported, special features
- When adding a new provider, update `mkdocs.yml` navigation and redirects
- Example code should include complete imports and environment setup
- Tutorials should progress from simple to complex concepts
- New features should include conceptual explanation in `docs/concepts/`
- **Writing Style**: Grade 10 reading level, all examples must be working code

## Branch and Development Workflow
1. Fork and clone the repository
2. Create feature branch: `git checkout -b feat/your-feature`
3. Make changes and add tests
4. Run tests and linting
5. Commit with conventional commit message
6. Push to your fork and create PR
7. Use stacked PRs for complex features

## Adding New Providers

### Step-by-Step Guide
1. **Update Provider Enum** in `instructor/utils.py`:
   ```python
   class Provider(Enum):
       YOUR_PROVIDER = "your_provider"
   ```

2. **Add Provider Modes** in `instructor/mode.py`:
   ```python
   class Mode(enum.Enum):
       YOUR_PROVIDER_TOOLS = "your_provider_tools"
       YOUR_PROVIDER_JSON = "your_provider_json"
   ```

3. **Create Client Implementation** `instructor/client_your_provider.py`:
   - Use overloads for sync/async variants
   - Validate mode compatibility
   - Return appropriate Instructor/AsyncInstructor instance
   - Handle provider-specific edge cases

4. **Add Conditional Import** in `instructor/__init__.py`:
   ```python
   if importlib.util.find_spec("your_provider_sdk") is not None:
       from .client_your_provider import from_your_provider
       __all__ += ["from_your_provider"]
   ```

5. **Update Auto Client** in `instructor/auto_client.py`:
   - Add to `supported_providers` list
   - Implement provider handling in `from_provider()`
   - Update `get_provider()` function if URL-detectable

6. **Create Tests** in `tests/llm/test_your_provider/`:
   - `conftest.py` with client fixtures
   - Basic extraction tests
   - Streaming tests
   - Validation/retry tests
   - No mocking - use real API calls

7. **Add Documentation** in `docs/integrations/your_provider.md`:
   - Installation instructions
   - Basic usage examples
   - Supported modes
   - Provider-specific features

8. **Update Navigation** in `mkdocs.yml`:
   - Add to integrations section
   - Include redirects if needed

## Contributing to Evals
- Standard evals for each provider test model capabilities
- Create new evals following existing patterns
- Run evals as part of integration test suite
- Performance tracking and comparison

## Pull Request Guidelines
- Keep PRs small and focused
- Include tests for all changes
- Update documentation as needed
- Follow PR template
- Link to relevant issues
- **Update CHANGELOG.md**: Every PR that changes behavior (fix, feat, security, deprecation) must add an entry under the current `[Unreleased]` section in `CHANGELOG.md`. Format: `- **Area**: Description ([#PR](url))`

## Type System and Best Practices

### Type Checking with ty
- **Type Checker**: Using `ty` for fast, incremental type checking
- **Python Version**: 3.9+ for compatibility
- **Configuration**: Uses `pyproject.toml` settings for type checking
- Run `uv run ty check` before committing - aim for zero errors

### Code Quality Checks Before Committing
Always run these checks before committing code:
1. **Ruff linting**: `uv run ruff check .` - Fix all errors
2. **Ruff formatting**: `uv run ruff format .` - Apply consistent formatting
3. **Type checking**: `uv run ty check` - Aim for zero type errors
4. **Tests**: Run relevant tests to ensure changes don't break functionality

### Type Patterns
- **Bounded TypeVars**: Use `T = TypeVar("T", bound=Union[BaseModel, ...])` for constraints
- **Version Compatibility**: Handle Python 3.9 vs 3.10+ typing differences explicitly
- **Union Type Syntax**: Use `from __future__ import annotations` to enable Python 3.10+ union syntax (`|`) in Python 3.9
- **Simple Type Detection**: Special handling for `list[Union[int, str]]` patterns
- **Runtime Type Handling**: Graceful fallbacks for compatibility

### Pydantic Integration
- Heavy use of `BaseModel` for structured outputs
- `TypeAdapter` used internally for JSON schema generation
- Field validators and custom types
- Models serve dual purpose: validation and documentation

## Building Documentation

### Setup
```bash
# Install documentation dependencies
pip install -r requirements-doc.txt
```

### Local Development
```bash
# Serve documentation locally with hot reload
uv run mkdocs serve

# Build documentation for production
./build_mkdocs.sh
```

### Documentation Features
- **Material Theme**: Modern UI with extensive customization
- **Plugins**:
  - `mkdocstrings` - API documentation from docstrings
  - `mkdocs-jupyter` - Notebook integration
  - `mkdocs-redirects` - URL management
  - Custom hooks for code processing
- **Custom Processing**: `hide_lines.py` removes code marked with `# <%hide%>`
- **Redirect Management**: Comprehensive redirect maps for moved content

### Writing Documentation
- Follow templates in `docs/templates/` for consistency
- Grade 10 reading level for accessibility
- All code examples must be runnable
- Include complete imports and environment setup
- Progressive complexity: simple → advanced

## Project Structure
- `instructor/` - Core library code
  - Base classes (`client.py`): `Instructor` and `AsyncInstructor`
  - Provider clients (`client_*.py`): Factory functions for each provider
  - DSL components (`dsl/`): Partial, Iterable, Maybe, Citation extensions
  - Core logic: `patch.py`, `process_response.py`, `function_calls.py`
  - CLI tools (`cli/`): Batch processing, file management, usage tracking
- `tests/` - Test suite organized by provider
  - Provider-specific tests in `tests/llm/test_<provider>/`
  - Evaluation tests for model capabilities
  - No mocking - all tests use real API calls
- `docs/` - MkDocs documentation
  - `concepts/` - Core concepts and features
  - `integrations/` - Provider-specific guides
  - `examples/` - Practical examples and cookbooks
  - `learning/` - Progressive tutorial path
  - `blog/posts/` - Technical articles and announcements
  - `templates/` - Templates for new docs (provider, concept, cookbook)
- `examples/` - Runnable code examples
  - Feature demos: caching, streaming, validation, parallel processing
  - Use cases: classification, extraction, knowledge graphs
  - Provider examples: anthropic, openai, groq, mistral
  - Each example has `run.py` as the main entry point
- `typings/` - Type stubs for untyped dependencies

## Documentation Structure
- **Getting Started Path**: Installation → First Extraction → Response Models → Structured Outputs
- **Learning Patterns**: Simple Objects → Lists → Nested Structures → Validation → Streaming
- **Example Organization**: Self-contained directories with runnable code demonstrating specific features
- **Blog Posts**: Technical deep-dives with code examples in `docs/blog/posts/`

## Example Patterns
When creating examples:
- Use `run.py` as the main file name
- Include clear imports: stdlib → third-party → instructor
- Define Pydantic models with descriptive fields
- Show expected output in comments
- Handle errors appropriately
- Make examples self-contained and runnable

## Dependency Management

### Core Dependencies
- **Minimal core**: `openai`, `pydantic`, `docstring-parser`, `typer`, `rich`
- **Python requirement**: `<4.0,>=3.9`
- **Pydantic version**: `<3.0.0,>=2.8.0` (constrained for stability)

### Optional Dependencies
Provider-specific packages as extras:
```bash
# Install with specific provider
pip install "instructor[anthropic]"
pip install "instructor[google-generativeai]"
pip install "instructor[groq]"
```

### Development Dependencies
```bash
# Install all development dependencies
uv pip install -e ".[dev]"
```
Includes:
- ty 
- `pytest` and `pytest-asyncio` - Testing
- `ruff` - Linting and formatting
- `coverage` - Test coverage
- `mkdocs` and plugins - Documentation

### Version Constraints
- **Upper bounds on all dependencies** for stability
- **Provider SDK versions** pinned to tested versions
- **Test dependencies** include evaluation frameworks

### Managing Dependencies
- Update `pyproject.toml` for new dependencies
- Test with multiple Python versions (3.9-3.12)
- Run full test suite after dependency updates
- Document any provider-specific version requirements

The library enables structured LLM outputs using Pydantic models across multiple providers with type safety.
