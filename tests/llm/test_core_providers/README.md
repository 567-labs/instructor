# Core Provider Tests

This directory contains unified tests that run across **all core providers**: OpenAI, Anthropic, Google (Gemini), Cohere, xAI, Mistral, Cerebras, Fireworks, Writer, and Perplexity.

## Interpreting test evidence

[ProviderSpec](../../../instructor/v2/core/provider_specs.py) declares normalized
modes, aliases and explicit rejections. The [shared configuration](../shared_config.py)
selects models and modes for this suite; [capability policy](capabilities.py)
controls feature skips. Neither a registry entry nor a green job certifies a
provider/API/mode combination that was not exercised.

The [Test workflow](../../../.github/workflows/test.yml) uses the shared
[JUnit reporter](../../../scripts/provider_test_evidence.py) for live jobs:

- Missing credentials/configuration and no collected tests remain non-failing but
  explicitly untested. Missing or unreadable reports have unknown counts; an empty
  report has zero outcomes. An all-skipped run provides no passing evidence.
- Initial, retry and documented-model reports remain separate. A passing retry
  does not erase an initial failure. Counts are JUnit outcomes, not API calls or
  unique tests across attempts.
- Skip categories are coarse labels; inspect `pytest -rs` logs for exact reasons.
  Configuration presence is reported without secret values. Tests omitted during
  collection or selection cannot be inferred from XML, and jobs that never start
  cannot emit a summary.

The [retry tests](test_retries.py) exercise the retry budget and check a
second validation after an intentionally rejected answer. [Raw-response tests](test_response_modes.py) check non-null completions;
[local Responses SDK contracts](../../v2/test_responses_sdk_contract.py) separately
check sync/async wire mapping, SDK response types and HTTP error causes using
loopback HTTP. Unit/handler tests in [tests/coverage](../../coverage) also use test
doubles; their coverage is not live-provider evidence. Marker selection is not a
network sandbox, so keep provider credentials out of offline coverage runs.

## Run the cases shared with Rust

The paired marker identifies nine async extraction cases. They progress from
a scalar and a flat record to a string-valued enum, a list of records and a
nested address. The manifest combines a list of records, enum states and an
optional carrier. The remaining cases check a validation retry and both partial
and iterable streaming. The validation case rejects the first valid answer,
then checks the accepted name and age.

| Pytest ID | Model | API key |
| --- | --- | --- |
| `openai` | `openai/gpt-6-luna` | `OPENAI_API_KEY` |
| `openai-responses` | `openai/gpt-6-luna` via Responses tools | `OPENAI_API_KEY` |
| `anthropic` | `anthropic/claude-haiku-5-5` | `ANTHROPIC_API_KEY` |
| `google` | `google/gemini-3.8-flash` or `GOOGLE_GENAI_MODEL` | `GOOGLE_API_KEY` |
| `cohere` | `cohere/command-a-03-2025` | `COHERE_API_KEY` |
| `mistral` | `mistral/ministral-8b-latest` | `MISTRAL_API_KEY` |
| `openrouter` | `openrouter/google/gemini-3.8-flash` | `OPENROUTER_API_KEY` |

The shared cases have stable route IDs even when credentials or SDKs are
missing. Pytest reports a skip with the missing requirement. The command below
lists the 63 paired nodes without making paid requests. Remove
`--collect-only` to run them with the installed SDKs and available credentials.

```bash
uv run --frozen pytest tests/llm/test_core_providers -m paired \
  -k 'openai or anthropic or google or cohere or mistral or openrouter' \
  -rs --collect-only
```

OpenRouter and OpenAI Responses join these nine cases. The two new shipment
tests use only the shared routes. Existing core cases also run for the other
providers configured in this suite. The separate documentation smoke checks
use the IDs from the guides, including `mistral/mistral-small-latest` for Mistral.
[Documented model cases](../test_current_doc_models.py) · [Decisions cases](../test_decisions/test_live_decisions.py)

## Philosophy

Instead of duplicating the same tests for each provider, we use `instructor.from_provider()` with parameterization to run the same test suite against all providers simultaneously.

## Test Organization

### Core Tests (Run on All Providers)

These tests verify that core instructor functionality works consistently across providers:

- **test_basic_extraction.py** - Simple extraction, lists, nested models, field descriptions
- **test_streaming.py** - Partial streaming, Iterable streaming, union types
- **test_validation.py** - Validators, field constraints, custom validation
- **test_retries.py** - Retry logic and max_retries parameter
- **test_response_modes.py** - Different client methods (create, messages.create, etc.)
- **test_simple_types.py** - Simple types (int, bool, str, Literal, Union, Enum)


## Configuration

### shared_config.py

Located in `tests/llm/shared_config.py`, this file:

- Defines `PROVIDER_CONFIGS` with model names, modes, and required API keys
- Implements `get_available_providers()` to detect which providers are available
- Provides `pytest_generate_tests()` hook for automatic parameterization
- Handles skipping when API keys or packages are missing

### Usage in Tests

Tests use the `provider_config` fixture which is automatically parametrized:

```python
def test_something(provider_config):
    model, mode = provider_config
    client = instructor.from_provider(model, mode=mode)

    result = client.create(
        response_model=MyModel,
        messages=[{"role": "user", "content": "..."}],
    )

    assert isinstance(result, MyModel)
```

The test will automatically run for each available provider:
- OpenAI (if OPENAI_API_KEY is set)
- Anthropic (if ANTHROPIC_API_KEY is set)
- Google (if GOOGLE_API_KEY is set)
- Cohere (if COHERE_API_KEY is set)
- xAI (if XAI_API_KEY is set)
- Mistral (if MISTRAL_API_KEY is set)
- Cerebras (if CEREBRAS_API_KEY is set)
- Fireworks (if FIREWORKS_API_KEY is set)
- Writer (if WRITER_API_KEY is set)
- Perplexity (if PERPLEXITY_API_KEY is set)

Tests automatically skip if the API key or package is not available.

## Running Tests

`uv` is Astral's fast Python package manager. Install it by following the [official guide](https://docs.astral.sh/uv/getting-started/install/) if it is not already on your PATH.

### Run all core provider tests:
```bash
uv run pytest tests/llm/test_core_providers/ -v
```

### Run specific test file:
```bash
uv run pytest tests/llm/test_core_providers/test_basic_extraction.py -v
```

### Run specific test:
```bash
uv run pytest tests/llm/test_core_providers/test_basic_extraction.py::test_simple_extraction -v
```

### Run tests for specific provider only:
```bash
# Only OpenAI
uv run pytest tests/llm/test_core_providers/ -k "openai" -v

# Only Anthropic
uv run pytest tests/llm/test_core_providers/ -k "anthropic" -v

# Only Google
uv run pytest tests/llm/test_core_providers/ -k "google" -v
```

### Skip tests when API keys are missing:
Tests automatically skip if the required API key or package is not available.

Required API keys (set only what you have):
- `OPENAI_API_KEY` - for OpenAI
- `ANTHROPIC_API_KEY` - for Anthropic
- `GOOGLE_API_KEY` - for Google (Gemini)
- `GOOGLE_GENAI_MODEL` - model string for Google GenAI tests (e.g., `google/gemini-3-flash`)
- `COHERE_API_KEY` - for Cohere
- `XAI_API_KEY` - for xAI (Grok)
- `MISTRAL_API_KEY` - for Mistral
- `CEREBRAS_API_KEY` - for Cerebras
- `FIREWORKS_API_KEY` - for Fireworks
- `WRITER_API_KEY` - for Writer
- `PERPLEXITY_API_KEY` - for Perplexity

## Current Models

Read and update `PROVIDER_CONFIGS` in [shared_config.py](../shared_config.py) for
current model names, modes, SDK requirements and credential checks.

## Benefits

✅ **Less code**: ~3,500+ lines of duplicate code eliminated
✅ **Easier maintenance**: Update test logic once, applies to all providers
✅ **Shared coverage**: Applies the same assertions to configured provider cases
✅ **Faster development**: Add new providers by updating one config file
✅ **Consistent behavior**: Catches provider-specific quirks early

## Migration Status

- ✅ Shared configuration created
- ✅ Core test files created (basic_extraction, streaming, validation, retries, response_modes, simple_types)
- ✅ util.py files updated to use `provider/model` format
- ✅ Provider-specific tests cleaned up (removed all duplicates)
- ✅ Deleted 6 entire provider directories (cerebras, fireworks, perplexity, cohere, xai, mistral)
- ✅ Deleted 35+ duplicate test files across remaining providers

## Adding New Core Tests

1. Create test file in `tests/llm/test_core_providers/`
2. Use `provider_config` parameter in test functions
3. Extract `model, mode = provider_config`
4. Create client with `instructor.from_provider(model, mode=mode)`
5. Write provider-agnostic assertions

## Adding New Providers

To add a new provider to core tests:

1. Update `PROVIDER_CONFIGS` in `tests/llm/shared_config.py`
2. Add tuple: `("provider/model-name", instructor.Mode.PROVIDER_SPECIFIC_MODE, "API_KEY_ENV_VAR", "package.name")`
3. Pick the mode that matches the provider's client (see `instructor.Mode` or the provider guide).
4. Tests will automatically run against the new provider!
