# V2 Test Suite

This directory contains tests for the Instructor v2 architecture, which uses a hierarchical registry system for providers and modes.

## Test Organization

### Unified Tests (Cross-Provider)

These tests use parametrization to test common behavior across all providers:

- **`test_client_unified.py`** - Unified client factory tests
  - Mode registry tests (supported/unsupported modes)
  - Mode normalization tests (generic and legacy modes)
  - Import tests (from_* functions)
  - Error handling tests (unsupported modes)
  - SDK availability tests

- **`test_handler_registration_unified.py`** - Unified handler registration tests
  - Mode registration verification
  - Handler method existence checks
  - Provider-mode mapping tests
  - Handler inheritance tests (OpenAI-compatible providers)

- **`test_handlers_parametrized.py`** - Unified handler method tests
  - `prepare_request` tests
  - `parse_response` tests
  - `handle_reask` tests
  - Tests handler methods across all providers and modes

- **`test_provider_modes.py`** - Integration tests with actual API calls
  - Mode registration verification
  - Basic extraction tests (sync and async)
  - Provider-specific mode tests (e.g., Anthropic parallel tools)

- **`test_mode_normalization.py`** - Comprehensive mode normalization tests
  - Legacy mode deprecation warnings
  - Mode normalization mappings
  - Backwards compatibility tests

### Provider-Specific Tests

Each provider has two test files that focus on provider-specific behavior:

- **`test_*_client.py`** - Provider-specific client factory tests
  - SDK-specific integration tests
  - Provider-specific helper functions (e.g., xAI's `_get_model_schema`)
  - Provider-specific validation logic
  - Custom error messages

- **`test_*_handlers.py`** - Provider-specific handler tests
  - Provider-specific response formats (e.g., Cohere V1/V2)
  - Provider-specific message conversion
  - Provider-specific edge cases
  - Handler inheritance tests (for OpenAI-compatible providers)

### Provider-Specific Feature Tests

These tests focus on unique provider features that cannot be unified:

- **`test_genai_integration.py`** - GenAI-specific integration tests
  - Tests GenAI's unique `use_async` parameter pattern (not `async_client=True`)
  - Tests GenAI's backwards compatibility with legacy modes
  - Tests GenAI's unique client structure (`models` and `aio.models`)
  - Cannot be unified because GenAI's API pattern differs from other providers

- **`test_openai_streaming.py`** - OpenAI-specific streaming behavior tests
  - Tests OpenAI handler's `_consume_streaming_flag` method
  - Tests `tool_choice` behavior for streaming iterables
  - Cannot be unified because it tests OpenAI handler implementation details

### Core Tests

- **`test_registry.py`** - Core registry functionality tests (parameterized by provider and mode)
- **`test_routing.py`** - Tests for `from_provider()` routing

## Test Principles

### What Should Be Unified

Tests that are identical or nearly identical across providers should be unified:

- ✅ Mode registry checks
- ✅ Mode normalization
- ✅ Handler method signatures
- ✅ Import availability
- ✅ Common error handling

### What Should Remain Provider-Specific

Tests that are unique to a provider should remain in provider-specific files:

- ✅ Provider-specific response formats (Cohere V1/V2, Mistral list content)
- ✅ Provider-specific helper functions
- ✅ Provider-specific message conversion logic
- ✅ Provider-specific edge cases
- ✅ SDK-specific integration tests

## Adding a New Provider

When adding a new provider, follow these steps:

1. **Add to unified test configs**:
   - Add provider config to `PROVIDER_CLIENT_CONFIGS` in `test_client_unified.py`
   - Add provider modes to `PROVIDER_HANDLER_MODES` in `test_handlers_parametrized.py`

2. **Create provider-specific test files**:
   - `test_<provider>_client.py` - Only provider-specific client tests
   - `test_<provider>_handlers.py` - Only provider-specific handler tests

3. **Avoid duplicating unified tests**:
   - Don't add mode registry tests (covered by unified tests)
   - Don't add mode normalization tests (covered by unified tests)
   - Don't add handler registration tests (covered by unified tests)

## Running Tests

Run all v2 tests:

```bash
uv run pytest tests/v2/
```

Run unified tests only:

```bash
uv run pytest tests/v2/test_client_unified.py tests/v2/test_handler_registration_unified.py tests/v2/test_handlers_parametrized.py
```

Run provider-specific tests:

```bash
uv run pytest tests/v2/test_fireworks_client.py tests/v2/test_fireworks_handlers.py
```

Run tests for a specific provider:

```bash
uv run pytest tests/v2/test_fireworks_*.py
```

## Test Coverage

The unified tests provide comprehensive coverage across all providers:

- **Client Factory**: Mode normalization, registry, imports, errors
- **Handler Registration**: Mode registration, handler methods, provider-mode mapping
- **Handler Methods**: prepare_request, parse_response, handle_reask

Provider-specific tests add coverage for:

- Provider-specific formats and conversions
- Provider-specific edge cases
- SDK integration details

Shared test helpers:

- `get_registered_provider_mode_pairs()` in `tests/v2/conftest.py` keeps registry
  assertions parameterized across providers and modes.

## Migration Notes

As of the unification effort:

- Common client factory tests moved to `test_client_unified.py`
- Common handler registration tests moved to `test_handler_registration_unified.py`
- Provider-specific files now focus on unique behavior
- ~74% reduction in duplicated test code

See `UNIFICATION_OPPORTUNITIES.md` for details on the unification effort.
