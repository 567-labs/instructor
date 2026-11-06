# V2 Test Unification Opportunities

This document outlines opportunities to unify more of the v2 tests to reduce duplication and improve maintainability.

## Current State

### Already Unified
1. **`test_handlers_parametrized.py`** - Tests handler methods (`prepare_request`, `parse_response`, `handle_reask`) across all providers
2. **`test_provider_modes.py`** - Tests mode registration and basic extraction across providers
3. **`test_mode_normalization.py`** - Tests mode normalization (but could be expanded)

### Provider-Specific Test Files
Each provider has separate test files:
- `test_*_client.py` - Client factory tests (8 files)
- `test_*_handlers.py` - Handler-specific tests (8 files)

## Unification Opportunities

### 1. Client Factory Tests (HIGH PRIORITY)

**Current State**: Each provider has a `test_*_client.py` file with nearly identical tests:
- Mode normalization tests
- Mode registry tests
- Error handling tests
- Import tests
- SDK availability tests

**Solution**: Created `test_client_unified.py` that parametrizes all these tests across providers.

**Benefits**:
- Reduces ~800 lines of duplicated code
- Single source of truth for client factory behavior
- Easier to add new providers (just add to config dict)
- Consistent test coverage across providers

**What's Unified**:
- ✅ Mode registry tests (supported/unsupported modes)
- ✅ Mode normalization tests (generic and legacy modes)
- ✅ Import tests (from_* functions)
- ✅ Error handling tests (unsupported modes)
- ✅ SDK availability tests

**What Remains Provider-Specific**:
- Client helper functions (e.g., `_get_model_schema` for xAI)
- Provider-specific validation logic
- Custom error messages

### 2. Handler Registration Tests (MEDIUM PRIORITY)

**Current State**: Each `test_*_handlers.py` file has handler registration tests that are nearly identical.

**Solution**: Created `test_handler_registration_unified.py` that parametrizes registration tests.

**Benefits**:
- Reduces ~200 lines of duplicated code
- Ensures consistent registration behavior
- Tests handler inheritance patterns

**What's Unified**:
- ✅ Mode registration verification
- ✅ Handler method existence checks
- ✅ Provider-mode mapping tests
- ✅ Handler inheritance tests (OpenAI-compatible providers)

### 3. Handler-Specific Tests (LOW PRIORITY - Already Mostly Unified)

**Current State**: `test_handlers_parametrized.py` already covers most handler method tests.

**Remaining Opportunities**:
- Edge case tests (complex models, optional fields, nested models)
- Provider-specific response format tests (V1 vs V2 for Cohere)
- Provider-specific message conversion tests

**Recommendation**: Keep provider-specific handler tests for:
- Provider-specific response formats (Cohere V1/V2, Mistral list content)
- Provider-specific edge cases
- Provider-specific helper functions

### 4. Mode Normalization Tests (MEDIUM PRIORITY)

**Current State**: Mode normalization tests are duplicated in:
- `test_mode_normalization.py` (comprehensive)
- `test_*_client.py` files (provider-specific)
- `test_*_handlers.py` files (provider-specific)

**Solution**: Expand `test_mode_normalization.py` to cover all providers, remove duplicates from provider-specific files.

**Benefits**:
- Single source of truth for normalization
- Easier to verify all legacy modes are handled
- Consistent deprecation warning behavior

### 5. Edge Case Tests (LOW PRIORITY)

**Current State**: Some edge case tests are duplicated across providers:
- Complex/nested models
- Optional fields
- Strict validation
- Empty messages
- Incomplete output handling

**Opportunity**: Create `test_handler_edge_cases_unified.py` for common edge cases.

**Consideration**: Some edge cases are provider-specific (e.g., Cohere V1/V2 format handling), so this may not be worth unifying.

## Implementation Plan

### Phase 1: Client Factory Tests ✅
- [x] Create `test_client_unified.py`
- [ ] Update existing `test_*_client.py` files to remove duplicated tests
- [ ] Keep provider-specific tests (helper functions, custom validation)

### Phase 2: Handler Registration Tests ✅
- [x] Create `test_handler_registration_unified.py`
- [ ] Update existing `test_*_handlers.py` files to remove registration tests
- [ ] Keep provider-specific handler tests

### Phase 3: Mode Normalization
- [ ] Expand `test_mode_normalization.py` to cover all providers
- [ ] Remove normalization tests from provider-specific files
- [ ] Ensure deprecation warnings are tested consistently

### Phase 4: Cleanup
- [ ] Review remaining provider-specific tests
- [ ] Document what should remain provider-specific
- [ ] Update test documentation

## Metrics

### Before Unification
- **Client test files**: 8 files, ~200 lines each = ~1600 lines
- **Handler test files**: 8 files, ~400 lines each = ~3200 lines
- **Total**: ~4800 lines

### After Unification
- **Unified client tests**: 1 file, ~300 lines
- **Unified handler registration**: 1 file, ~150 lines
- **Provider-specific tests**: ~8 files, ~100 lines each = ~800 lines
- **Total**: ~1250 lines

### Reduction
- **~74% reduction** in test code
- **Easier maintenance** - changes in one place
- **Better coverage** - consistent tests across providers

## What Should Remain Provider-Specific

1. **Provider-specific response formats**
   - Cohere V1 vs V2 handling
   - Mistral list content format
   - xAI tuple responses

2. **Provider-specific helper functions**
   - `_get_model_schema` (xAI)
   - `_detect_client_version` (Cohere)
   - `_convert_messages_to_cohere_v1` (Cohere)

3. **Provider-specific edge cases**
   - Cohere V1/V2 message conversion
   - Mistral list-format system messages
   - Provider-specific error messages

4. **Integration tests**
   - Actual API calls (already in `test_provider_modes.py`)
   - Provider-specific features (e.g., Anthropic reasoning tools)

## Next Steps

1. **Review unified tests** - Ensure they cover all cases
2. **Run test suite** - Verify no regressions
3. **Update provider-specific tests** - Remove duplicated code
4. **Document patterns** - Help future developers understand what to test where
