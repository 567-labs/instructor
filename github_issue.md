# Refactor OpenAISchema class methods to standalone functions

## Summary

Currently, schema generation for different LLM providers requires models to inherit from `OpenAISchema` or be wrapped with the `@openai_schema` decorator. This creates an unnecessary inheritance requirement and couples schema generation to class-based patterns.

We should refactor the schema generation logic into standalone, provider-agnostic functions.

## Current State Analysis

**Current usage pattern**: `response_model.openai_schema` (where response_model inherits from OpenAISchema)

**Affected files with usage counts**:
- `instructor/utils/` (12 calls across cerebras.py, writer.py, fireworks.py, openai.py, mistral.py)
- `instructor/process_response.py` (11 calls)
- `instructor/dsl/parallel.py` (3 calls - handles parallel tools)
- `instructor/distil.py` (1 call)
- `instructor/function_calls.py` (13 calls - method definitions and internal usage)
- `instructor/utils/core.py` (1 call - decorator application)
- `instructor/utils/anthropic.py` (1 call - anthropic_schema)
- `instructor/utils/google.py` (1 call - gemini_schema)
- Examples and tests (20+ calls)

**Total**: ~60 usages across codebase

## Proposed Solution

### 1. Create `instructor/schema_utils.py` with standalone functions:

```python
from __future__ import annotations
import functools
from typing import Any, Type
from docstring_parser import parse
from pydantic import BaseModel

@functools.lru_cache(maxsize=256)
def generate_openai_schema(model: Type[BaseModel]) -> dict[str, Any]:
    """Generate OpenAI function schema from Pydantic model."""
    # Move logic from OpenAISchema.openai_schema here

def generate_anthropic_schema(model: Type[BaseModel]) -> dict[str, Any]:
    """Generate Anthropic tool schema from Pydantic model."""
    # Move logic from OpenAISchema.anthropic_schema here

def generate_gemini_schema(model: Type[BaseModel]) -> Any:
    """Generate Gemini function schema from Pydantic model."""
    # Move logic from OpenAISchema.gemini_schema here
```

### 2. Update OpenAISchema class to delegate to new functions:

```python
class OpenAISchema(BaseModel):
    @classproperty
    def openai_schema(cls):
        return generate_openai_schema(cls)

    @classproperty  
    def anthropic_schema(cls):
        return generate_anthropic_schema(cls)

    @classproperty
    def gemini_schema(cls):
        return generate_gemini_schema(cls)
```

### 3. Migration path:

**Phase 1**: Add new functions, maintain backward compatibility
- All existing `response_model.openai_schema` calls continue working
- New code can use `generate_openai_schema(response_model)` directly

**Phase 2**: Internal migration  
- Replace internal usage in utils/ and process_response.py
- Update parallel tools handling in dsl/parallel.py

**Phase 3**: Deprecation
- Mark `@openai_schema` decorator as deprecated
- Encourage users to migrate to standalone functions

## Benefits

1. **No inheritance requirement** - Any Pydantic model can generate schemas
2. **Provider-agnostic** - Clean separation of schema generation logic
3. **Better testability** - Functions are easier to unit test
4. **Performance** - LRU cache maintains current performance characteristics
5. **Backward compatibility** - Zero breaking changes during transition
6. **Cleaner API** - More functional approach vs class-based inheritance

## Implementation Checklist

- [ ] Create `instructor/schema_utils.py` with standalone functions
- [ ] Update `OpenAISchema` class to delegate to new functions  
- [ ] Add comprehensive tests comparing old vs new output
- [ ] Update internal usage in utils/ (12 locations)
- [ ] Update process_response.py (11 locations)
- [ ] Update parallel tools handling in dsl/parallel.py
- [ ] Update distil.py usage
- [ ] Mark decorator as deprecated with warning
- [ ] Update documentation and examples
- [ ] Run full test suite to ensure no regressions

## Special Considerations

- **Parallel tools**: `dsl/parallel.py` uses both `openai_schema(model).openai_schema` and `openai_schema(model).anthropic_schema` patterns
- **Caching**: Current `@classproperty` provides implicit memoization - maintain with `@lru_cache`
- **Error handling**: Preserve current validation and error behavior
- **Provider compatibility**: Ensure schema output remains identical for all providers

This refactoring will modernize the schema generation approach while maintaining full backward compatibility.
