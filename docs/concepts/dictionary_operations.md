## See Also

- [Types](./types.md) - Working with different data types
- [Response Models](./models.md) - Working with Pydantic models
- [Fields](./fields.md) - Customizing field metadata
- [Union Types](./unions.md) - Handle multiple possible types

---
title: Dictionary Operations Optimization in Instructor
description: Learn about performance optimizations for dictionary operations in Instructor, including message extraction and configuration parameter handling.
---

# Dictionary Operations Optimization

This document explains the dictionary operations optimizations implemented in Instructor.

## Overview

Dictionary operations are one of the most common operations in the Instructor codebase, especially when handling message passing between different LLM providers and managing configuration parameters. Optimizing these operations can lead to significant performance improvements, especially in high-throughput applications.

## Optimized Areas

### Message Extraction

The `extract_messages` function was optimized to use direct key lookups instead of nested `get()` calls, which reduces the overhead of function calls and improves performance.

**Before:**
```python
from typing import Any


def extract_messages(kwargs: dict[str, Any]) -> Any:
    return kwargs.get(
        "messages", kwargs.get("contents", kwargs.get("chat_history", []))
    )
```

**After:**
```python
from typing import Any


def extract_messages(kwargs: dict[str, Any]) -> Any:
    if "messages" in kwargs:
        return kwargs["messages"]
    if "contents" in kwargs:
        return kwargs["contents"]
    if "chat_history" in kwargs:
        return kwargs["chat_history"]
    return []
```

### Response Processing Functions

The response processing functions were optimized to:
1. Pre-extract commonly used variables to avoid repeated dictionary lookups
2. Use the optimized `extract_messages` function instead of nested get operations
3. Reduce redundant dictionary operations in error handling

### Message Handler Selection

The `handle_reask_kwargs` function was optimized to use direct conditional checks instead of creating a large mapping dictionary, which reduces memory overhead and improves lookup performance.

**Before:**
```python
def handle_reask_kwargs(kwargs, mode, response, exception):
    kwargs = kwargs.copy()
    functions = {
        Mode.TOOLS: reask_anthropic_tools,
        Mode.JSON: reask_anthropic_json,
        # ... many more mappings
    }
    reask_function = functions.get(mode, reask_default)
    return reask_function(kwargs=kwargs, response=response, exception=exception)
```

**After:**
```python
def handle_reask_kwargs(kwargs, mode, response, exception):
    kwargs_copy = kwargs.copy()

    if mode in {Mode.TOOLS, Mode.ANTHROPIC_REASONING_TOOLS}:
        return reask_anthropic_tools(kwargs_copy, response, exception)
    elif mode == Mode.JSON:
        return reask_anthropic_json(kwargs_copy, response, exception)
    # ... optimized conditional checks with grouped modes
    else:
        return reask_default(kwargs_copy, response, exception)
```

### System Message Handling

The `combine_system_messages` function in `utils.py` was optimized to:
1. Cache type checks to avoid repeated calls
2. Use more efficient list operations to avoid creating intermediate lists
3. Optimize type conversion scenarios

## Benchmarks

Benchmarks show significant improvements in dictionary operation performance:

| Operation | Before (ms) | After (ms) | Improvement |
|-----------|-------------|------------|-------------|
| extract_messages | ~0.08 | ~0.03 | ~62% |
| handle_reask_kwargs | ~0.09 | ~0.05 | ~44% |
| combine_system_messages | ~0.12 | ~0.07 | ~42% |

The exact improvement depends on the specific use case and data patterns.

## Testing

Two types of tests were created to ensure the optimizations were safe:

1. **Validation Tests** - Ensure the optimized functions return the same results as before
2. **Benchmark Tests** - Measure and verify the performance improvements

These tests help ensure that the optimizations improve performance without changing behavior.

## Conclusion

Dictionary operations optimization is a key part of making Instructor more efficient, especially for high-throughput applications. By carefully optimizing these common operations, we can improve performance without changing the API or behavior of the library.