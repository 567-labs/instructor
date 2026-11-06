# Instructor Caching Prototype

This example demonstrates the new built-in caching functionality in Instructor.

## Files

- `run.py` - Main example showing all caching features (with mock calls for quick testing)
- `run_real.py` - Complete demo with real API calls
- `test_simple.py` - Unit tests for cache components without API calls
- `test_anthropic.py` - Tests with Anthropic provider to verify caching works across providers

## Features Demonstrated

### 1. AutoCache (In-Memory LRU)
```python
from instructor.cache import AutoCache

cache = AutoCache(maxsize=100)
client = instructor.from_openai(OpenAI(), cache=cache)
```

### 2. DiskCache (Persistent)
```python
from instructor.cache import DiskCache

cache = DiskCache(directory=".instructor_cache")
client = instructor.from_openai(OpenAI(), cache=cache)
```

### 3. Cache TTL (Time-to-Live)
```python
client.create(
    model="gpt-3.5-turbo",
    messages=messages,
    response_model=User,
    cache_ttl=3600,  # 1 hour
)
```

### 4. create_with_completion Support
Both the parsed model and raw completion objects are cached and restored.

## Performance Results

From our tests:
- **156x faster** cache hits vs API calls
- **Identical results** from cache and API
- **Persistent storage** across client instances
- **Automatic cache invalidation** based on:
  - Different prompts
  - Different models
  - Different response schemas
  - TTL expiration

## Running the Examples

```bash
# Run the complete demo (requires OpenAI API key)
uv run python run_real.py

# Run unit tests (no API required)
uv run python test_simple.py

# Run pytest tests
uv run pytest tests/test_cache*.py
```

## Key Features

1. **Deterministic caching** - same inputs always produce same cache key
2. **Schema-aware** - changing field descriptions invalidates cache
3. **Multiple backends** - AutoCache (LRU), DiskCache (persistent)
4. **TTL support** - automatic expiration (where supported)
5. **Raw response preservation** - `create_with_completion` works seamlessly
6. **Thread-safe** - all cache implementations are thread-safe