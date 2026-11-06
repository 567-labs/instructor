---
authors:
- jxnl
categories:
- Performance Optimization
- Cost Reduction
- API Efficiency
- Python Development
comments: true
date: 2025-01-08
description: Instructor v1.9.1 introduces native caching support for all providers. Learn how to drastically reduce API costs and improve response times with built-in cache adapters.
draft: false
slug: native-caching-v1-9-1
tags:
- Python
- Caching
- Performance Optimization
- API Cost Optimization
- LLM Applications
- Production Scaling
- from_provider
---

# Native Caching in Instructor v1.9.1: Zero-Configuration Performance Boost

> **New in v1.9.1**: Instructor now ships with built-in caching support for all providers. Simply pass a cache adapter when creating your client to dramatically reduce API costs and improve response times.

Starting with Instructor v1.9.1, we've introduced native caching support that makes optimization effortless. Instead of implementing complex caching decorators or wrapper functions, you can now pass a cache adapter directly to `from_provider()` and automatically cache all your structured LLM calls.

## The Game Changer: Built-in Caching

Before v1.9.1, caching required custom decorators and manual implementation. Now, it's as simple as:

```python
from instructor import from_provider
from instructor.cache import AutoCache

# Works with any provider - caching flows through automatically
client = from_provider("openai/gpt-4o", cache=AutoCache(maxsize=1000))

# Your normal calls are now cached automatically
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


first = client.create(
    messages=[{"role": "user", "content": "Extract: John is 25"}], response_model=User
)

second = client.create(
    messages=[{"role": "user", "content": "Extract: John is 25"}], response_model=User
)

# second call was served from cache - same result, zero cost!
assert first.name == second.name
```

## Universal Provider Support

The beauty of native caching is that it works with **every provider** through the same simple API:

```python
from instructor.cache import AutoCache, DiskCache

# Works with OpenAI
openai_client = from_provider("openai/gpt-5-nano", cache=AutoCache())

# Works with Anthropic
anthropic_client = from_provider("anthropic/claude-3-haiku", cache=AutoCache())

# Works with Google
google_client = from_provider("google/gemini-pro", cache=DiskCache())

# Works with any provider in the ecosystem
groq_client = from_provider("groq/llama-3.1-8b", cache=AutoCache())
```

No provider-specific configuration needed. The cache parameter flows through `**kwargs` to all underlying implementations automatically.

## Built-in Cache Adapters

Instructor v1.9.1 ships with two production-ready cache implementations:

### 1. AutoCache - In-Process LRU Cache

Perfect for single-process applications and development:

```python
from instructor.cache import AutoCache

# Thread-safe in-memory cache with LRU eviction
cache = AutoCache(maxsize=1000)
client = from_provider("openai/gpt-4o", cache=cache)
```

**When to use**:
- Development and testing
- Single-process applications
- When you need maximum speed (200,000x+ faster cache hits)
- Applications where cache persistence isn't required

### 2. DiskCache - Persistent Storage

Ideal when you need cache persistence across sessions:

```python
from instructor.cache import DiskCache

# Persistent disk-based cache
cache = DiskCache(directory=".instructor_cache")
client = from_provider("anthropic/claude-3-sonnet", cache=cache)
```

**When to use**:
- Applications that restart frequently
- Development workflows where you want to preserve cache between sessions
- When working with expensive or time-intensive API calls
- Local applications with moderate performance requirements

## Smart Cache Key Generation

Instructor automatically generates intelligent cache keys that include:

- **Provider/model name** - Different models get different cache entries
- **Complete message history** - Full conversation context is hashed
- **Response model schema** - Any changes to your Pydantic model automatically bust the cache
- **Mode configuration** - JSON vs Tools mode changes are tracked

This means when you update your Pydantic model (adding fields, changing descriptions, etc.), the cache automatically invalidates old entries - no stale data!

```python
from instructor.cache import make_cache_key

# Generate deterministic cache key
key = make_cache_key(
    messages=[{"role": "user", "content": "hello"}],
    model="gpt-5.4-mini",
    response_model=User,
    mode="TOOLS",
)
print(key)  # SHA-256 hash: 9b8f5e2c8c9e...
```

## Custom Cache Implementations

Want Redis, Memcached, or a custom backend? Simply inherit from `BaseCache`:

```python
from instructor.cache import BaseCache
import redis


class RedisCache(BaseCache):
    def __init__(self, host="localhost", port=6379, **kwargs):
        self.redis = redis.Redis(host=host, port=port, **kwargs)

    def get(self, key: str):
        value = self.redis.get(key)
        return value.decode() if value else None

    def set(self, key: str, value, ttl: int | None = None):
        if ttl:
            self.redis.setex(key, ttl, value)
        else:
            self.redis.set(key, value)


# Use your custom cache
redis_cache = RedisCache(host="my-redis-server")
client = from_provider("openai/gpt-4o", cache=redis_cache)
```

The `BaseCache` interface is intentionally minimal - just implement `get()` and `set()` methods and you're ready to go.

## Time-to-Live (TTL) Support

Control cache expiration with per-call TTL overrides:

```python
# Cache this result for 1 hour
result = client.create(
    messages=[{"role": "user", "content": "Generate daily report"}],
    response_model=Report,
    cache_ttl=3600,  # 1 hour in seconds
)
```

TTL support depends on your cache backend:
- **AutoCache**: TTL is ignored (no expiration)
- **DiskCache**: Full TTL support with automatic expiration
- **Custom backends**: Implement TTL handling in your `set()` method

## Migration from Manual Caching

If you were using custom caching decorators, migrating is straightforward:

**Before v1.9.1**:
```python
@functools.cache
def extract_user(text: str) -> User:
    return client.create(
        messages=[{"role": "user", "content": text}], response_model=User
    )
```

**With v1.9.1**:
```python
# Remove decorator, add cache to client
client = from_provider("openai/gpt-4o", cache=AutoCache())


def extract_user(text: str) -> User:
    return client.create(
        messages=[{"role": "user", "content": text}], response_model=User
    )
```

No more function-level caching logic - just create your client with caching enabled and all calls benefit automatically.

## Real-World Performance Impact

Native caching delivers the same dramatic performance improvements you'd expect:

- **AutoCache**: 200,000x+ speed improvement for cache hits
- **DiskCache**: 5-10x improvement with persistence benefits
- **Cost Reduction**: 50-90% API cost savings depending on cache hit rate

For a comprehensive deep-dive into caching strategies and performance analysis, check out our [complete caching guide](caching.md).

## Getting Started

Ready to enable native caching? Here's your quick start:

1. **Upgrade to v1.9.1+**:
   ```bash
   pip install "instructor>=1.9.1"
   ```

2. **Choose your cache backend**:
   ```python
   from instructor.cache import AutoCache, DiskCache
   
   # For development/single-process
   cache = AutoCache(maxsize=1000)
   
   # For persistence
   cache = DiskCache(directory=".cache")
   ```

3. **Add cache to your client**:
   ```python
   from instructor import from_provider
   
   client = from_provider("your/favorite/model", cache=cache)
   ```

4. **Use normally - caching happens automatically**:
   ```python
   result = client.create(
       messages=[{"role": "user", "content": "your prompt"}], response_model=YourModel
   )
   ```

## Learn More

For detailed information about cache design, custom implementations, and advanced patterns, visit our [Caching Concepts](../../concepts/caching.md) documentation.

The native caching feature represents our commitment to making high-performance LLM applications simple and accessible. No more complex caching logic - just fast, cost-effective structured outputs out of the box.

---

*Have questions about native caching or want to share your use case? Join the discussion in our [GitHub repository](https://github.com/jxnl/instructor) or check out the [complete documentation](../../concepts/caching.md).*