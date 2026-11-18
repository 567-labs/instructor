## See Also

- [Prompt Caching](./prompt_caching.md) - Cache prompts for cost optimization
- [Performance Optimization](../examples/sqlmodel.md#performance-optimization) - Performance best practices
- [Cost Optimization](../examples/batch_job_oai.md) - Reduce API costs
- [Hooks](./hooks.md) - Monitor cache hits and misses

---
title: Caching Strategies with Instructor
description: Learn how to use caching with Instructor to reduce API costs and improve performance.
---

For more details on caching concepts, see our [blog](../blog/posts/caching.md).

## Built-in Caching (v1.9.1 and later)

### Cache isolation in 1.17

Each patched client has its own cache namespace by default. Repeated calls through
that client can reuse entries; a newly constructed client gets a separate namespace,
even when it uses the same disk cache. This prevents accidental reuse across
endpoints or accounts that use the same model name.

For intentional reuse across client instances or process restarts, pass a stable
`cache_namespace` string alongside `cache` when calling `client.create`. Choose a
namespace that identifies the endpoint, account or tenant, and application policy.
Never use credentials as namespace values or share one namespace between tenants.
Change the namespace when switching endpoints, accounts, or validation policy.

Provider and SDK client identity, the complete prepared request, positional
arguments, validation context, and strictness contribute to cache keys.
Cached models are revalidated using the current call's context and strictness.
If request objects cannot be serialized reliably, the built-in cache is bypassed
for that call instead of reusing an ambiguous key.

Instructor supports caching for every client. Pass a cache adapter when you create the client. The cache parameter flows through to all provider implementations via **kwargs:

```python
from instructor import from_provider
from instructor.cache import AutoCache, DiskCache

# Works with any provider - cache flows through **kwargs automatically
client = from_provider("openai/gpt-6-luna", cache=AutoCache(maxsize=1000))
client = from_provider("anthropic/claude-haiku-5-5", cache=AutoCache(maxsize=1000))
client = from_provider("google/gemini-2.5-flash", cache=DiskCache(directory=".cache"))

# Your normal calls are now cached automatically
from pydantic import BaseModel


class User(BaseModel):
    name: str


first = client.create(
    messages=[{"role": "user", "content": "Hi."}], response_model=User
)
second = client.create(
    messages=[{"role": "user", "content": "Hi."}], response_model=User
)
assert first.name == second.name  # second call was served from cache
```

### `cache_ttl` per-call override

Pass `cache_ttl=<seconds>` alongside `cache=` if you want a result to
expire automatically:

```python
from instructor import from_provider
from instructor.cache import DiskCache
from pydantic import BaseModel


class User(BaseModel):
    name: str


cache = DiskCache(directory=".cache")
client = from_provider("openai/gpt-6-luna")

client.create(
    messages=[{"role": "user", "content": "Hi"}],
    response_model=User,
    cache=cache,
    cache_ttl=3600,  # 1 hour
)
```

If the underlying cache backend supports TTL (e.g. `DiskCache` does), the
entry will be evicted after the specified duration.  For `AutoCache` the
parameter is ignored.

DiskCache stores strings, bytes, and JSON-compatible values directly in SQLite.
It preserves TTLs and existing inline scalar entries. Older pickled or file-backed
records become cache misses. Use a private directory for cached responses.

### Cache-key design

The built-in cache uses `instructor.cache.make_request_cache_key` for the complete
prepared request. It includes client identity, namespace, validation context and
strictness in addition to the components below. Unsupported values bypass caching.

Components that influence the key:

| Part                        | Why it matters                               |
|-----------------------------|----------------------------------------------|
| `model`                     | Different model names can yield different answers |
| `messages` / `contents`     | The full chat history is hashed              |
| `mode`                      | JSON vs. TOOLS vs. RESPONSES changes formatting |
| `response_model` schema     | The entire `model_json_schema()` is included so **any** change in field names, types or *descriptions* busts the cache automatically |

Keys are SHA-256 hex digests of constant length. The lower-level
`make_cache_key` helper remains available for custom integrations, as shown below;
callers must supply all relevant identity and policy inputs themselves.

```python
from instructor.cache import make_cache_key
from pydantic import BaseModel


class User(BaseModel):
    name: str


key = make_cache_key(
    messages=[{"role": "user", "content": "hello"}],
    model="gpt-6-luna",
    response_model=User,
    mode="TOOLS",
)
print(key)  # → 9b8f5e2c8c9e…
#> 2e2a9521bd269d62ee9a8559d7deacba0025c1f6da0ec1fc63d472788be096fe
```

If you need custom behaviour (e.g. ignoring certain prompt fields) you can
write your own helper and pass a derived key into a bespoke cache adapter.

### Raw Response Reconstruction

For raw completion objects (used with `create_with_completion`), we use a `SimpleNamespace` trick to reconstruct the original object structure:

```python
from pydantic import BaseModel


class Completion(BaseModel):
    content: str
    usage: dict


# Example completion object
completion = Completion(content="Hello", usage={"tokens": 10})

# When caching:
raw_json = completion.model_dump_json()  # Serialize to JSON

# When restoring from cache:
import json
from types import SimpleNamespace

restored = json.loads(raw_json, object_hook=lambda d: SimpleNamespace(**d))
```

This approach allows us to restore the original dot-notation access patterns (e.g., `completion.usage.total_tokens`) without requiring the original class definitions. The `SimpleNamespace` objects behave identically to the original completion objects for attribute access while being much simpler to reconstruct from JSON.

## 1. `functools.cache` for Simple In-Memory Caching

**When to Use**: Good for functions with immutable arguments, called repeatedly with the same parameters in small to medium-sized applications. Use this when reusing the same data within a single session.

```python
import time
import functools
import instructor
from pydantic import BaseModel

client = instructor.from_provider("openai/gpt-6-luna")


class UserDetail(BaseModel):
    name: str
    age: int


@functools.cache
def extract(data) -> UserDetail:
    return client.create(
        response_model=UserDetail,
        messages=[
            {"role": "user", "content": data},
        ],
    )


start = time.perf_counter()  # (1)
model = extract("Extract jason is 25 years old")
print(f"Time taken: {time.perf_counter() - start}")
#> Time taken: 0.43337099999189377

start = time.perf_counter()
model = extract("Extract jason is 25 years old")  # (2)
print(f"Time taken: {time.perf_counter() - start}")
#> Time taken: 1.166015863418579e-06
```

1. Using `time.perf_counter()` to measure the time taken to run the function is better than using `time.time()` because it's more accurate and less susceptible to system clock changes.
2. The second time we call `extract`, the result is returned from the cache, and the function is not called.

!!! warning "Changing the Model does not Invalidate the Cache"

    Note that changing the model does not invalidate the cache. This is because the cache key is based on the function's name and arguments, not the model. This means that if we change the model, the cache will still return the old result.

Call `extract` multiple times with the same argument, and the result will be cached in memory for faster access.

**Benefits**: Easy to implement, fast access due to in-memory storage, and requires no additional libraries.

??? question "What is a decorator?"

    A decorator is a function that takes another function and extends the behavior of the latter function without explicitly modifying it. In Python, decorators are functions that take a function as an argument and return a closure.

    ```python hl_lines="3-5 9"
    def decorator(func):
        def wrapper(*args, **kwargs):
            print("Do something before")  # (1)
            #> Do something before
            result = func(*args, **kwargs)
            print("Do something after")  # (2)
            #> Do something after
            return result

        return wrapper


    @decorator
    def say_hello():
        #> Hello!
        print("Hello!")
        #> Hello!


    say_hello()
    #> "Do something before"
    #> "Hello!"
    #> "Do something after"
    ```

    1. The code is executed before the function is called
    2. The code is executed after the function is called

## 2. Persistent disk caching

Use the same `DiskCache` directory for persistent storage. To share entries
across clients serving the same account and endpoint, pass the same stable
`cache_namespace` on each request.

```python
from instructor import from_provider
from instructor.cache import DiskCache

cache = DiskCache("./my_cache_directory")
client = from_provider("openai/gpt-6-luna", cache=cache)
async_client = from_provider("openai/gpt-6-luna", async_client=True, cache=cache)
```

## 3. Shared Redis caching

Implement `BaseCache.get` and `BaseCache.set` to store Instructor's serialized
responses in Redis. Passing the TTL to Redis lets each request choose its expiry.

```python
from instructor import from_provider
from instructor.cache import BaseCache
from redis import Redis


class RedisCache(BaseCache):
    def __init__(self, store):
        self.store = store

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, ttl=None):
        self.store.set(key, value, ex=ttl)


store = Redis(host="localhost", decode_responses=True)
client = from_provider("openai/gpt-6-luna", cache=RedisCache(store))
```
