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

Provider identity and prepared generation settings also contribute to cache keys.
Cached models are revalidated using the current call's context and strictness.
Opaque request objects are rejected with a `TypeError` when a cache key cannot be
constructed. Use serializable request settings or disable caching for that call;
arbitrary object string representations are not reliable cache identities.

Instructor supports caching for every client. Pass a cache adapter when you create the client. The cache parameter flows through to all provider implementations via **kwargs:

```python
from instructor import from_provider
from instructor.cache import AutoCache, DiskCache

# Works with any provider - cache flows through **kwargs automatically
client = from_provider("openai/gpt-4.1-mini", cache=AutoCache(maxsize=1000))
client = from_provider("anthropic/claude-3-haiku", cache=AutoCache(maxsize=1000))
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
client = from_provider("openai/gpt-4.1-mini")

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

### Cache-key design

Under the hood Instructor generates a **deterministic** key for every
 call using `instructor.cache.make_cache_key`.

Components that influence the key:

| Part                        | Why it matters                               |
|-----------------------------|----------------------------------------------|
| `model`                     | Different model names can yield different answers |
| `messages` / `contents`     | The full chat history is hashed              |
| `mode`                      | JSON vs. TOOLS vs. RESPONSES changes formatting |
| `response_model` schema     | The entire `model_json_schema()` is included so **any** change in field names, types or *descriptions* busts the cache automatically |

The function returns a SHA-256 hex digest; its length is constant regardless
of prompt size, so it is safe to use as a Redis key, file path, etc.

```python
from instructor.cache import make_cache_key
from pydantic import BaseModel


class User(BaseModel):
    name: str


key = make_cache_key(
    messages=[{"role": "user", "content": "hello"}],
    model="gpt-4.1-mini",
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

client = instructor.from_provider("openai/gpt-4.1-mini")


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

## 2. `diskcache` for Persistent, Large Data Caching

??? note "Copy Caching Code"

    The same `instructor_cache` decorator works for both `diskcache` and `redis` caching. Copy the code below and use it for both examples.

    ```python
    import functools
    import inspect
    import diskcache

    cache = diskcache.Cache('./my_cache_directory')  # (1)


    def instructor_cache(func):
        """Cache a function that returns a Pydantic model"""
        return_type = inspect.signature(func).return_annotation
        if not issubclass(return_type, BaseModel):  # (2)
            raise ValueError("The return type must be a Pydantic model")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}-{functools._make_key(args, kwargs, typed=False)}"
            # Check if the result is already cached
            if (cached := cache.get(key)) is not None:
                # Deserialize from JSON based on the return type
                return return_type.model_validate_json(cached)

            # Call the function and cache its result
            result = func(*args, **kwargs)
            serialized_result = result.model_dump_json()
            cache.set(key, serialized_result)

            return result

        return wrapper
    ```

    1. We create a new `diskcache.Cache` instance to store the cached data. This will create a new directory called `my_cache_directory` in the current working directory.
    2. We only want to cache functions that return a Pydantic model to simplify serialization and deserialization logic in this example code

    Remember that you can change this code to support non-Pydantic models, or to use a different caching backend. More over, don't forget that this cache does not invalidate when the model changes, so you might want to encode the `Model.model_json_schema()` as part of the key.

**When to Use**: Good for applications that need cache persistence between sessions or deal with large datasets. Use this when you want to reuse the same data across multiple sessions or store large amounts of data.

```python hl_lines="10"
import functools
import inspect
import instructor
import diskcache
from pydantic import BaseModel

client = instructor.from_provider("openai/gpt-4.1-mini")
cache = diskcache.Cache('./my_cache_directory')


def instructor_cache(func):
    """Cache a function that returns a Pydantic model"""
    return_type = inspect.signature(func).return_annotation  # (4)
    if not issubclass(return_type, BaseModel):  # (1)
        raise ValueError("The return type must be a Pydantic model")

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = (
            f"{func.__name__}-{functools._make_key(args, kwargs, typed=False)}"  #  (2)
        )
        # Check if the result is already cached
        if (cached := cache.get(key)) is not None:
            # Deserialize from JSON based on the return type (3)
            return return_type.model_validate_json(cached)

        # Call the function and cache its result
        result = func(*args, **kwargs)
        serialized_result = result.model_dump_json()
        cache.set(key, serialized_result)

        return result

    return wrapper


class UserDetail(BaseModel):
    name: str
    age: int


@instructor_cache
def extract(data) -> UserDetail:
    return client.create(
        response_model=UserDetail,
        messages=[
            {"role": "user", "content": data},
        ],
    )
```

1. We only want to cache functions that return a Pydantic model to simplify serialization and deserialization logic
2. We use functool's `_make_key` to generate a unique key based on the function's name and arguments. This is important because we want to cache the result of each function call separately.
3. We use Pydantic's `model_validate_json` to deserialize the cached result into a Pydantic model.
4. We use `inspect.signature` to get the function's return type annotation, which we use to validate the cached result.

**Benefits**: Reduces computation time for heavy data processing and provides disk-based caching for persistence.

## 3. Redis Caching Decorator for Distributed Systems

??? note "Copy Caching Code"

    The same `instructor_cache` decorator works for both `diskcache` and `redis` caching. Copy the code below and use it for both examples.

    ```python
    import functools
    import inspect
    import redis

    cache = redis.Redis("localhost")


    def instructor_cache(func):
        """Cache a function that returns a Pydantic model"""
        return_type = inspect.signature(func).return_annotation
        if not issubclass(return_type, BaseModel):
            raise ValueError("The return type must be a Pydantic model")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}-{functools._make_key(args, kwargs, typed=False)}"
            # Check if the result is already cached
            if (cached := cache.get(key)) is not None:
                # Deserialize from JSON based on the return type
                return return_type.model_validate_json(cached)

            # Call the function and cache its result
            result = func(*args, **kwargs)
            serialized_result = result.model_dump_json()
            cache.set(key, serialized_result)

            return result

        return wrapper
    ```

    Remember that you can change this code to support non-Pydantic models, or to use a different caching backend. More over, don't forget that this cache does not invalidate when the model changes, so you might want to encode the `Model.model_json_schema()` as part of the key.

**When to Use**: Good for distributed systems where multiple processes need to access cached data, or for applications that need fast read/write access and handle complex data structures.

```python
import redis
import functools
import inspect
import instructor

from pydantic import BaseModel

client = instructor.from_provider("openai/gpt-4.1-mini")
cache = redis.Redis("localhost")


def instructor_cache(func):
    """Cache a function that returns a Pydantic model"""
    return_type = inspect.signature(func).return_annotation
    if not issubclass(return_type, BaseModel):  # (1)
        raise ValueError("The return type must be a Pydantic model")

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = f"{func.__name__}-{functools._make_key(args, kwargs, typed=False)}"  # (2)
        # Check if the result is already cached
        if (cached := cache.get(key)) is not None:
            # Deserialize from JSON based on the return type
            return return_type.model_validate_json(cached)

        # Call the function and cache its result
        result = func(*args, **kwargs)
        serialized_result = result.model_dump_json()
        cache.set(key, serialized_result)

        return result

    return wrapper


class UserDetail(BaseModel):
    name: str
    age: int


@instructor_cache
def extract(data) -> UserDetail:
    # Assuming client.chat.completions.create returns a UserDetail instance
    return client.create(
        response_model=UserDetail,
        messages=[
            {"role": "user", "content": data},
        ],
    )
```

1. We only want to cache functions that return a Pydantic model to simplify serialization and deserialization logic
2. We use functool's `_make_key` to generate a unique key based on the function's name and arguments. This is important because we want to cache the result of each function call separately.

**Benefits**: Scalable for large-scale systems, supports fast in-memory data storage and retrieval, and works with various data types.

!!! note "Same Decorator, Different Backend"

    The code above uses the same `instructor_cache` decorator as before. The implementation is the same, but it uses a different caching backend.
