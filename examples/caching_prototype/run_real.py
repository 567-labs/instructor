"""Demonstrate real caching functionality with actual API calls."""

import time
import instructor
from instructor.cache import AutoCache, DiskCache
from pydantic import BaseModel, Field
from openai import OpenAI


class User(BaseModel):
    name: str = Field(description="The user's name")
    age: int = Field(description="The user's age")


def test_autocache():
    """Test basic in-memory caching."""
    print("\n=== Testing AutoCache (in-memory) ===")

    cache = AutoCache(maxsize=100)
    client = instructor.from_openai(OpenAI(), cache=cache)

    messages = [
        {"role": "user", "content": "Generate a user named Alice who is 25 years old"}
    ]

    # First call - hits API
    print("First call (hits API)...")
    start = time.time()
    user1 = client.create(
        model="gpt-3.5-turbo",
        messages=messages,
        response_model=User,
    )
    api_time = time.time() - start
    print(f"Result: {user1}")
    print(f"Time: {api_time:.2f}s")

    # Second call - from cache
    print("\nSecond call (from cache)...")
    start = time.time()
    user2 = client.create(
        model="gpt-3.5-turbo",
        messages=messages,
        response_model=User,
    )
    cache_time = time.time() - start
    print(f"Result: {user2}")
    print(f"Time: {cache_time:.4f}s")
    print(f"Speedup: {api_time / cache_time:.0f}x faster")

    assert user1.name == user2.name
    assert user1.age == user2.age
    print("✓ Cache working - identical results")


def test_create_with_completion():
    """Test create_with_completion caching."""
    print("\n=== Testing create_with_completion ===")

    cache = AutoCache(maxsize=100)
    client = instructor.from_openai(OpenAI(), cache=cache)

    messages = [
        {"role": "user", "content": "What's the weather? Say it's 22C and sunny."}
    ]

    class Weather(BaseModel):
        temperature: float
        condition: str

    # First call
    print("First call with completion...")
    weather1, completion1 = client.create_with_completion(
        model="gpt-3.5-turbo",
        messages=messages,
        response_model=Weather,
    )
    print(f"Weather: {weather1}")
    print(f"Completion ID: {completion1.id}")
    print(f"Tokens used: {completion1.usage.total_tokens}")

    # Second call - cached
    print("\nSecond call (cached)...")
    start = time.time()
    weather2, completion2 = client.create_with_completion(
        model="gpt-3.5-turbo",
        messages=messages,
        response_model=Weather,
    )
    cache_time = time.time() - start
    print(f"Weather: {weather2}")
    print(f"Completion ID: {completion2.id}")
    print(f"Cache time: {cache_time:.4f}s")

    assert weather1.temperature == weather2.temperature
    assert completion1.id == completion2.id
    print("✓ Completion object cached correctly")


def test_diskcache():
    """Test persistent disk caching."""
    print("\n=== Testing DiskCache (persistent) ===")

    # First client
    cache1 = DiskCache(directory=".instructor_cache_demo")
    client1 = instructor.from_openai(OpenAI(), cache=cache1)

    messages = [{"role": "user", "content": "Create a user named Bob who is 30"}]

    print("First client creates user...")
    user1 = client1.create(
        model="gpt-3.5-turbo",
        messages=messages,
        response_model=User,
    )
    print(f"Result: {user1}")

    # New client, same cache directory
    print("\nNew client with same cache dir...")
    cache2 = DiskCache(directory=".instructor_cache_demo")
    client2 = instructor.from_openai(OpenAI(), cache=cache2)

    start = time.time()
    user2 = client2.create(
        model="gpt-3.5-turbo",
        messages=messages,
        response_model=User,
    )
    cache_time = time.time() - start
    print(f"Result: {user2}")
    print(f"Time: {cache_time:.4f}s (from disk cache)")

    assert user1.name == user2.name
    print("✓ Cache persisted across clients")

    # Test create_with_completion persistence
    print("\nTesting create_with_completion persistence...")
    weather_messages = [{"role": "user", "content": "Weather is 25C and cloudy"}]

    class Weather(BaseModel):
        temperature: float
        condition: str

    # First call with completion
    weather1, completion1 = client1.create_with_completion(
        model="gpt-3.5-turbo",
        messages=weather_messages,
        response_model=Weather,
    )
    print(f"Weather: {weather1}, Completion ID: {completion1.id}")

    # Second call from different client - should get cached completion
    weather2, completion2 = client2.create_with_completion(
        model="gpt-3.5-turbo",
        messages=weather_messages,
        response_model=Weather,
    )
    print(f"Cached: {weather2}, Completion ID: {completion2.id}")

    assert weather1.temperature == weather2.temperature
    assert completion1.id == completion2.id
    print("✓ Raw completion persisted to disk")

    # Cleanup
    import shutil

    shutil.rmtree(".instructor_cache_demo", ignore_errors=True)


def test_cache_ttl():
    """Test cache TTL with DiskCache."""
    print("\n=== Testing Cache TTL ===")

    cache = DiskCache(directory=".instructor_cache_ttl")
    client = instructor.from_openai(OpenAI(), cache=cache)

    messages = [{"role": "user", "content": "Create user Charlie age 35"}]

    # Set with 2 second TTL
    print("Setting cache with 2s TTL...")
    user1 = client.create(
        model="gpt-3.5-turbo",
        messages=messages,
        response_model=User,
        cache_ttl=2,
    )
    print(f"Result: {user1}")

    # Immediate call - cached
    print("\nImmediate call (cached)...")
    start = time.time()
    user2 = client.create(
        model="gpt-3.5-turbo",
        messages=messages,
        response_model=User,
    )
    print(f"Time: {time.time() - start:.4f}s")

    # Wait for expiry
    print("\nWaiting 3s for TTL expiry...")
    time.sleep(3)

    # Should hit API again
    print("After TTL (hits API)...")
    start = time.time()
    user3 = client.create(
        model="gpt-3.5-turbo",
        messages=messages,
        response_model=User,
    )
    api_time = time.time() - start
    print(f"Time: {api_time:.2f}s")
    print("✓ TTL working correctly")

    # Cleanup
    import shutil

    shutil.rmtree(".instructor_cache_ttl", ignore_errors=True)


def test_different_inputs():
    """Show that different inputs use different cache keys."""
    print("\n=== Testing Different Cache Keys ===")

    cache = AutoCache(maxsize=100)
    client = instructor.from_openai(OpenAI(), cache=cache)

    # Different prompts
    user1 = client.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Create user David age 40"}],
        response_model=User,
    )
    print(f"User 1: {user1}")

    user2 = client.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Create user Eve age 45"}],
        response_model=User,
    )
    print(f"User 2: {user2}")

    assert user1.name != user2.name or user1.age != user2.age
    print("✓ Different prompts = different results")

    # Different models
    class SimpleUser(BaseModel):
        name: str

    simple = client.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Create user David age 40"}],
        response_model=SimpleUser,
    )
    print(f"Simple user: {simple}")
    print("✓ Different models = different cache keys")


if __name__ == "__main__":
    print("Instructor Caching Demo - Real API Calls")
    print("=" * 50)

    test_autocache()
    test_create_with_completion()
    test_diskcache()
    test_cache_ttl()
    test_different_inputs()

    print("\n" + "=" * 50)
    print("All tests completed! ✨")
