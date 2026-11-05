"""
Comprehensive Caching Example for Instructor
===========================================

This example demonstrates various caching strategies for LLM applications:
1. functools.cache - Simple in-memory caching
2. diskcache - Persistent disk-based caching
3. Redis - Distributed caching
4. Performance benchmarks and cost analysis
5. Advanced patterns: hierarchical caching, monitoring, schema invalidation

Run this example to see real-world performance improvements and cost savings.
"""

import asyncio
import functools
import inspect
import logging
import time
from collections import defaultdict
from typing import Any, Callable, Optional, TypeVar

import instructor
from instructor.cache import DiskCache, make_cache_key
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients
client = instructor.from_openai(OpenAI())
aclient = instructor.from_openai(AsyncOpenAI())

# Test data
TEST_QUERIES = [
    "Extract: Jason is 25 years old and works as a software engineer",
    "Extract: Sarah is 30 years old and is a data scientist",
    "Extract: Mike is 28 years old and works in marketing",
    "Extract: Lisa is 32 years old and is a product manager",
    "Extract: Jason is 25 years old and works as a software engineer",  # Duplicate for cache hit
]

F = TypeVar("F", bound=Callable[..., Any])


class UserDetail(BaseModel):
    """Enhanced user model with more fields for testing"""

    name: str = Field(description="User's full name")
    age: int = Field(description="User's age", ge=0, le=150)
    occupation: Optional[str] = Field(None, description="User's job title")


class CacheMetrics:
    """Production-ready cache monitoring"""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.total_time_saved = 0.0
        self.error_count = 0
        self.hit_rate_by_function: dict[str, dict[str, int]] = defaultdict(
            lambda: {"hits": 0, "misses": 0}
        )

    def record_hit(self, func_name: str, time_saved: float):
        self.hits += 1
        self.total_time_saved += time_saved
        self.hit_rate_by_function[func_name]["hits"] += 1
        logger.debug(f"Cache HIT for {func_name}, saved {time_saved:.3f}s")

    def record_miss(self, func_name: str):
        self.misses += 1
        self.hit_rate_by_function[func_name]["misses"] += 1
        logger.debug(f"Cache MISS for {func_name}")

    def record_error(self, func_name: str, error: str):
        self.error_count += 1
        logger.warning(f"Cache ERROR in {func_name}: {error}")

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def get_stats(self) -> dict[str, Any]:
        return {
            "hit_rate": f"{self.hit_rate:.2%}",
            "total_hits": self.hits,
            "total_misses": self.misses,
            "error_count": self.error_count,
            "time_saved_seconds": f"{self.total_time_saved:.3f}",
            "function_stats": dict(self.hit_rate_by_function),
        }

    def reset(self):
        """Reset all metrics for new test runs"""
        self.hits = 0
        self.misses = 0
        self.total_time_saved = 0.0
        self.error_count = 0
        self.hit_rate_by_function.clear()


# Global metrics instance
metrics = CacheMetrics()


def smart_cache_key(
    func_name: str, args: tuple, kwargs: dict, model_class: type[BaseModel]
) -> str:
    return make_cache_key(
        namespace=func_name,
        messages=args,
        request_kwargs=kwargs,
        model=None,
        response_model=model_class,
    )


# 1. Simple functools.cache implementation
@functools.lru_cache(maxsize=1000)
def extract_functools(data: str) -> UserDetail:
    """Simple in-memory caching with functools.lru_cache"""
    start_time = time.perf_counter()

    result = client.chat.completions.create(
        model="gpt-4.1-mini",
        response_model=UserDetail,
        messages=[
            {"role": "user", "content": data},
        ],
    )

    # This won't be called on cache hits, so we track metrics differently
    return result


def monitored_functools_cache(func: F) -> F:
    """functools.cache with monitoring"""
    cached_func = functools.lru_cache(maxsize=1000)(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check if we'll get a cache hit by calling cache_info
        info_before = cached_func.cache_info()

        start_time = time.perf_counter()
        result = cached_func(*args, **kwargs)
        execution_time = time.perf_counter() - start_time

        info_after = cached_func.cache_info()

        if info_after.hits > info_before.hits:
            # We got a cache hit
            metrics.record_hit(func.__name__, 0.8)  # Assume 800ms saved
        else:
            # Cache miss
            metrics.record_miss(func.__name__)

        return result

    # Preserve cache_info method
    wrapper.cache_info = cached_func.cache_info
    wrapper.cache_clear = cached_func.cache_clear

    return wrapper


@monitored_functools_cache
def extract_functools_monitored(data: str) -> UserDetail:
    """functools.cache with monitoring"""
    return client.chat.completions.create(
        model="gpt-4.1-mini",
        response_model=UserDetail,
        messages=[
            {"role": "user", "content": data},
        ],
    )


# 2. Enhanced diskcache implementation
def create_diskcache_decorator(
    cache_dir: str = "./cache_directory", ttl: Optional[int] = None
):
    """Factory for diskcache decorator with enhanced features"""
    try:
        cache = DiskCache(cache_dir)
    except ImportError:
        logger.warning("diskcache not available, skipping disk cache example")
        return lambda func: func

    def decorator(func: F) -> F:
        return_type = inspect.signature(func).return_annotation
        if not (inspect.isclass(return_type) and issubclass(return_type, BaseModel)):
            raise ValueError("The return type must be a Pydantic model")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate smart cache key with schema versioning
            key = smart_cache_key(func.__name__, args, kwargs, return_type)

            try:
                # Check if the result is already cached
                if (cached := cache.get(key)) is not None:
                    metrics.record_hit(func.__name__, 0.8)  # Assume 800ms saved
                    return return_type.model_validate_json(cached)

                metrics.record_miss(func.__name__)
            except Exception as e:
                metrics.record_error(func.__name__, type(e).__name__)
                logger.warning("Cache read error (%s)", type(e).__name__)

            # Call the function and cache its result
            result = func(*args, **kwargs)

            try:
                serialized_result = result.model_dump_json()
                cache.set(key, serialized_result, ttl=ttl)
            except Exception as e:
                metrics.record_error(func.__name__, type(e).__name__)
                logger.warning("Cache write error (%s)", type(e).__name__)

            return result

        return wrapper

    return decorator


@create_diskcache_decorator(ttl=3600)  # 1 hour TTL
def extract_diskcache(data: str) -> UserDetail:
    """Persistent disk-based caching with TTL"""
    return client.chat.completions.create(
        model="gpt-4.1-mini",
        response_model=UserDetail,
        messages=[
            {"role": "user", "content": data},
        ],
    )


# 3. Enhanced Redis implementation (with fallback)
def create_redis_decorator(
    redis_url: str = "redis://localhost:6379",
    ttl: int = 3600,
    prefix: str = "instructor",
):
    """Factory for Redis decorator with production features"""
    try:
        import redis

        cache = redis.from_url(redis_url, decode_responses=True)
        # Test connection
        cache.ping()
        logger.info("Connected to Redis successfully")
    except ImportError as e:
        logger.warning(f"Redis not available (ImportError: {e}), using fallback")
        return lambda func: func
    except Exception as e:  # Covers redis.RedisError and other connection issues
        logger.warning(f"Redis not available ({e}), using fallback")
        return lambda func: func

    def decorator(func: F) -> F:
        return_type = inspect.signature(func).return_annotation
        if not (inspect.isclass(return_type) and issubclass(return_type, BaseModel)):
            raise ValueError("The return type must be a Pydantic model")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key with schema versioning
            key = (
                f"{prefix}:{smart_cache_key(func.__name__, args, kwargs, return_type)}"
            )

            try:
                # Check if the result is already cached
                if (cached := cache.get(key)) is not None:
                    metrics.record_hit(func.__name__, 0.8)  # Assume 800ms saved
                    logger.debug(f"Cache hit for key: {key}")
                    return return_type.model_validate_json(cached)

                metrics.record_miss(func.__name__)
                logger.debug(f"Cache miss for key: {key}")
            except redis.RedisError as e:
                metrics.record_error(func.__name__, type(e).__name__)
                logger.warning("Redis read error (%s)", type(e).__name__)

            # Call the function and cache its result
            result = func(*args, **kwargs)

            try:
                serialized_result = result.model_dump_json()
                cache.setex(key, ttl, serialized_result)
                logger.debug(f"Cached result for key: {key}")
            except redis.RedisError as e:
                metrics.record_error(func.__name__, type(e).__name__)
                logger.warning("Redis write error (%s)", type(e).__name__)

            return result

        return wrapper

    return decorator


@create_redis_decorator(ttl=3600)
def extract_redis(data: str) -> UserDetail:
    """Distributed Redis caching with error handling"""
    return client.chat.completions.create(
        model="gpt-4.1-mini",
        response_model=UserDetail,
        messages=[
            {"role": "user", "content": data},
        ],
    )


# 4. No cache baseline for comparison
def extract_no_cache(data: str) -> UserDetail:
    """Baseline function without caching"""
    metrics.record_miss("extract_no_cache")
    return client.chat.completions.create(
        model="gpt-4.1-mini",
        response_model=UserDetail,
        messages=[
            {"role": "user", "content": data},
        ],
    )


# 5. Hierarchical caching example
@functools.lru_cache(maxsize=50)  # L1: Fast in-memory
def extract_l1(data: str) -> UserDetail:
    return extract_l2(data)


@create_diskcache_decorator()  # L2: Persistent disk
def extract_l2(data: str) -> UserDetail:
    return extract_l3(data)


@create_redis_decorator()  # L3: Shared distributed
def extract_l3(data: str) -> UserDetail:
    return client.chat.completions.create(
        model="gpt-4.1-mini",
        response_model=UserDetail,
        messages=[
            {"role": "user", "content": data},
        ],
    )


def benchmark_caching_strategy(
    func: Callable, name: str, queries: list[str]
) -> dict[str, Any]:
    """Benchmark a specific caching strategy"""
    logger.info(f"\n=== Benchmarking {name} ===")

    # Reset metrics for this test
    metrics.reset()

    times = []
    results = []

    for i, query in enumerate(queries):
        start_time = time.perf_counter()
        try:
            result = func(query)
            execution_time = time.perf_counter() - start_time
            times.append(execution_time)
            results.append(result)
            logger.info(
                f"Query {i + 1}: {execution_time:.3f}s - {result.name}, {result.age}, {result.occupation}"
            )
        except Exception as e:
            logger.error(f"Error in {name}: {e}")
            times.append(float("inf"))
            results.append(None)

    # Calculate statistics
    valid_times = [t for t in times if t != float("inf")]
    if valid_times:
        avg_time = sum(valid_times) / len(valid_times)
        total_time = sum(valid_times)
        fastest_time = min(valid_times)
        slowest_time = max(valid_times)
    else:
        avg_time = total_time = fastest_time = slowest_time = 0

    stats = {
        "name": name,
        "total_time": total_time,
        "avg_time": avg_time,
        "fastest_time": fastest_time,
        "slowest_time": slowest_time,
        "cache_metrics": metrics.get_stats(),
        "success_rate": len(valid_times) / len(queries),
    }

    logger.info(f"Total time: {total_time:.3f}s")
    logger.info(f"Average time: {avg_time:.3f}s")
    logger.info(f"Cache hit rate: {metrics.hit_rate:.2%}")

    return stats


def calculate_cost_savings(baseline_stats: dict, cached_stats: dict) -> dict[str, Any]:
    """Calculate cost savings from caching"""
    baseline_time = baseline_stats["total_time"]
    cached_time = cached_stats["total_time"]

    # Assume $0.002 per API call (rough average)
    cost_per_call = 0.002
    num_queries = len(TEST_QUERIES)

    # Without caching: every call costs money
    cost_without_cache = num_queries * cost_per_call

    # With caching: only cache misses cost money
    cache_misses = cached_stats["cache_metrics"]["total_misses"]
    cost_with_cache = cache_misses * cost_per_call

    savings = cost_without_cache - cost_with_cache
    savings_percent = (
        (savings / cost_without_cache) * 100 if cost_without_cache > 0 else 0
    )

    time_saved = baseline_time - cached_time
    time_savings_percent = (
        (time_saved / baseline_time) * 100 if baseline_time > 0 else 0
    )

    return {
        "cost_without_cache": cost_without_cache,
        "cost_with_cache": cost_with_cache,
        "cost_savings": savings,
        "cost_savings_percent": savings_percent,
        "time_saved": time_saved,
        "time_savings_percent": time_savings_percent,
        "speed_improvement": (
            baseline_time / cached_time if cached_time > 0 else float("inf")
        ),
    }


async def run_async_example():
    """Demonstrate async caching patterns"""
    logger.info("\n=== Async Caching Example ===")

    # Simple async function with metrics
    async def extract_async(data: str) -> UserDetail:
        metrics.record_miss("extract_async")
        return await aclient.chat.completions.create(
            model="gpt-4.1-mini",
            response_model=UserDetail,
            messages=[
                {"role": "user", "content": data},
            ],
        )

    # Run concurrent requests
    start_time = time.perf_counter()
    tasks = [
        extract_async(query) for query in TEST_QUERIES[:3]
    ]  # First 3 to save costs
    results = await asyncio.gather(*tasks)
    total_time = time.perf_counter() - start_time

    logger.info(f"Async processing time: {total_time:.3f}s")
    for i, result in enumerate(results):
        logger.info(f"Result {i + 1}: {result.name}, {result.age}, {result.occupation}")


def demonstrate_schema_invalidation():
    """Show how cache keys change when model schema changes"""
    logger.info("\n=== Schema-Based Cache Invalidation ===")

    # Original model
    class OriginalUser(BaseModel):
        name: str
        age: int

    # Modified model (different schema)
    class ModifiedUser(BaseModel):
        name: str
        age: int
        email: Optional[str] = None  # New field

    # Generate cache keys for same function args but different models
    args = ("test data",)
    kwargs = {}

    key1 = smart_cache_key("test_func", args, kwargs, OriginalUser)
    key2 = smart_cache_key("test_func", args, kwargs, ModifiedUser)

    logger.info(f"Original model cache key: {key1}")
    logger.info(f"Modified model cache key: {key2}")
    logger.info(f"Keys are different: {key1 != key2}")
    logger.info("This ensures cache invalidation when model schemas change!")


def main():
    """Run comprehensive caching demonstration"""
    logger.info("🚀 Starting Comprehensive Caching Demonstration")
    logger.info("=" * 60)

    # Run benchmarks for each strategy
    strategies = [
        (extract_no_cache, "No Cache (Baseline)"),
        (extract_functools_monitored, "functools.lru_cache"),
        (extract_diskcache, "diskcache"),
        (extract_redis, "Redis"),
        (extract_l1, "Hierarchical (L1→L2→L3)"),
    ]

    all_stats = {}

    for func, name in strategies:
        try:
            stats = benchmark_caching_strategy(func, name, TEST_QUERIES)
            all_stats[name] = stats
        except Exception as e:
            logger.error(f"Failed to benchmark {name}: {e}")
            continue

    # Print summary comparison
    logger.info("\n" + "=" * 60)
    logger.info("📊 PERFORMANCE COMPARISON SUMMARY")
    logger.info("=" * 60)

    baseline_stats = all_stats.get("No Cache (Baseline)")

    if baseline_stats:
        for name, stats in all_stats.items():
            if name == "No Cache (Baseline)":
                continue

            logger.info(f"\n{name}:")
            logger.info(f"  Total time: {stats['total_time']:.3f}s")
            logger.info(f"  Cache hit rate: {stats['cache_metrics']['hit_rate']}")

            # Calculate savings
            savings = calculate_cost_savings(baseline_stats, stats)
            logger.info(f"  Speed improvement: {savings['speed_improvement']:.1f}x")
            logger.info(
                f"  Time saved: {savings['time_saved']:.3f}s ({savings['time_savings_percent']:.1f}%)"
            )
            logger.info(
                f"  Cost savings: ${savings['cost_savings']:.4f} ({savings['cost_savings_percent']:.1f}%)"
            )

    # Additional demonstrations
    demonstrate_schema_invalidation()

    # Run async example
    asyncio.run(run_async_example())

    # Print cache info for functools
    logger.info(
        f"\nfunctools.lru_cache info: {extract_functools_monitored.cache_info()}"
    )

    logger.info("\n" + "=" * 60)
    logger.info("✅ Caching demonstration completed!")
    logger.info("💡 Key takeaways:")
    logger.info("  - Caching can provide 10x-1000x speed improvements")
    logger.info("  - Choose the right strategy based on your needs:")
    logger.info("    • functools.cache: Development, single process")
    logger.info("    • diskcache: Persistence, moderate performance")
    logger.info("    • Redis: Distributed systems, high performance")
    logger.info("    • Hierarchical: Best of all worlds")
    logger.info("  - Smart cache keys prevent stale data")
    logger.info("  - Monitoring helps optimize cache performance")


if __name__ == "__main__":
    main()
