"""
Asyncio Benchmarks with Instructor

This script demonstrates and benchmarks different asyncio patterns for LLM processing:
- Sequential processing (baseline)
- asyncio.gather (concurrent, ordered results)
- asyncio.as_completed (concurrent, streaming results)
- Rate-limited processing with semaphores
- Error handling patterns
- Progress tracking
- Batch processing with chunking

Run this script to see performance comparisons and verify all code examples work.
"""

import asyncio
import time
import logging
import instructor
from pydantic import BaseModel, field_validator
from openai import AsyncOpenAI, OpenAI
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up the async client with Instructor
client = instructor.from_openai(AsyncOpenAI())
sync_client = instructor.from_openai(OpenAI())


class Person(BaseModel):
    name: str
    age: int
    occupation: str

    @field_validator("age")
    @classmethod
    def validate_age(cls, v):
        if v < 0 or v > 150:
            raise ValueError(f"Age {v} is invalid")
        return v


# Sample dataset
dataset = [
    "John Smith is a 30-year-old software engineer",
    "Sarah Johnson is a 25-year-old data scientist",
    "Mike Davis is a 35-year-old product manager",
    "Lisa Wilson is a 28-year-old UX designer",
    "Tom Brown is a 32-year-old DevOps engineer",
    "Emma Garcia is a 27-year-old frontend developer",
    "David Lee is a 33-year-old backend developer",
]


async def extract_person(text: str) -> Person:
    """Extract person information from text using LLM."""
    return await client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=Person,
        messages=[{"role": "user", "content": f"Extract person info: {text}"}],
    )


# Method 1: Sequential Processing (Baseline)
async def sequential_processing() -> tuple[list[Person], float]:
    """Process items one by one - slowest method."""
    start_time = time.time()
    persons = []

    for text in dataset:
        person = await extract_person(text)
        persons.append(person)
        print(f"Processed: {person.name}")

    end_time = time.time()
    duration = end_time - start_time
    print(f"Sequential processing took: {duration:.2f} seconds")
    return persons, duration


# Method 2: asyncio.gather - Concurrent Processing
async def gather_processing() -> tuple[list[Person], float]:
    """Process all items concurrently and return in order."""
    start_time = time.time()

    # Create tasks for all items
    tasks = [extract_person(text) for text in dataset]

    # Execute all tasks concurrently
    persons = await asyncio.gather(*tasks)

    end_time = time.time()
    duration = end_time - start_time
    print(f"asyncio.gather took: {duration:.2f} seconds")

    # Results maintain original order
    for person in persons:
        print(f"Processed: {person.name}")

    return persons, duration


# Method 3: asyncio.as_completed - Streaming Results
async def as_completed_processing() -> tuple[list[Person], float]:
    """Process items concurrently and handle results as they complete."""
    start_time = time.time()
    persons = []

    # Create tasks for all items
    tasks = [extract_person(text) for text in dataset]

    # Process results as they complete
    for task in asyncio.as_completed(tasks):
        person = await task
        persons.append(person)
        print(f"Completed: {person.name}")

    end_time = time.time()
    duration = end_time - start_time
    print(f"asyncio.as_completed took: {duration:.2f} seconds")
    return persons, duration


# Method 4: Rate-Limited Processing with Semaphores
async def rate_limited_extract_person(
    text: str, semaphore: asyncio.Semaphore
) -> Person:
    """Extract person info with rate limiting."""
    async with semaphore:
        return await extract_person(text)


async def rate_limited_gather(concurrency_limit: int = 3) -> tuple[list[Person], float]:
    """Process items with controlled concurrency using asyncio.gather."""
    start_time = time.time()

    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(concurrency_limit)

    # Create rate-limited tasks
    tasks = [rate_limited_extract_person(text, semaphore) for text in dataset]

    # Execute with rate limiting
    persons = await asyncio.gather(*tasks)

    end_time = time.time()
    duration = end_time - start_time
    print(
        f"Rate-limited gather (limit={concurrency_limit}) took: {duration:.2f} seconds"
    )
    return persons, duration


async def rate_limited_as_completed(
    concurrency_limit: int = 3,
) -> tuple[list[Person], float]:
    """Process items with controlled concurrency using asyncio.as_completed."""
    start_time = time.time()
    persons = []

    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(concurrency_limit)

    # Create rate-limited tasks
    tasks = [rate_limited_extract_person(text, semaphore) for text in dataset]

    # Process results as they complete
    for task in asyncio.as_completed(tasks):
        person = await task
        persons.append(person)
        print(f"Rate-limited completed: {person.name}")

    end_time = time.time()
    duration = end_time - start_time
    print(
        f"Rate-limited as_completed (limit={concurrency_limit}) took: {duration:.2f} seconds"
    )
    return persons, duration


# Advanced Patterns
async def robust_gather_processing() -> tuple[list[Person], float]:
    """Process items with error handling."""
    start_time = time.time()
    tasks = [extract_person(text) for text in dataset]

    # Execute with error handling
    results = await asyncio.gather(*tasks, return_exceptions=True)

    persons = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Error processing item {i}: {result}")
        else:
            persons.append(result)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Robust gather processing took: {duration:.2f} seconds")
    return persons, duration


async def timeout_gather_processing(
    timeout_seconds: float = 30.0,
) -> tuple[list[Person], float]:
    """Process items with timeout."""
    start_time = time.time()
    tasks = [extract_person(text) for text in dataset]

    try:
        persons = await asyncio.wait_for(
            asyncio.gather(*tasks), timeout=timeout_seconds
        )
        end_time = time.time()
        duration = end_time - start_time
        print(f"Timeout gather processing took: {duration:.2f} seconds")
        return persons, duration
    except asyncio.TimeoutError:
        end_time = time.time()
        duration = end_time - start_time
        print(
            f"Processing timed out after {timeout_seconds} seconds (took {duration:.2f}s)"
        )
        return [], duration


async def progress_tracking_processing() -> tuple[list[Person], float]:
    """Process items with progress tracking."""
    start_time = time.time()
    persons = []
    total_items = len(dataset)
    completed = 0

    tasks = [extract_person(text) for text in dataset]

    for task in asyncio.as_completed(tasks):
        person = await task
        persons.append(person)
        completed += 1
        print(
            f"Progress: {completed}/{total_items} ({completed / total_items * 100:.1f}%)"
        )

    end_time = time.time()
    duration = end_time - start_time
    print(f"Progress tracking processing took: {duration:.2f} seconds")
    return persons, duration


async def chunked_processing(chunk_size: int = 3) -> tuple[list[Person], float]:
    """Process items in chunks to manage memory and rate limits."""
    start_time = time.time()
    all_persons = []

    # Process in chunks
    for i in range(0, len(dataset), chunk_size):
        chunk = dataset[i : i + chunk_size]
        print(f"Processing chunk {i // chunk_size + 1}")

        tasks = [extract_person(text) for text in chunk]
        chunk_results = await asyncio.gather(*tasks)
        all_persons.extend(chunk_results)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Chunked processing took: {duration:.2f} seconds")
    return all_persons, duration


async def benchmark_all_methods():
    """Run all processing methods and compare performance."""
    print("=== Python asyncio.gather and asyncio.as_completed Performance Test ===\n")

    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set. Using mock responses for demonstration.")
        return

    # Test different methods
    methods = [
        ("Sequential", sequential_processing),
        ("asyncio.gather", gather_processing),
        ("asyncio.as_completed", as_completed_processing),
        ("Rate-limited gather (3)", lambda: rate_limited_gather(3)),
        ("Rate-limited as_completed (3)", lambda: rate_limited_as_completed(3)),
        ("Robust gather", robust_gather_processing),
        ("Timeout gather", timeout_gather_processing),
        ("Progress tracking", progress_tracking_processing),
        ("Chunked processing", chunked_processing),
    ]

    results = {}

    for name, method in methods:
        print(f"\n{'=' * 50}")
        print(f"Testing: {name}")
        print("=" * 50)

        try:
            persons, duration = await method()
            results[name] = {
                "count": len(persons),
                "duration": duration,
                "success": True,
            }
            print(f"✓ Success: {len(persons)} items processed in {duration:.2f}s")

            # Show first few results
            for person in persons[:3]:
                print(f"  - {person.name}, {person.age}, {person.occupation}")
            if len(persons) > 3:
                print(f"  ... and {len(persons) - 3} more")

        except Exception as e:
            results[name] = {
                "count": 0,
                "duration": 0,
                "success": False,
                "error": str(e),
            }
            print(f"✗ Failed: {e}")

    # Print summary table
    print(f"\n{'=' * 80}")
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"{'Method':<25} {'Items':<6} {'Time (s)':<10} {'Speed':<15} {'Status'}")
    print("-" * 80)

    for name, result in results.items():
        if result["success"]:
            speed = (
                f"{result['count'] / result['duration']:.1f} items/s"
                if result["duration"] > 0
                else "N/A"
            )
            status = "✓ Success"
        else:
            speed = "N/A"
            status = "✗ Failed"

        print(
            f"{name:<25} {result['count']:<6} {result['duration']:<10.2f} {speed:<15} {status}"
        )

    # Calculate speedup compared to sequential
    if "Sequential" in results and results["Sequential"]["success"]:
        baseline = results["Sequential"]["duration"]
        print(f"\nSpeedup compared to sequential processing:")
        for name, result in results.items():
            if name != "Sequential" and result["success"] and result["duration"] > 0:
                speedup = baseline / result["duration"]
                print(f"  {name}: {speedup:.1f}x faster")


def sync_example():
    """Show sync version for comparison."""
    print("\n" + "=" * 50)
    print("Sync Example (for comparison)")
    print("=" * 50)

    start_time = time.time()
    persons = []

    for text in dataset[:3]:  # Just first 3 for demo
        person = sync_client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=Person,
            messages=[{"role": "user", "content": f"Extract person info: {text}"}],
        )
        persons.append(person)
        print(f"Sync processed: {person.name}")

    end_time = time.time()
    duration = end_time - start_time
    print(f"Sync processing (3 items) took: {duration:.2f} seconds")


async def main():
    """Main function to run all examples."""
    try:
        await benchmark_all_methods()

        # Run sync example if API key is available
        if os.getenv("OPENAI_API_KEY"):
            sync_example()

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.exception("Unexpected error occurred")


if __name__ == "__main__":
    print("🚀 Starting asyncio benchmarks with Instructor...")
    print("💡 Make sure to set OPENAI_API_KEY environment variable")
    print("⏱️  This will take a few minutes to complete all benchmarks\n")

    asyncio.run(main())
