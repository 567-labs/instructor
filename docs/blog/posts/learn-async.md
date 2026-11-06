---
authors:
- jxnl
categories:
- LLM Techniques
comments: true
date: 2023-11-13
description: "Master Python asyncio.gather and asyncio.as_completed for efficient concurrent LLM processing with Instructor. Learn async programming patterns, rate limiting, and performance optimization for AI applications."
draft: false
slug: learn-async
tags:
- asyncio
- asyncio.gather
- asyncio.as_completed
- OpenAI
- Python
- data processing
- async programming
- concurrent processing
- LLM optimization
---

# Mastering Python asyncio.gather and asyncio.as_completed for LLM Processing

Learn how to use Python's `asyncio.gather` and `asyncio.as_completed` for efficient concurrent processing of Large Language Models (LLMs) with Instructor. This comprehensive guide covers async programming patterns, rate limiting strategies, and performance optimization techniques.

<!-- more -->

!!! notes "Complete Example Code"

    You can find the complete working example on [GitHub](https://github.com/jxnl/instructor/blob/main/examples/learn-async/run.py)

## Understanding asyncio.gather vs asyncio.as_completed

Python's `asyncio` library provides two powerful methods for concurrent execution:

- **`asyncio.gather`**: Executes all tasks concurrently and returns results in the same order as input
- **`asyncio.as_completed`**: Returns results as they complete, regardless of input order

Both methods significantly outperform sequential processing, but they serve different use cases.

## Complete Setup: Async LLM Processing

Here's a complete, self-contained example showing how to set up async processing with Instructor:

```python
import instructor
from pydantic import BaseModel

# Set up the async client with Instructor
client = instructor.from_provider("openai/gpt-5-nano", async_client=True)


class Person(BaseModel):
    name: str
    age: int
    occupation: str


async def extract_person(text: str) -> Person:
    """Extract person information from text using LLM."""
    return await client.create(
        model="gpt-4o-mini",
        response_model=Person,
        messages=[{"role": "user", "content": f"Extract person info: {text}"}],
    )


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
```

## Method 1: Sequential Processing (Baseline)

```python
async def sequential_processing() -> List[Person]:
    """Process items one by one - slowest method."""
    start_time = time.time()
    persons = []

    for text in dataset:
        person = await extract_person(text)
        persons.append(person)
        print(f"Processed: {person.name}")

    end_time = time.time()
    print(f"Sequential processing took: {end_time - start_time:.2f} seconds")
    return persons


# Run sequential processing
# persons = await sequential_processing()
```

## Method 2: asyncio.gather - Concurrent Processing

```python
async def gather_processing() -> List[Person]:
    """Process all items concurrently and return in order."""
    start_time = time.time()

    # Create tasks for all items
    tasks = [extract_person(text) for text in dataset]

    # Execute all tasks concurrently
    persons = await asyncio.gather(*tasks)

    end_time = time.time()
    print(f"asyncio.gather took: {end_time - start_time:.2f} seconds")

    # Results maintain original order
    for person in persons:
        print(f"Processed: {person.name}")

    return persons


# Run gather processing
# persons = await gather_processing()
```

## Method 3: asyncio.as_completed - Streaming Results

```python
async def as_completed_processing() -> List[Person]:
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
    print(f"asyncio.as_completed took: {end_time - start_time:.2f} seconds")
    return persons


# Run as_completed processing
# persons = await as_completed_processing()
```

## Method 4: Rate-Limited Processing with Semaphores

```python
async def rate_limited_extract_person(
    text: str, semaphore: asyncio.Semaphore
) -> Person:
    """Extract person info with rate limiting."""
    async with semaphore:
        return await extract_person(text)


async def rate_limited_gather(concurrency_limit: int = 3) -> List[Person]:
    """Process items with controlled concurrency using asyncio.gather."""
    start_time = time.time()

    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(concurrency_limit)

    # Create rate-limited tasks
    tasks = [rate_limited_extract_person(text, semaphore) for text in dataset]

    # Execute with rate limiting
    persons = await asyncio.gather(*tasks)

    end_time = time.time()
    print(
        f"Rate-limited gather (limit={concurrency_limit}) took: {end_time - start_time:.2f} seconds"
    )
    return persons


async def rate_limited_as_completed(concurrency_limit: int = 3) -> List[Person]:
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
    print(
        f"Rate-limited as_completed (limit={concurrency_limit}) took: {end_time - start_time:.2f} seconds"
    )
    return persons


# Run rate-limited processing
# persons = await rate_limited_gather(concurrency_limit=2)
# persons = await rate_limited_as_completed(concurrency_limit=2)
```

## Performance Comparison

Here are typical performance results when processing 7 items:

| Method | Execution Time | Concurrency | Use Case |
|--------|---------------|-------------|----------|
| Sequential | 6.17 seconds | 1 | Baseline |
| asyncio.gather | 0.85 seconds | 7 | Fast processing, ordered results |
| asyncio.as_completed | 0.95 seconds | 7 | Streaming results |
| Rate-limited gather | 3.04 seconds | 2 | API-friendly |
| Rate-limited as_completed | 3.26 seconds | 2 | Streaming + rate limiting |

## When to Use Each Method

### Use asyncio.gather when:
- You need results in the same order as input
- All tasks must complete successfully
- You want the fastest possible execution
- Memory usage isn't a concern

### Use asyncio.as_completed when:
- You want to process results as they arrive
- Order doesn't matter
- You're streaming data to clients
- You want to handle large datasets efficiently

### Use rate limiting when:
- Working with API rate limits
- Being respectful to external services
- Managing resource consumption
- Building production applications

## Key Takeaways

1. **asyncio.gather** is fastest for ordered results
2. **asyncio.as_completed** is best for streaming and large datasets
3. **Rate limiting** is essential for production applications
4. **Error handling** should be implemented for robustness
5. **Monitoring** helps optimize performance

## Related Resources

- [Python asyncio Documentation](https://docs.python.org/3/library/asyncio.html)
- [Real Python Async IO Tutorial](https://realpython.com/async-io-python/)
- [Instructor Documentation](https://python.useinstructor.com)
- [OpenAI Async API Guide](https://platform.openai.com/docs/guides/async)

---

**Next Steps**: Learn about [error handling patterns](../../concepts/error_handling.md) or explore [rate limiting with tenacity](../../concepts/retrying.md) for production applications.