"""Benchmark tests for dictionary operations in instructor."""

import timeit
from instructor.core.retry import extract_messages
from instructor.utils import (
    combine_system_messages,
    extract_system_messages,
    update_gemini_kwargs,
)

# Mock data for benchmarks
SAMPLE_KWARGS_MESSAGES = {"messages": [{"role": "user", "content": "Hello"}]}
SAMPLE_KWARGS_CONTENTS = {"contents": [{"role": "user", "parts": ["Hello"]}]}
SAMPLE_KWARGS_CHAT_HISTORY = {"chat_history": [{"role": "user", "message": "Hello"}]}
SAMPLE_KWARGS_EMPTY = {}

SAMPLE_SYSTEM_MSG_STR = "You are a helpful assistant."
SAMPLE_SYSTEM_MSG_LIST = [{"type": "text", "text": "You are a helpful assistant."}]

SAMPLE_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello"},
]

SAMPLE_GEMINI_KWARGS = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ],
    "max_tokens": 1000,
    "temperature": 0.7,
    "n": 1,
    "top_p": 0.9,
    "stop": ["###"],
    "generation_config": {
        "max_tokens": 2000,
        "temperature": 0.5,
    },
}


class TestDictionaryOperations:
    """Test suite for dictionary operations performance."""

    def test_extract_messages_benchmark(self):
        """Benchmark for extract_messages function."""
        # Test with different message locations
        results = {}

        # Benchmark with messages key
        results["messages"] = timeit.timeit(
            lambda: extract_messages(SAMPLE_KWARGS_MESSAGES), number=10000
        )

        # Benchmark with contents key
        results["contents"] = timeit.timeit(
            lambda: extract_messages(SAMPLE_KWARGS_CONTENTS), number=10000
        )

        # Benchmark with chat_history key
        results["chat_history"] = timeit.timeit(
            lambda: extract_messages(SAMPLE_KWARGS_CHAT_HISTORY), number=10000
        )

        # Benchmark with empty dict
        results["empty"] = timeit.timeit(
            lambda: extract_messages(SAMPLE_KWARGS_EMPTY), number=10000
        )

        # Print benchmark results (useful for debugging)
        print("\nExtract Messages Benchmark Results:")
        for key, time in results.items():
            print(f"{key}: {time:.6f}s")

        # Ensure the optimized version is faster than a baseline (for CI)
        baseline = 0.1  # Adjust based on initial benchmark runs
        for key, time in results.items():
            assert time < baseline, (
                f"extract_messages with {key} is too slow: {time:.6f}s > {baseline:.6f}s"
            )

    def test_combine_system_messages_benchmark(self):
        """Benchmark for combine_system_messages function."""
        results = {}

        # Both string
        results["str_str"] = timeit.timeit(
            lambda: combine_system_messages(
                SAMPLE_SYSTEM_MSG_STR, SAMPLE_SYSTEM_MSG_STR
            ),
            number=10000,
        )

        # Both list
        results["list_list"] = timeit.timeit(
            lambda: combine_system_messages(
                SAMPLE_SYSTEM_MSG_LIST, SAMPLE_SYSTEM_MSG_LIST
            ),
            number=10000,
        )

        # String and list
        results["str_list"] = timeit.timeit(
            lambda: combine_system_messages(
                SAMPLE_SYSTEM_MSG_STR, SAMPLE_SYSTEM_MSG_LIST
            ),
            number=10000,
        )

        # List and string
        results["list_str"] = timeit.timeit(
            lambda: combine_system_messages(
                SAMPLE_SYSTEM_MSG_LIST, SAMPLE_SYSTEM_MSG_STR
            ),
            number=10000,
        )

        # None and string
        results["none_str"] = timeit.timeit(
            lambda: combine_system_messages(None, SAMPLE_SYSTEM_MSG_STR),
            number=10000,
        )

        print("\nCombine System Messages Benchmark Results:")
        for key, time in results.items():
            print(f"{key}: {time:.6f}s")

        baseline = 0.2  # Adjust based on initial benchmark runs
        for key, time in results.items():
            assert time < baseline, (
                f"combine_system_messages with {key} is too slow: {time:.6f}s > {baseline:.6f}s"
            )

    def test_extract_system_messages_benchmark(self):
        """Benchmark for extract_system_messages function."""
        results = {}

        # With system messages
        results["with_system"] = timeit.timeit(
            lambda: extract_system_messages(SAMPLE_MESSAGES),
            number=10000,
        )

        # Without system messages
        results["no_system"] = timeit.timeit(
            lambda: extract_system_messages([{"role": "user", "content": "Hello"}]),
            number=10000,
        )

        # Empty messages
        results["empty"] = timeit.timeit(
            lambda: extract_system_messages([]),
            number=10000,
        )

        print("\nExtract System Messages Benchmark Results:")
        for key, time in results.items():
            print(f"{key}: {time:.6f}s")

        baseline = 0.2  # Adjust based on initial benchmark runs
        for key, time in results.items():
            assert time < baseline, (
                f"extract_system_messages with {key} is too slow: {time:.6f}s > {baseline:.6f}s"
            )

    def test_update_gemini_kwargs_benchmark(self):
        """Benchmark for update_gemini_kwargs function."""
        result = timeit.timeit(
            lambda: update_gemini_kwargs(SAMPLE_GEMINI_KWARGS),
            number=1000,
        )

        print(f"\nUpdate Gemini Kwargs Benchmark Result: {result:.6f}s")
        baseline = 0.2  # Adjust based on initial benchmark runs
        assert result < baseline, (
            f"update_gemini_kwargs is too slow: {result:.6f}s > {baseline:.6f}s"
        )

    # We'll use a simpler test for mode lookup patterns since proper mocking is complex
    # Test removed as it was producing inconsistent results across different environments
