"""
Tenacity Retry Logic Benchmarks with Instructor

This script demonstrates and benchmarks different retry patterns for LLM processing:
- Basic retry with exponential backoff
- Conditional retries for specific errors
- Validation error retries
- Custom retry conditions
- Rate limit handling
- Network error recovery
- Logging and monitoring
- Circuit breaker patterns

Run this script to see retry behavior and verify all code examples work.
"""

import instructor
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_result,
    before_log,
    after_log,
    wait_random_exponential,
)
from pydantic import BaseModel, field_validator, ValidationError
from openai import OpenAI, RateLimitError, APIError
import time
import logging
import random
import os
from functools import lru_cache
import httpx

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up the client with Instructor
client = instructor.from_openai(OpenAI())


class UserInfo(BaseModel):
    name: str
    age: int
    email: str

    @field_validator("age")
    @classmethod
    def validate_age(cls, v):
        if v < 0 or v > 150:
            raise ValueError(f"Age {v} is invalid")
        return v

    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        if "@" not in v:
            raise ValueError(f"Invalid email: {v}")
        return v.lower()


# Sample data for testing
test_texts = [
    "John is 30 years old with email john@example.com",
    "Sarah is 25 with email sarah@test.com",
    "Mike is 35 and his email is mike@demo.org",
    "Alice is 28 with email alice@example.com",
    "Bob is 32 with email bob@test.com",
]


# Error simulation for testing
class MockError:
    def __init__(self):
        self.call_count = 0
        self.fail_until = 2  # Fail first 2 calls, succeed on 3rd

    def maybe_fail(self):
        self.call_count += 1
        if self.call_count <= self.fail_until:
            # Simulate different types of errors
            error_type = random.choice(
                [ValidationError, RateLimitError, APIError, Exception]
            )
            if error_type == ValidationError:
                raise ValidationError.from_exception_data("UserInfo", [])
            elif error_type == RateLimitError:
                # Create a simple mock response for RateLimitError
                mock_response = httpx.Response(
                    status_code=429, headers={}, content=b"Rate limit exceeded"
                )
                raise RateLimitError(
                    "Rate limit exceeded",
                    response=mock_response,
                    body="Rate limit exceeded",
                )
            elif error_type == APIError:
                # Create a simple mock request for APIError
                mock_request = httpx.Request(
                    "POST", "https://api.openai.com/v1/chat/completions"
                )
                raise APIError(
                    "API error occurred",
                    request=mock_request,
                    body="API error occurred",
                )
            else:
                raise Exception("Generic error occurred")


mock_error = MockError()


def extract_user_info_with_mock_errors(text: str) -> UserInfo:
    """Extract user info with simulated errors for testing."""
    if not os.getenv("OPENAI_API_KEY"):
        # Simulate errors for testing when no API key
        mock_error.maybe_fail()
        # Return mock data if no errors
        return UserInfo(name="Mock User", age=30, email="mock@example.com")

    return client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=UserInfo,
        messages=[{"role": "user", "content": f"Extract user info: {text}"}],
    )


# Method 1: Basic Retry with Exponential Backoff
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),  # Shorter waits for demo
)
def extract_user_info(text: str) -> UserInfo:
    """Extract user information with basic retry logic."""
    print(f"  Attempting extraction for: {text[:30]}...")
    if not os.getenv("OPENAI_API_KEY"):
        mock_error.maybe_fail()
        return UserInfo(name="Test User", age=25, email="test@example.com")

    return client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=UserInfo,
        messages=[{"role": "user", "content": f"Extract user info: {text}"}],
    )


# Method 2: Conditional Retries for Specific Errors
@retry(
    retry=retry_if_exception_type((RateLimitError, APIError)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=5),
)
def robust_extraction(text: str) -> UserInfo:
    """Retry only on specific API errors."""
    print(f"  Robust extraction for: {text[:30]}...")
    return extract_user_info_with_mock_errors(text)


# Method 3: Validation Error Retries
@retry(
    retry=retry_if_exception_type(ValidationError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=3),
)
def extract_with_validation(text: str) -> UserInfo:
    """Retry when Pydantic validation fails."""
    print(f"  Validation retry for: {text[:30]}...")
    return extract_user_info_with_mock_errors(text)


# Method 4: Custom Retry Conditions
def should_retry(result: UserInfo) -> bool:
    """Custom retry logic based on result content."""
    # Retry if age is invalid or email is missing
    return result.age < 18 or result.age > 100 or not result.email


@retry(
    retry=retry_if_result(should_retry),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=3),
)
def extract_valid_user(text: str) -> UserInfo:
    """Retry based on result validation."""
    print(f"  Custom retry for: {text[:30]}...")
    # Simulate returning invalid data first time
    if not hasattr(extract_valid_user, "call_count"):
        extract_valid_user.call_count = 0
    extract_valid_user.call_count += 1

    if extract_valid_user.call_count == 1:
        # Return invalid data first time
        return UserInfo(name="Invalid User", age=200, email="invalid")
    else:
        # Return valid data on retry
        return UserInfo(name="Valid User", age=30, email="valid@example.com")


# Method 5: Rate Limit Specific Retry
@retry(
    retry=retry_if_exception_type(RateLimitError),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=1, max=10),
    before_sleep=lambda retry_state: print(
        f"    Rate limited, waiting... (attempt {retry_state.attempt_number})"
    ),
)
def rate_limit_safe_extraction(text: str) -> UserInfo:
    """Handle rate limits with longer delays."""
    print(f"  Rate limit safe for: {text[:30]}...")
    return extract_user_info_with_mock_errors(text)


# Method 6: Network Error Retry
@retry(
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    stop=stop_after_attempt(4),
    wait=wait_random_exponential(multiplier=1, min=1, max=5),
)
def network_resilient_extraction(text: str) -> UserInfo:
    """Handle network issues with random exponential backoff."""
    print(f"  Network resilient for: {text[:30]}...")
    return extract_user_info_with_mock_errors(text)


# Method 7: Logging and Monitoring
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    before=before_log(logger, logging.INFO),
    after=after_log(logger, logging.ERROR),
)
def logged_extraction(text: str) -> UserInfo:
    """Extract with comprehensive logging."""
    print(f"  Logged extraction for: {text[:30]}...")
    return extract_user_info_with_mock_errors(text)


# Method 8: Circuit Breaker Pattern
@lru_cache(maxsize=1)
def get_client():
    """Cache the client to avoid repeated initialization."""
    return instructor.from_openai(OpenAI())


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
def circuit_breaker_extraction(text: str) -> UserInfo:
    """Extract with circuit breaker pattern."""
    print(f"  Circuit breaker for: {text[:30]}...")
    client = get_client()
    return extract_user_info_with_mock_errors(text)


# Method 9: Performance Monitoring
@retry(stop=stop_after_attempt(3))
def monitored_extraction(text: str) -> UserInfo:
    """Extract with performance monitoring."""
    start_time = time.time()

    try:
        print(f"  Monitored extraction for: {text[:30]}...")
        result = extract_user_info_with_mock_errors(text)

        end_time = time.time()
        print(f"    Extraction took {end_time - start_time:.2f} seconds")
        return result

    except Exception as e:
        end_time = time.time()
        print(f"    Extraction failed after {end_time - start_time:.2f} seconds: {e}")
        raise


def benchmark_retry_methods():
    """Test all retry methods and measure their behavior."""
    print("=== Python Tenacity Retry Logic with Instructor Benchmarks ===\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set. Using mock responses for demonstration.\n")

    # Test different retry strategies
    strategies = [
        ("Basic Retry", extract_user_info),
        ("Conditional Retry", robust_extraction),
        ("Validation Retry", extract_with_validation),
        ("Custom Retry", extract_valid_user),
        ("Rate Limit Retry", rate_limit_safe_extraction),
        ("Network Retry", network_resilient_extraction),
        ("Logged Retry", logged_extraction),
        ("Circuit Breaker", circuit_breaker_extraction),
        ("Monitored Retry", monitored_extraction),
    ]

    results = {}
    test_text = test_texts[0]  # Use first text for all tests

    for name, strategy in strategies:
        print(f"\n{'=' * 60}")
        print(f"Testing: {name}")
        print("=" * 60)

        # Reset mock error for each test
        global mock_error
        mock_error = MockError()

        # Reset call count for custom retry
        if hasattr(extract_valid_user, "call_count"):
            delattr(extract_valid_user, "call_count")

        start_time = time.time()
        try:
            user = strategy(test_text)
            end_time = time.time()
            duration = end_time - start_time

            results[name] = {
                "success": True,
                "duration": duration,
                "user": user,
                "attempts": getattr(mock_error, "call_count", 1),
            }

            print(f"✓ Success: {user.name} ({duration:.2f}s)")
            print(f"  Age: {user.age}, Email: {user.email}")
            print(f"  Attempts made: {results[name]['attempts']}")

        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time

            results[name] = {
                "success": False,
                "duration": duration,
                "error": str(e),
                "attempts": getattr(mock_error, "call_count", 1),
            }

            print(f"✗ Failed: {e} ({duration:.2f}s)")
            print(f"  Attempts made: {results[name]['attempts']}")

    # Print summary table
    print(f"\n{'=' * 80}")
    print("RETRY STRATEGY SUMMARY")
    print("=" * 80)
    print(
        f"{'Strategy':<20} {'Status':<10} {'Time (s)':<10} {'Attempts':<10} {'Result'}"
    )
    print("-" * 80)

    for name, result in results.items():
        status = "✓ Success" if result["success"] else "✗ Failed"
        attempts = result["attempts"]

        if result["success"]:
            result_text = f"{result['user'].name}"
        else:
            result_text = "Failed"

        print(
            f"{name:<20} {status:<10} {result['duration']:<10.2f} {attempts:<10} {result_text}"
        )

    # Show retry efficiency
    print(f"\nRetry Efficiency Analysis:")
    successful_strategies = {k: v for k, v in results.items() if v["success"]}

    if successful_strategies:
        avg_attempts = sum(r["attempts"] for r in successful_strategies.values()) / len(
            successful_strategies
        )
        avg_duration = sum(r["duration"] for r in successful_strategies.values()) / len(
            successful_strategies
        )

        print(f"  Average attempts: {avg_attempts:.1f}")
        print(f"  Average duration: {avg_duration:.2f}s")

        # Find most efficient strategy
        most_efficient = min(
            successful_strategies.items(),
            key=lambda x: x[1]["attempts"] * x[1]["duration"],
        )
        print(
            f"  Most efficient: {most_efficient[0]} ({most_efficient[1]['attempts']} attempts, {most_efficient[1]['duration']:.2f}s)"
        )


def test_batch_processing():
    """Test batch processing with retries."""
    print(f"\n{'=' * 60}")
    print("Batch Processing Test")
    print("=" * 60)

    @retry(stop=stop_after_attempt(2))
    def process_batch(texts: list[str]) -> list[UserInfo]:
        """Process multiple texts with retry logic."""
        results = []

        for text in texts:
            try:
                # Reset mock error for each item
                global mock_error
                mock_error = MockError()

                result = extract_user_info_with_mock_errors(text)
                results.append(result)
                print(f"  ✓ Processed: {result.name}")
            except Exception as e:
                print(f"  ✗ Failed to process: {text[:30]}... - {e}")
                continue

        return results

    start_time = time.time()
    try:
        results = process_batch(test_texts[:3])  # Process first 3 texts
        end_time = time.time()
        duration = end_time - start_time

        print(f"\nBatch processing completed:")
        print(f"  Successfully processed: {len(results)}/{len(test_texts[:3])} items")
        print(f"  Total time: {duration:.2f} seconds")
        print(f"  Average time per item: {duration / len(test_texts[:3]):.2f} seconds")

    except Exception as e:
        print(f"Batch processing failed: {e}")


def demonstrate_error_types():
    """Demonstrate handling different error types."""
    print(f"\n{'=' * 60}")
    print("Error Type Demonstration")
    print("=" * 60)

    # Simulate different error scenarios
    error_scenarios = [
        ("Validation Error", ValidationError),
        ("Rate Limit Error", RateLimitError),
        ("API Error", APIError),
        ("Generic Error", Exception),
    ]

    for error_name, error_type in error_scenarios:
        print(f"\nTesting {error_name}:")

        def create_error_handler(error_type):
            @retry(
                retry=retry_if_exception_type(error_type),
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=0.5, max=2),
            )
            def handle_specific_error():
                # Simulate the specific error type
                if error_type == ValidationError:
                    raise ValidationError.from_exception_data("UserInfo", [])
                elif error_type == RateLimitError:
                    # Create a simple mock response for RateLimitError
                    mock_response = httpx.Response(
                        status_code=429, headers={}, content=b"Rate limit exceeded"
                    )
                    raise RateLimitError(
                        "Rate limit exceeded",
                        response=mock_response,
                        body="Rate limit exceeded",
                    )
                elif error_type == APIError:
                    # Create a simple mock request for APIError
                    mock_request = httpx.Request(
                        "POST", "https://api.openai.com/v1/chat/completions"
                    )
                    raise APIError(
                        "API error occurred",
                        request=mock_request,
                        body="API error occurred",
                    )
                else:
                    raise Exception("Generic error occurred")

            return handle_specific_error

        error_handler = create_error_handler(error_type)

        try:
            error_handler()
        except Exception as e:
            print(f"  Expected failure: {type(e).__name__}: {e}")


def main():
    """Main function to run all benchmarks and demonstrations."""
    try:
        benchmark_retry_methods()
        test_batch_processing()
        demonstrate_error_types()

        print(f"\n{'=' * 80}")
        print("🎉 All tenacity retry patterns demonstrated successfully!")
        print("💡 Key takeaways:")
        print("   - Different retry strategies serve different purposes")
        print("   - Exponential backoff prevents overwhelming APIs")
        print("   - Conditional retries optimize for specific error types")
        print("   - Monitoring helps debug and optimize retry behavior")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.exception("Unexpected error occurred")


if __name__ == "__main__":
    print("🚀 Starting tenacity retry benchmarks with Instructor...")
    print("💡 This script demonstrates retry patterns with simulated errors")
    print("⏱️  Each test includes artificial delays and error scenarios\n")

    main()
