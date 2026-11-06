"""
This example demonstrates how to use hooks in Instructor for monitoring,
logging, and debugging your LLM interactions.

Hooks allow you to attach handlers to events that occur during the completion
and parsing process. This can be useful for:
- Logging API requests and responses
- Debugging parsing errors
- Collecting statistics about API usage
- Adding custom error handling
"""

import instructor
import openai
import pydantic


class User(pydantic.BaseModel):
    """A simple user model with validation."""

    name: str
    age: int

    @pydantic.field_validator("age")
    def validate_age(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Age must be non-negative")
        return v


class CompletionStats:
    """A simple class to collect statistics about completions."""

    def __init__(self):
        self.total_completions = 0
        self.errors = 0
        self.successful = 0
        self.tokens_used = 0

    def report(self):
        """Print a report of the statistics."""
        print("\n--- Completion Statistics ---")
        print(f"Total completions: {self.total_completions}")
        print(f"Successful: {self.successful}")
        print(f"Errors: {self.errors}")
        print(f"Total tokens used: {self.tokens_used}")


def main():
    # Initialize the OpenAI client with Instructor
    client = instructor.from_openai(openai.OpenAI())

    # Create a statistics collector
    stats = CompletionStats()

    # Define hook handlers
    def log_completion_kwargs(_, **kwargs):
        """Handler for completion:kwargs hook."""
        stats.total_completions += 1
        print(
            f"\n🔍 Sending completion request using model: {kwargs.get('model', 'unknown')}"
        )
        if "messages" in kwargs:
            for msg in kwargs["messages"]:
                if msg.get("role") == "user":
                    print(f"📝 User prompt: {msg.get('content')}")

    def log_completion_response(response):
        """Handler for completion:response hook."""
        stats.successful += 1

        # Extract token usage if available
        if hasattr(response, "usage") and response.usage:
            token_usage = response.usage.total_tokens
            stats.tokens_used += token_usage
            print(f"📊 Token usage: {token_usage}")

        print(f"✅ Received completion response")

    def log_completion_error(error):
        """Handler for completion:error hook."""
        stats.errors += 1
        print(f"❌ Completion error: {type(error).__name__}: {str(error)}")

    def log_parse_error(error):
        """Handler for parse:error hook."""
        stats.errors += 1
        print(f"⚠️ Parse error: {type(error).__name__}: {str(error)}")

    # Register the hooks
    client.on("completion:kwargs", log_completion_kwargs)
    client.on("completion:response", log_completion_response)
    client.on("completion:error", log_completion_error)
    client.on(
        "completion:last_attempt", lambda _: print(f"🔄 Last retry attempt failed")
    )
    client.on("parse:error", log_parse_error)

    # Example 1: Successful extraction
    try:
        print("\n--- Example 1: Successful Extraction ---")
        user = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Extract: John is 30 years old."}],
            response_model=User,
        )
        print(f"Result: {user}")
    except Exception as e:
        print(f"Main exception: {e}")

    # Example 2: Parse error (validation fails)
    try:
        print("\n--- Example 2: Parse Error (Age Validation) ---")
        user = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Extract: Alice is -5 years old."}],
            response_model=User,
        )
        print(f"Result: {user}")
    except Exception as e:
        print(f"Main exception: {e}")

    # Example 3: Multiple hooks for the same event
    print("\n--- Example 3: Multiple Hooks ---")

    # Add another hook for completion:kwargs that counts message tokens
    def count_input_tokens(_, **kwargs):
        """Handler for counting approximate tokens in input messages."""
        if "messages" in kwargs:
            total_chars = sum(len(msg.get("content", "")) for msg in kwargs["messages"])
            # Rough approximation of tokens (not accurate)
            approx_tokens = total_chars / 4
            print(f"📏 Approximate input tokens: {approx_tokens:.0f}")

    # Register the additional hook
    client.on("completion:kwargs", count_input_tokens)

    try:
        user = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Extract: Bob is 25 years old."}],
            response_model=User,
        )
        print(f"Result: {user}")
    except Exception as e:
        print(f"Main exception: {e}")

    # Print the final statistics
    stats.report()

    # Clean up hooks
    print("\n--- Cleaning Up Hooks ---")
    client.clear()
    print("All hooks cleared")


if __name__ == "__main__":
    main()

"""

--- Example 1: Successful Extraction ---

🔍 Sending completion request using model: gpt-3.5-turbo
📝 User prompt: Extract: John is 30 years old.
📊 Token usage: 82
✅ Received completion response
Result: name='John' age=30

--- Example 2: Parse Error (Age Validation) ---

🔍 Sending completion request using model: gpt-3.5-turbo
📝 User prompt: Extract: Alice is -5 years old.
📊 Token usage: 82
✅ Received completion response
⚠️ Parse error: ValidationError: 1 validation error for User
age
  Value error, Age must be non-negative [type=value_error, input_value=-5, input_type=int]
    For further information visit https://errors.pydantic.dev/2.9/v/value_error

🔍 Sending completion request using model: gpt-3.5-turbo
📝 User prompt: Extract: Alice is -5 years old.
📊 Token usage: 170
✅ Received completion response
Result: name='Alice' age=5

--- Example 3: Multiple Hooks ---

🔍 Sending completion request using model: gpt-3.5-turbo
📝 User prompt: Extract: Bob is 25 years old.
📏 Approximate input tokens: 7
📊 Token usage: 82
✅ Received completion response
Result: name='Bob' age=25

--- Completion Statistics ---
Total completions: 4
Successful: 4
Errors: 1
Total tokens used: 416

--- Cleaning Up Hooks ---
All hooks cleared
"""
