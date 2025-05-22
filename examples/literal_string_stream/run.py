#!/usr/bin/env python3
"""
Example demonstrating how Literal types work with streaming in Instructor.

This example shows:
1. Non-streaming mode works fine with Literal types
2. Direct streaming without Partial doesn't work (expected behavior)
3. Streaming with Partial now works correctly with Literal types
"""

import instructor
from openai import OpenAI
from pydantic import BaseModel
from typing import Literal


class User(BaseModel):
    name: Literal["Bob", "Alice", "John"]
    role: Literal["admin", "user", "guest"]
    age: int


def test_non_streaming():
    """Test with non-streaming mode - works correctly"""
    print("Testing non-streaming mode...")

    client = instructor.from_openai(OpenAI())

    try:
        user = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Create a user named Alice who is 25 years old and has admin role",
                }
            ],
            response_model=User,
        )
        print(f"✓ Non-streaming success: {user}")
    except Exception as e:
        print(f"✗ Non-streaming failed: {type(e).__name__}: {e}")


def test_streaming_without_partial():
    """Test with streaming mode without Partial - expected to fail"""
    print("\nTesting streaming without Partial (expected to fail)...")

    client = instructor.from_openai(OpenAI())

    try:
        # This will fail because direct streaming requires using Partial
        user = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Create a user named Bob who is 30 years old and has user role",
                }
            ],
            response_model=User,
            stream=True,
        )
        print(f"✓ Streaming success: {user}")
    except Exception as e:
        print(
            f"✗ Expected failure - {type(e).__name__}: Direct streaming requires Partial"
        )


def test_streaming_with_partial():
    """Test with streaming mode using Partial - now works with Literal types!"""
    print("\nTesting streaming with Partial (fixed to work with Literals)...")

    from instructor import Partial

    client = instructor.from_openai(OpenAI())

    try:
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Create a user named John who is 35 years old and has guest role",
                }
            ],
            response_model=Partial[User],
            stream=True,
        )

        # Track the progression of the stream
        updates = []
        for partial_user in stream:
            update = {
                "name": partial_user.name,
                "role": partial_user.role,
                "age": partial_user.age,
            }
            # Only print if something changed
            if not updates or updates[-1] != update:
                print(
                    f"  Partial update: name={update['name']!r}, role={update['role']!r}, age={update['age']}"
                )
                updates.append(update)

        print("✓ Streaming with Partial completed successfully!")
        print(f"  Final result: {updates[-1]}")
    except Exception as e:
        print(f"✗ Streaming with Partial failed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()


def test_complex_literal_streaming():
    """Test more complex model with multiple Literal fields"""
    print("\nTesting complex model with multiple Literal fields...")

    from instructor import Partial

    class Product(BaseModel):
        name: str
        category: Literal["electronics", "clothing", "food", "books"]
        status: Literal["available", "out_of_stock", "discontinued"]
        condition: Literal["new", "used", "refurbished"]
        priority: Literal["low", "medium", "high"]
        price: float

    client = instructor.from_openai(OpenAI())

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Create a product: iPhone 15 in electronics category, available, new condition, high priority, $999",
            }
        ],
        response_model=Partial[Product],
        stream=True,
    )

    final_product = None
    for partial_product in stream:
        final_product = partial_product

    print(f"✓ Complex streaming completed: {final_product}")


if __name__ == "__main__":
    print("Literal Types with Streaming in Instructor")
    print("=" * 50)
    print("This example demonstrates the fix for Literal types in streaming mode.")
    print("Previously, Literal fields would cause validation errors during streaming")
    print("because partial values (e.g., 'Al' while typing 'Alice') are not valid.")
    print("The fix allows any string during streaming, then validates the final value.")
    print("=" * 50)

    test_non_streaming()
    test_streaming_without_partial()
    test_streaming_with_partial()
    test_complex_literal_streaming()
