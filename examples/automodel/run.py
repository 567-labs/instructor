#!/usr/bin/env python
"""
Example demonstrating the unified provider interface with string-based initialization.
Creates clients for multiple providers with both sync and async interfaces.
"""

import os
import asyncio
from typing import Any
import instructor
from pydantic import BaseModel, Field


class UserInfo(BaseModel):
    """Simple model to extract user information from text."""

    name: str = Field(description="The user's full name")
    age: int = Field(description="The user's age in years")
    occupation: str = Field(description="The user's job or profession")


async def test_async_client(
    client_name: str, client: instructor.AsyncInstructor
) -> dict[str, Any]:
    """Test an async client and return the results."""
    print(f"Testing async client: {client_name}")
    try:
        result = await client.chat.completions.create(
            response_model=UserInfo,
            messages=[
                {
                    "role": "user",
                    "content": "John Smith is a 35-year-old software engineer.",
                }
            ],
        )
        print(f"✅ Async {client_name} result: {result.model_dump()}")
        return {"provider": client_name, "success": True, "result": result.model_dump()}
    except Exception as e:
        print(f"❌ Async {client_name} error: {str(e)}")
        return {"provider": client_name, "success": False, "error": str(e)}


def test_sync_client(client_name: str, client: instructor.Instructor) -> dict[str, Any]:
    """Test a sync client and return the results."""
    print(f"Testing sync client: {client_name}")
    try:
        result = client.chat.completions.create(
            response_model=UserInfo,
            messages=[
                {"role": "user", "content": "Jane Doe is a 28-year-old data scientist."}
            ],
        )
        print(f"✅ Sync {client_name} result: {result.model_dump()}")
        return {"provider": client_name, "success": True, "result": result.model_dump()}
    except Exception as e:
        print(f"❌ Sync {client_name} error: {str(e)}")
        return {"provider": client_name, "success": False, "error": str(e)}


async def main():
    """Create and test multiple clients using the unified provider interface."""
    # Collect the test results
    sync_results = []
    async_results = []

    # Test OpenAI clients
    if os.environ.get("OPENAI_API_KEY"):
        # Sync client
        openai_client = instructor.from_provider("openai/gpt-3.5-turbo")
        sync_results.append(test_sync_client("OpenAI", openai_client))

        # Async client
        openai_async = instructor.from_provider(
            "openai/gpt-3.5-turbo", async_client=True
        )
        async_results.append(
            asyncio.create_task(test_async_client("OpenAI", openai_async))
        )
    else:
        print("⚠️ OPENAI_API_KEY not set, skipping OpenAI tests")

    # Test Anthropic clients
    if os.environ.get("ANTHROPIC_API_KEY"):
        # Sync client
        anthropic_client = instructor.from_provider(
            model="anthropic/claude-3-haiku-20240307", max_tokens=400
        )
        sync_results.append(test_sync_client("Anthropic", anthropic_client))

        # Async client
        anthropic_async = instructor.from_provider(
            model="anthropic/claude-3-haiku-20240307", async_client=True, max_tokens=400
        )
        async_results.append(
            asyncio.create_task(test_async_client("Anthropic", anthropic_async))
        )
    else:
        print("⚠️ ANTHROPIC_API_KEY not set, skipping Anthropic tests")

    # Test Cohere clients
    if os.environ.get("COHERE_API_KEY"):
        # Sync client
        cohere_client = instructor.from_provider("cohere/command")
        sync_results.append(test_sync_client("Cohere", cohere_client))

        # Async client
        cohere_async = instructor.from_provider("cohere/command", async_client=True)
        async_results.append(
            asyncio.create_task(test_async_client("Cohere", cohere_async))
        )
    else:
        print("⚠️ COHERE_API_KEY not set, skipping Cohere tests")

    # Test Mistral clients
    if os.environ.get("MISTRAL_API_KEY"):
        # Sync client
        mistral_client = instructor.from_provider("mistral/mistral-small")
        sync_results.append(test_sync_client("Mistral", mistral_client))

        # Async client
        mistral_async = instructor.from_provider(
            "mistral/mistral-small", async_client=True
        )
        async_results.append(
            asyncio.create_task(test_async_client("Mistral", mistral_async))
        )
    else:
        print("⚠️ MISTRAL_API_KEY not set, skipping Mistral tests")

    # Process async results
    if async_results:
        completed_tasks = await asyncio.gather(*async_results)
        async_results = completed_tasks

    # Print summary
    print("\n----- Test Results Summary -----")

    print("\nSync Clients:")
    for result in sync_results:
        if result.get("success", False):
            print(f"✅ {result['provider']} - Success")
        else:
            print(
                f"❌ {result['provider']} - Failed: {result.get('error', 'Unknown error')}"
            )

    print("\nAsync Clients:")
    for result in async_results:
        if result.get("success", False):
            print(f"✅ {result['provider']} - Success")
        else:
            print(
                f"❌ {result['provider']} - Failed: {result.get('error', 'Unknown error')}"
            )


if __name__ == "__main__":
    asyncio.run(main())
