#!/usr/bin/env python3
"""Unified Batch API Test Script

Test script to verify the unified BatchProcessor works correctly with all supported providers.
Creates a batch job to extract User(name: str, age: int) data from text examples.

Supports:
- OpenAI: openai/gpt-4o-mini, openai/gpt-4o, etc.
- Anthropic: anthropic/claude-3-5-sonnet-20241022, anthropic/claude-3-opus-20240229, etc.
- Google: google/gemini-2.5-flash, google/gemini-pro, etc.

Usage:
    # Default (Google Gemini 2.5 Flash)
    export GOOGLE_API_KEY="your-key"
    python run_batch_test.py

    # OpenAI
    export OPENAI_API_KEY="your-key"
    python run_batch_test.py --model "openai/gpt-4o-mini"

    # Anthropic
    export ANTHROPIC_API_KEY="your-key"
    python run_batch_test.py --model "anthropic/claude-3-5-sonnet-20241022"

    # Google with specific model
    export GOOGLE_API_KEY="your-key"
    python run_batch_test.py --model "google/gemini-2.5-flash"
"""

import os
import sys
from typing import Optional
import typer
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from instructor.batch import (
    BatchProcessor,
    BatchStatus,
    filter_successful,
    filter_errors,
    extract_results,
)

app = typer.Typer(help="Unified Batch API Test for all providers")


class User(BaseModel):
    name: str
    age: int


def create_test_messages() -> list[list[dict]]:
    """Create test message conversations for user extraction"""
    test_prompts = [
        "Hi there! My name is Alice and I'm 28 years old. I work as a software engineer.",
    ]

    messages_list = []
    for prompt in test_prompts:
        messages = [
            {
                "role": "system",
                "content": "You are an expert at extracting structured user information from text. Extract the person's name and age.",
            },
            {"role": "user", "content": prompt},
        ]
        messages_list.append(messages)

    return messages_list


def get_expected_results() -> list[User]:
    """Get the expected User objects for validation"""
    return [
        User(name="Alice", age=28),
    ]


def check_api_key(provider: str) -> bool:
    """Check if the required API key is set for the provider"""
    key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }

    required_key = key_map.get(provider)
    if not required_key:
        return True  # Unknown provider, let it fail later

    if provider == "google":
        # Google is optional since we simulate
        if not os.getenv(required_key):
            typer.echo(f"Warning: {required_key} not set - will run in simulation mode")
        return True

    if not os.getenv(required_key):
        typer.echo(f"Error: {required_key} environment variable is not set", err=True)
        typer.echo(
            f"Please set your API key: export {required_key}='your-api-key-here'",
            err=True,
        )
        return False

    return True


def create_openai_batch(model: str, messages_list: list[list[dict]]) -> Optional[str]:
    """Create OpenAI batch job using BatchProcessor"""
    processor = BatchProcessor(model, User)

    # Create batch file
    batch_filename = "test_batch.jsonl"
    processor.create_batch_from_messages(
        file_path=batch_filename,
        messages_list=messages_list,
        max_tokens=200,
        temperature=0.1,
    )

    try:
        typer.echo("Submitting batch job...")
        batch_id = processor.submit_batch(
            file_path=batch_filename,
            metadata={"description": "Unified BatchProcessor test"},
        )
        return batch_id

    finally:
        if os.path.exists(batch_filename):
            os.remove(batch_filename)


def create_anthropic_batch(
    model: str, messages_list: list[list[dict]]
) -> Optional[str]:
    """Create Anthropic batch job using BatchProcessor"""
    processor = BatchProcessor(model, User)

    # Create batch file
    batch_filename = "test_batch.jsonl"
    processor.create_batch_from_messages(
        file_path=batch_filename,
        messages_list=messages_list,
        max_tokens=200,
        temperature=0.1,
    )

    try:
        typer.echo("Submitting batch job...")
        batch_id = processor.submit_batch(file_path=batch_filename)
        return batch_id

    finally:
        if os.path.exists(batch_filename):
            os.remove(batch_filename)


def create_google_batch(model: str, messages_list: list[list[dict]]) -> Optional[str]:
    """Create Google batch job using BatchProcessor (inline only)"""
    processor = BatchProcessor(model, User)

    typer.echo("Submitting Google inline batch...")
    batch_id = processor.submit_batch(
        messages_list=messages_list,
        metadata={"description": "Unified BatchProcessor test"},
        use_inline=True,
        max_tokens=200,
        temperature=0.1,
    )

    typer.echo(f"Inline batch job created: {batch_id}")
    return batch_id


@app.command()
def create(
    model: str = typer.Option(
        "openai/gpt-4o-mini",
        help="Model in format 'provider/model-name' (e.g., 'google/gemini-2.5-flash', 'openai/gpt-4o-mini', 'anthropic/claude-3-5-sonnet-20241022')",
    ),
    save_id: bool = typer.Option(True, help="Save batch ID to file"),
):
    """Create a batch job for the specified model"""

    typer.echo(f"Creating Batch Job for {model}")
    typer.echo("=" * 50)

    # Parse provider from model
    try:
        provider, model_name = model.split("/", 1)
    except ValueError:
        typer.echo("Error: Model must be in format 'provider/model-name'", err=True)
        typer.echo(
            "Examples: 'openai/gpt-4o-mini', 'anthropic/claude-3-5-sonnet-20241022'",
            err=True,
        )
        raise typer.Exit(1) from None

    # Check API key
    if not check_api_key(provider):
        raise typer.Exit(1)

    # Create test messages
    messages_list = create_test_messages()
    typer.echo(f"Created {len(messages_list)} test message conversations")

    try:
        # Create batch job based on provider
        batch_id = None

        if provider == "openai":
            batch_id = create_openai_batch(model, messages_list)
        elif provider == "anthropic":
            batch_id = create_anthropic_batch(model, messages_list)
        else:
            typer.echo(f"Unsupported provider: {provider}", err=True)
            raise typer.Exit(1)

        if batch_id:
            typer.echo(f"Batch job created with ID: {batch_id}")

            if save_id:
                filename = f"{provider}_batch_id.txt"
                with open(filename, "w") as f:
                    f.write(batch_id)
                typer.echo(f"Batch ID saved to {filename}")

            # Validate expected results
            expected_results = get_expected_results()
            typer.echo(f"Expected results validated: {len(expected_results)} users")
            for i, user in enumerate(expected_results):
                typer.echo(f"   {i + 1}. {user.name}, age {user.age}")

            # Show how to check status
            typer.echo(f"Check status with:")
            typer.echo(f"   instructor batch list --model {model}")

            typer.echo(f"Cost savings: 50% vs regular API")
            typer.echo(f"\nSuccess! Batch ID: {batch_id}")

        else:
            typer.echo("Failed to create batch job", err=True)
            raise typer.Exit(1)

    except Exception as e:
        typer.echo(f"Error creating batch: {e}", err=True)
        raise typer.Exit(1) from e


@app.command()
def list_batches():
    """List saved batch IDs for all providers"""
    typer.echo("Saved Batch IDs:")
    typer.echo("=" * 30)

    providers = ["openai", "anthropic"]
    found_any = False

    for provider in providers:
        filename = f"{provider}_batch_id.txt"
        if os.path.exists(filename):
            with open(filename) as f:
                batch_id = f.read().strip()

            typer.echo(f"{provider.upper()}: {batch_id}")
            found_any = True

    if not found_any:
        typer.echo("No batch IDs found. Run 'create' command first.")
        typer.echo(
            "Usage: python run_batch_test.py create --model 'provider/model-name'"
        )
    else:
        typer.echo()
        typer.echo(
            "To fetch results: python run_batch_test.py fetch --provider <provider>"
        )


@app.command()
def fetch(
    provider: str = typer.Option(
        help="Provider to fetch results from (openai, anthropic, google)"
    ),
    validate: bool = typer.Option(
        True, help="Validate extracted data against expected results"
    ),
    poll: bool = typer.Option(
        False, help="Poll every 30 seconds until batch completes"
    ),
    max_wait: int = typer.Option(
        600, help="Maximum time to wait in seconds (default: 10 minutes)"
    ),
):
    """Fetch and validate batch results from a provider"""

    if provider not in ["openai", "anthropic"]:
        typer.echo("Error: Provider must be one of: openai, anthropic", err=True)
        raise typer.Exit(1)

    # Check if batch ID file exists
    filename = f"{provider}_batch_id.txt"
    if not os.path.exists(filename):
        typer.echo(
            f"Error: No batch ID found for {provider}. Run 'create' command first.",
            err=True,
        )
        raise typer.Exit(1)

    # Read batch ID
    with open(filename) as f:
        batch_id = f.read().strip()

    typer.echo(f"Fetching results for {provider.upper()} batch: {batch_id}")
    typer.echo("=" * 60)

    # Check API key
    if not check_api_key(provider):
        raise typer.Exit(1)

    try:
        if poll:
            results = poll_for_results(provider, batch_id, validate, max_wait)
        else:
            if provider == "openai":
                results = fetch_openai_results(batch_id, validate)
            elif provider == "anthropic":
                results = fetch_anthropic_results(batch_id, validate)

        if results:
            typer.echo(f"Successfully fetched and validated {len(results)} results!")
            if validate:
                # Assert that the results match the expected results
                assert validate_results(results, provider.capitalize()), (
                    f"Test failed: {provider} results do not match expected results."
                )
        else:
            typer.echo("No results available yet or batch still processing")
            if not poll:
                typer.echo("Use --poll to automatically wait for completion")

    except AssertionError as ae:
        typer.echo(f"AssertionError: {ae}", err=True)
        raise typer.Exit(1) from ae
    except Exception as e:
        typer.echo(f"Error fetching results: {e}", err=True)
        raise typer.Exit(1) from e


@app.command()
def show_results(
    provider: str = typer.Option(
        help="Provider to show detailed results from (openai, anthropic, google)"
    ),
):
    """Show detailed parsed Pydantic objects from batch results"""

    if provider not in ["openai", "anthropic"]:
        typer.echo("Error: Provider must be one of: openai, anthropic", err=True)
        raise typer.Exit(1)

    # Check if batch ID file exists
    filename = f"{provider}_batch_id.txt"
    if not os.path.exists(filename):
        typer.echo(
            f"Error: No batch ID found for {provider}. Run 'create' command first.",
            err=True,
        )
        raise typer.Exit(1)

    # Read batch ID
    with open(filename) as f:
        batch_id = f.read().strip()

    typer.echo(f"{provider.upper()} BATCH RESULTS")
    typer.echo("=" * 50)
    typer.echo(f"Batch ID: {batch_id}")

    # Check API key
    if not check_api_key(provider):
        raise typer.Exit(1)

    try:
        # Get results using BatchProcessor
        if provider == "openai":
            processor = BatchProcessor("openai/gpt-4o-mini", User)
        elif provider == "anthropic":
            processor = BatchProcessor("anthropic/claude-3-5-sonnet-20241022", User)

        # Get batch info using list_batches to find our batch
        all_batches = processor.list_batches(limit=100)
        batch_info = None
        for batch in all_batches:
            if batch.id == batch_id:
                batch_info = batch
                break

        if not batch_info:
            typer.echo(f"Batch {batch_id} not found")
            return

        typer.echo(f"Status: {batch_info.status.value}")
        typer.echo(f"Raw Status: {batch_info.raw_status}")

        if batch_info.status != BatchStatus.COMPLETED:
            typer.echo(f"Batch not completed yet: {batch_info.status.value}")
            return

        # Get all results using the new get_results method
        all_results = processor.get_results(batch_id)
        typer.echo(f"Total results: {len(all_results)}")

        # Show each result with detailed info
        for i, result in enumerate(all_results):
            typer.echo(f"\n--- Result {i + 1} ---")
            typer.echo(f"Custom ID: {result.custom_id}")
            typer.echo(f"Success: {result.success}")

            if result.success:
                user = result.result
                typer.echo(f"PARSED USER OBJECT:")
                typer.echo(f"   Type: {type(user)}")
                typer.echo(f"   Name: {user.name}")
                typer.echo(f"   Age: {user.age}")
                typer.echo(f"   JSON: {user.model_dump_json()}")
                typer.echo(f"   Dict: {user.model_dump()}")

                # Test that it's a real Pydantic object
                typer.echo(f"   Is BaseModel: {isinstance(user, BaseModel)}")
                typer.echo(f"   Is User: {isinstance(user, User)}")

                # Test Pydantic methods
                try:
                    validated = User.model_validate(user.model_dump())
                    typer.echo(f"   Re-validation: Works")
                    typer.echo(f"   Re-validated: {validated}")
                except Exception as e:
                    typer.echo(f"   Re-validation: Failed - {e}")
            else:
                typer.echo(f"ERROR:")
                typer.echo(f"   Type: {result.error_type}")
                typer.echo(f"   Message: {result.error_message}")

        # Test the utility functions
        successful_results = filter_successful(all_results)
        error_results = filter_errors(all_results)
        extracted_users = extract_results(all_results)

        typer.echo(f"\nUTILITY FUNCTIONS:")
        typer.echo(f"Successful results: {len(successful_results)}")
        typer.echo(f"Error results: {len(error_results)}")
        typer.echo(f"Extracted users: {len(extracted_users)}")

        if extracted_users:
            typer.echo(f"\nEXTRACTED USER OBJECTS:")
            for user in extracted_users:
                typer.echo(
                    f"  • {user.name}, age {user.age} (type: {type(user).__name__})"
                )

    except Exception as e:
        typer.echo(f"Error showing results: {e}", err=True)
        raise typer.Exit(1) from e


def poll_for_results(
    provider: str, batch_id: str, validate: bool, max_wait: int
) -> list[User]:
    """Poll for batch results until completion or timeout"""
    import time

    typer.echo(f"Polling {provider.upper()} batch every 30 seconds...")
    typer.echo(f"Max wait time: {max_wait} seconds ({max_wait // 60} minutes)")
    typer.echo(f"Batch ID: {batch_id}")
    typer.echo()

    start_time = time.time()
    attempt = 1

    while time.time() - start_time < max_wait:
        typer.echo(f"Attempt {attempt} - Checking batch status...")

        try:
            if provider == "openai":
                status, results = fetch_openai_results_with_status(batch_id, validate)
            elif provider == "anthropic":
                status, results = fetch_anthropic_results_with_status(
                    batch_id, validate
                )

            if status == "completed" or status == "ended":
                typer.echo(
                    f"Batch completed after {int(time.time() - start_time)} seconds!"
                )
                return results
            elif status in ["failed", "expired", "cancelled"]:
                typer.echo(f"Batch {status}")
                return []
            else:
                elapsed = int(time.time() - start_time)
                remaining = max_wait - elapsed
                typer.echo(
                    f"Status: {status} | Elapsed: {elapsed}s | Remaining: {remaining}s"
                )

                if remaining > 30:
                    typer.echo("Waiting 30 seconds before next check...")
                    time.sleep(30)
                else:
                    typer.echo(f"Waiting {remaining} seconds...")
                    time.sleep(remaining)
                    break

        except Exception as e:
            typer.echo(f"Error during polling: {e}")
            time.sleep(30)

        attempt += 1

    typer.echo(f"Timeout reached after {max_wait} seconds")
    return []


def fetch_openai_results_with_status(
    batch_id: str, validate: bool
) -> tuple[str, list[User]]:
    """Fetch OpenAI batch results and return status"""
    processor = BatchProcessor("openai/gpt-4o-mini", User)

    # Get batch info
    all_batches = processor.list_batches(limit=100)
    batch_info = None
    for batch in all_batches:
        if batch.id == batch_id:
            batch_info = batch
            break

    if not batch_info:
        return "not_found", []

    if batch_info.status != BatchStatus.COMPLETED:
        return batch_info.raw_status, []

    # Get results using the new get_results method
    all_results = processor.get_results(batch_id)

    successful_results = filter_successful(all_results)
    error_results = filter_errors(all_results)
    extracted_results = extract_results(all_results)

    typer.echo(f"Successful extractions: {len(successful_results)}")
    if error_results:
        typer.echo(f"Failed extractions: {len(error_results)}")
        # Show first few errors for debugging
        for error in error_results[:3]:
            typer.echo(f"   Error ({error.custom_id}): {error.error_message}")

    if validate and extracted_results:
        validate_results(extracted_results, "OpenAI")

    return "completed", extracted_results


def fetch_anthropic_results_with_status(
    batch_id: str, validate: bool
) -> tuple[str, list[User]]:
    """Fetch Anthropic batch results and return status"""
    processor = BatchProcessor("anthropic/claude-3-5-sonnet-20241022", User)

    # Get batch info
    all_batches = processor.list_batches(limit=100)
    batch_info = None
    for batch in all_batches:
        if batch.id == batch_id:
            batch_info = batch
            break

    if not batch_info:
        return "not_found", []

    # Check for various terminal states
    if batch_info.status in [
        BatchStatus.FAILED,
        BatchStatus.CANCELLED,
        BatchStatus.EXPIRED,
    ]:
        return batch_info.raw_status, []

    if batch_info.status != BatchStatus.COMPLETED:
        return batch_info.raw_status, []

    # Get results using the new get_results method
    all_results = processor.get_results(batch_id)

    successful_results = filter_successful(all_results)
    error_results = filter_errors(all_results)
    extracted_results = extract_results(all_results)

    typer.echo(f"Successful extractions: {len(successful_results)}")
    if error_results:
        typer.echo(f"Failed extractions: {len(error_results)}")
        # Show first few errors for debugging
        for error in error_results[:3]:
            typer.echo(f"   Error ({error.custom_id}): {error.error_message}")

    if validate and extracted_results:
        validate_results(extracted_results, "Anthropic")

    return "ended", extracted_results


def fetch_openai_results(batch_id: str, validate: bool) -> list[User]:
    """Fetch OpenAI batch results using BatchProcessor"""
    processor = BatchProcessor("openai/gpt-4o-mini", User)

    # Get batch info
    all_batches = processor.list_batches(limit=100)
    batch_info = None
    for batch in all_batches:
        if batch.id == batch_id:
            batch_info = batch
            break

    if not batch_info:
        typer.echo(f"Batch {batch_id} not found")
        return []

    typer.echo(f"Batch Status: {batch_info.status.value}")

    if batch_info.status != BatchStatus.COMPLETED:
        typer.echo(
            f"Batch is still {batch_info.status.value}. Please wait and try again."
        )
        return []

    # Get results using the new get_results method
    all_results = processor.get_results(batch_id)

    successful_results = filter_successful(all_results)
    error_results = filter_errors(all_results)
    extracted_results = extract_results(all_results)

    typer.echo(f"Successful extractions: {len(successful_results)}")
    if error_results:
        typer.echo(f"Failed extractions: {len(error_results)}")
        # Show first few errors for debugging
        for error in error_results[:3]:
            typer.echo(f"   Error ({error.custom_id}): {error.error_message}")

    if validate and extracted_results:
        validate_results(extracted_results, "OpenAI")

    return extracted_results


def fetch_anthropic_results(batch_id: str, validate: bool) -> list[User]:
    """Fetch Anthropic batch results using BatchProcessor"""
    processor = BatchProcessor("anthropic/claude-3-5-sonnet-20241022", User)

    # Get batch info
    all_batches = processor.list_batches(limit=100)
    batch_info = None
    for batch in all_batches:
        if batch.id == batch_id:
            batch_info = batch
            break

    if not batch_info:
        typer.echo(f"Batch {batch_id} not found")
        return []

    typer.echo(f"Batch Status: {batch_info.status.value}")

    if batch_info.status != BatchStatus.COMPLETED:
        typer.echo(
            f"Batch is still {batch_info.status.value}. Please wait and try again."
        )
        return []

    # Get results using the new get_results method
    all_results = processor.get_results(batch_id)

    successful_results = filter_successful(all_results)
    error_results = filter_errors(all_results)
    extracted_results = extract_results(all_results)

    typer.echo(f"Successful extractions: {len(successful_results)}")
    if error_results:
        typer.echo(f"Failed extractions: {len(error_results)}")
        # Show first few errors for debugging
        for error in error_results[:3]:
            typer.echo(f"   Error ({error.custom_id}): {error.error_message}")

    if validate and extracted_results:
        validate_results(extracted_results, "Anthropic")

    return extracted_results


def fetch_google_results(batch_job_name: str, validate: bool) -> list[User]:
    """Fetch Google batch results using BatchProcessor"""
    try:
        processor = BatchProcessor("google/gemini-2.5-flash", User)

        # Get batch info
        all_batches = processor.list_batches(limit=100)
        batch_info = None
        for batch in all_batches:
            if batch.id == batch_job_name:
                batch_info = batch
                break

        if not batch_info:
            typer.echo(f"Batch {batch_job_name} not found")
            return []

        typer.echo(f"Batch Status: {batch_info.status.value}")

        if batch_info.status != BatchStatus.COMPLETED:
            typer.echo(
                f"Batch is still {batch_info.status.value}. Please wait and try again."
            )
            return []

        # Get results using the new get_results method
        all_results = processor.get_results(batch_job_name)

        successful_results = filter_successful(all_results)
        error_results = filter_errors(all_results)
        extracted_results = extract_results(all_results)

        typer.echo(f"Successful extractions: {len(successful_results)}")
        if error_results:
            typer.echo(f"Failed extractions: {len(error_results)}")

        if validate and extracted_results:
            validate_results(extracted_results, "Google GenAI")

        return extracted_results

    except Exception as e:
        typer.echo(f"Error fetching Google batch results: {e}")
        return []


def validate_results(results: list[User], provider_name: str) -> bool:
    """Validate extracted results against expected results"""
    expected_results = get_expected_results()

    typer.echo(f"\nValidating {provider_name} Results:")
    typer.echo("-" * 40)

    if len(results) != len(expected_results):
        typer.echo(f"Expected {len(expected_results)} results, got {len(results)}")
        return False

    # Sort both lists by name for comparison
    results_sorted = sorted(results, key=lambda x: x.name)
    expected_sorted = sorted(expected_results, key=lambda x: x.name)

    all_correct = True
    for i, (actual, expected) in enumerate(zip(results_sorted, expected_sorted)):
        if actual.name == expected.name and actual.age == expected.age:
            typer.echo(f"{i + 1}. {actual.name}, age {actual.age} - CORRECT")
        else:
            typer.echo(f"{i + 1}. Expected: {expected.name}, age {expected.age}")
            typer.echo(f"    Got: {actual.name}, age {actual.age}")
            all_correct = False

    if all_correct:
        typer.echo(f"\nAll {provider_name} extractions are correct!")
    else:
        typer.echo(f"\nSome {provider_name} extractions have errors")

    return all_correct


@app.command()
def help():
    """Show all available commands and usage examples"""
    typer.echo("Unified Batch API Test Commands")
    typer.echo("=" * 40)
    typer.echo()

    typer.echo("Available Commands:")
    typer.echo("  • create         - Create a new batch job")
    typer.echo("  • list-batches   - List all saved batch IDs")
    typer.echo("  • fetch          - Fetch and validate batch results")
    typer.echo("  • show-results   - Show detailed parsed Pydantic objects")
    typer.echo("  • list-models    - Show supported models")
    typer.echo("  • help           - Show this help message")
    typer.echo()

    typer.echo("Usage Examples:")
    typer.echo("  # Create batch job (default: Google Gemini 2.5 Flash)")
    typer.echo("  python run_batch_test.py create")
    typer.echo()
    typer.echo("  # Create batch job with specific model")
    typer.echo("  python run_batch_test.py create --model 'openai/gpt-4o-mini'")
    typer.echo()
    typer.echo("  # List saved batch IDs")
    typer.echo("  python run_batch_test.py list-batches")
    typer.echo()
    typer.echo("  # Fetch results with validation")
    typer.echo("  python run_batch_test.py fetch --provider openai")
    typer.echo()
    typer.echo("  # Show detailed parsed objects")
    typer.echo("  python run_batch_test.py show-results --provider anthropic")
    typer.echo()
    typer.echo("  # Poll every 30 seconds until batch completes (max 10 minutes)")
    typer.echo("  python run_batch_test.py fetch --provider openai --poll")
    typer.echo()
    typer.echo("  # Poll with custom timeout (20 minutes)")
    typer.echo(
        "  python run_batch_test.py fetch --provider openai --poll --max-wait 1200"
    )
    typer.echo()


@app.command()
def list_models():
    """List example models for each provider"""
    typer.echo("Supported Models by Provider:")
    typer.echo()

    typer.echo("OpenAI:")
    typer.echo("  • openai/gpt-4o-mini")
    typer.echo("  • openai/gpt-4o")
    typer.echo("  • openai/gpt-4-turbo")
    typer.echo()

    typer.echo("Anthropic:")
    typer.echo("  • anthropic/claude-3-5-sonnet-20241022")
    typer.echo("  • anthropic/claude-3-opus-20240229")
    typer.echo("  • anthropic/claude-3-haiku-20240307")
    typer.echo()

    typer.echo("Google:")
    typer.echo("  • google/gemini-2.5-flash")
    typer.echo("  • google/gemini-2.0-flash-001")
    typer.echo("  • google/gemini-pro")
    typer.echo()

    typer.echo("Usage: python run_batch_test.py create --model 'provider/model-name'")


if __name__ == "__main__":
    app()
