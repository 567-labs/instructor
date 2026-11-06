#!/usr/bin/env python3
"""Example of using in-memory batching for serverless deployments.

This example shows how to create and submit batch requests without writing to disk
"""

import time
from pydantic import BaseModel
from instructor.batch.processor import BatchProcessor


class User(BaseModel):
    """User model for extraction."""

    name: str
    age: int
    email: str


def main():
    """Demonstrate in-memory batch processing."""
    print("In-Memory Batch Processing Example")
    print("===================================\n")

    # Initialize batch processor
    # Note: Use gpt-4o-mini for JSON schema support in batch API
    processor = BatchProcessor("openai/gpt-4o-mini", User)

    # Sample messages for batch processing
    messages_list = [
        [
            {"role": "system", "content": "Extract user information from the text."},
            {
                "role": "user",
                "content": "John Doe is 25 years old and his email is john@example.com",
            },
        ],
        [
            {"role": "system", "content": "Extract user information from the text."},
            {
                "role": "user",
                "content": "Jane Smith, age 30, can be reached at jane.smith@company.com",
            },
        ],
        [
            {"role": "system", "content": "Extract user information from the text."},
            {
                "role": "user",
                "content": "Bob Wilson (bob.wilson@email.com) is 28 years old",
            },
        ],
    ]

    print("Creating batch requests in memory...")

    # Create batch in memory (no file_path specified)
    batch_buffer = processor.create_batch_from_messages(
        messages_list,
        file_path=None,  # This triggers in-memory mode
        max_tokens=150,
        temperature=0.1,
    )

    print(f"Created batch buffer: {type(batch_buffer)}")
    print(f"Buffer size: {len(batch_buffer.getvalue())} bytes\n")

    # Show the content of the buffer (first 200 chars)
    batch_buffer.seek(0)
    content_preview = batch_buffer.read(200).decode("utf-8")
    print("Buffer content preview:")
    print(f"{content_preview}...\n")

    # Reset buffer position for submission
    batch_buffer.seek(0)

    print("Submitting batch job...")

    try:
        # Submit the batch using the in-memory buffer
        batch_id = processor.submit_batch(
            batch_buffer, metadata={"description": "In-memory batch example"}
        )

        print(f"Batch submitted successfully!")
        print(f"Batch ID: {batch_id}")

        # Poll for completion
        print("\nWaiting for batch to complete...")
        max_wait_time = 300  # 5 minutes max
        start_time = time.time()
        status = {}

        while time.time() - start_time < max_wait_time:
            status = processor.get_batch_status(batch_id)
            current_status = status.get("status", "unknown")

            # Update status on the same line
            print(f"\rCurrent status: {current_status.ljust(20)}", end="")

            if current_status in ["completed", "failed", "cancelled", "expired"]:
                break

            time.sleep(10)

        print()  # Newline after polling is done

        # Use the last fetched status
        final_status = status
        print(f"\nFinal status: {final_status.get('status', 'unknown')}")

        if final_status.get("status") == "completed":
            print("\nBatch completed! Retrieving results...")

            # Retrieve and process results
            results = processor.get_results(batch_id)

            print(f"\nResults Summary:")
            print(f"   Total results: {len(results)}")

            successful_results = [r for r in results if hasattr(r, "result")]
            error_results = [r for r in results if hasattr(r, "error_message")]

            print(f"   Successful: {len(successful_results)}")
            print(f"   Errors: {len(error_results)}")

            # Show successful extractions
            if successful_results:
                print("\nExtracted Users:")
                for result in successful_results:
                    user = result.result
                    print(f"   - {user.name}, {user.age} years old, {user.email}")

            # Show any errors
            if error_results:
                print("\nErrors encountered:")
                for error in error_results:
                    print(f"   - {error.custom_id}: {error.error_message}")

        elif final_status.get("status") == "failed":
            print("\nBatch failed to complete")
            print("   Check your API usage and batch format")

        else:
            print(f"\nBatch did not complete within {max_wait_time} seconds")
            print(f"   Current status: {final_status.get('status', 'unknown')}")
            print(
                "   You can check status later with processor.get_batch_status(batch_id)"
            )

    except Exception as e:
        print(f"Error during batch processing: {e}")
        print("\nThis is expected if you don't have OpenAI API credentials set up.")
        print(
            "   The important part is that the in-memory buffer was created successfully!"
        )

    print("\nIn-memory batch processing demo complete!")
    print("\nKey benefits of in-memory batching:")
    print("   - No disk I/O required - perfect for serverless")
    print("   - Faster processing - no file system overhead")
    print("   - Better security - no temporary files on disk")
    print("   - Cleaner code - no file cleanup required")


def compare_file_vs_memory():
    """Compare file-based vs in-memory batch creation."""
    print("\nComparing File-based vs In-Memory Batching")
    print("===========================================\n")

    processor = BatchProcessor("openai/gpt-4o-mini", User)

    messages_list = [
        [{"role": "user", "content": "Extract: John, 25, john@example.com"}],
        [{"role": "user", "content": "Extract: Jane, 30, jane@example.com"}],
    ]

    # File-based approach (traditional)
    print("File-based approach:")
    file_path = processor.create_batch_from_messages(
        messages_list,
        file_path="temp_batch.jsonl",  # Specify file path
    )
    print(f"   Created file: {file_path}")

    # Clean up the file
    import os

    if os.path.exists(file_path):
        os.remove(file_path)
        print("   File cleaned up")

    # In-memory approach (new)
    print("\nIn-memory approach:")
    buffer = processor.create_batch_from_messages(
        messages_list,
        file_path=None,  # No file path = in-memory
    )
    print(f"   Created buffer: {type(buffer).__name__}")
    print(f"   Buffer size: {len(buffer.getvalue())} bytes")
    print("   No cleanup required!")


def demo_polling_logic():
    """Demonstrate how to properly poll for batch completion."""
    print("\nBatch Polling Best Practices")
    print("============================\n")

    print("When working with real batches, follow this pattern:")
    print("")
    print("```python")
    print("import time")
    print("")
    print("# Submit your batch")
    print("batch_id = processor.submit_batch(buffer)")
    print("")
    print("# Poll for completion")
    print("while True:")
    print("    status = processor.get_batch_status(batch_id)")
    print("    current_status = status.get('status')")
    print("    ")
    print("    if current_status == 'completed':")
    print("        results = processor.get_results(batch_id)")
    print("        break")
    print("    elif current_status in ['failed', 'cancelled', 'expired']:")
    print("        print(f'Batch failed with status: {current_status}')")
    print("        break")
    print("    else:")
    print("        print(f'Status: {current_status}, waiting...')")
    print("        time.sleep(10)  # Wait 10 seconds before checking again")
    print("```")
    print("")
    print("Typical batch statuses:")
    print("   - validating - Checking request format")
    print("   - in_progress - Processing requests")
    print("   - finalizing - Preparing results")
    print("   - completed - Ready for download")
    print("   - failed - Something went wrong")
    print("   - cancelled - Manually cancelled")
    print("   - expired - Took too long to process")


if __name__ == "__main__":
    main()
    compare_file_vs_memory()
