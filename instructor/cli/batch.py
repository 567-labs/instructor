import os
from rich.console import Console
from rich.table import Table
from rich.live import Live
import typer
import time
import json
import warnings
from instructor.batch import BatchProcessor, BatchJobInfo

from tqdm import tqdm

app = typer.Typer()

console = Console()

# Placeholder models used when an operation only needs the provider name
PROVIDER_MODEL_MAP = {
    "openai": "openai/gpt-4o-mini",
    "anthropic": "anthropic/claude-3-sonnet",
    "mistral": "mistral/mistral-small-latest",
}

PROVIDER_API_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}


def generate_table(
    batch_jobs: list[BatchJobInfo], provider: str, full_id: bool = False
):
    """Generate enhanced table for batch jobs using unified BatchJobInfo objects

    Args:
        batch_jobs: List of batch job info objects
        provider: Provider name (openai, anthropic, mistral)
        full_id: If True, show full batch IDs without truncation
    """
    table = Table(title=f"{provider.title()} Batch Jobs")

    # Adjust column width based on full_id flag
    id_max_width = None if full_id else 20
    table.add_column("Batch ID", style="dim", max_width=id_max_width, no_wrap=True)
    table.add_column("Status", min_width=10)
    table.add_column("Created", style="dim", min_width=10)
    table.add_column("Started", style="dim", min_width=10)
    table.add_column("Duration", style="dim", min_width=7)

    # Add provider-specific columns for request counts
    if provider == "openai":
        table.add_column("Completed", justify="right", min_width=8)
        table.add_column("Failed", justify="right", min_width=6)
        table.add_column("Total", justify="right", min_width=6)
    elif provider == "anthropic":
        table.add_column("Succeeded", justify="right", min_width=8)
        table.add_column("Errored", justify="right", min_width=7)
        table.add_column("Processing", justify="right", min_width=9)
    elif provider == "mistral":
        table.add_column("Succeeded", justify="right", min_width=8)
        table.add_column("Failed", justify="right", min_width=6)
        table.add_column("Total", justify="right", min_width=6)

    for batch_job in batch_jobs:
        # Color code status
        status_color = {
            "pending": "yellow",
            "processing": "blue",
            "completed": "green",
            "failed": "red",
            "cancelled": "red",
            "expired": "red",
        }.get(batch_job.status.value, "white")

        colored_status = f"[{status_color}]{batch_job.status.value}[/{status_color}]"

        # Format timestamps
        created_str = (
            batch_job.timestamps.created_at.strftime("%m/%d %H:%M")
            if batch_job.timestamps.created_at
            else "N/A"
        )
        started_str = (
            batch_job.timestamps.started_at.strftime("%m/%d %H:%M")
            if batch_job.timestamps.started_at
            else "N/A"
        )

        # Calculate duration
        duration_str = "N/A"
        if batch_job.timestamps.started_at and batch_job.timestamps.completed_at:
            duration = (
                batch_job.timestamps.completed_at - batch_job.timestamps.started_at
            )
            total_minutes = duration.total_seconds() / 60
            if total_minutes < 60:
                duration_str = f"{int(total_minutes)}m"
            else:
                hours = total_minutes / 60
                duration_str = f"{hours:.1f}h"
        elif batch_job.timestamps.started_at and batch_job.status.value == "processing":
            from datetime import datetime, timezone

            duration = datetime.now(timezone.utc) - batch_job.timestamps.started_at
            total_minutes = duration.total_seconds() / 60
            if total_minutes < 60:
                duration_str = f"{int(total_minutes)}m"
            else:
                hours = total_minutes / 60
                duration_str = f"{hours:.1f}h"

        # Truncate batch ID for display only if full_id is False
        batch_id_display = str(batch_job.id)
        if not full_id and len(batch_id_display) > 18:
            batch_id_display = batch_id_display[:15] + "..."

        if provider == "openai":
            table.add_row(
                batch_id_display,
                colored_status,
                created_str,
                started_str,
                duration_str,
                str(batch_job.request_counts.completed or 0),
                str(batch_job.request_counts.failed or 0),
                str(batch_job.request_counts.total or 0),
            )
        elif provider == "anthropic":
            table.add_row(
                str(batch_job.id),
                colored_status,
                created_str,
                started_str,
                duration_str,
                str(batch_job.request_counts.succeeded or 0),
                str(batch_job.request_counts.errored or 0),
                str(batch_job.request_counts.processing or 0),
            )
        elif provider == "mistral":
            table.add_row(
                batch_id_display,
                colored_status,
                created_str,
                started_str,
                duration_str,
                str(batch_job.request_counts.succeeded or 0),
                str(batch_job.request_counts.failed or 0),
                str(batch_job.request_counts.total or 0),
            )

    return table


def get_jobs(limit: int = 10, provider: str = "openai") -> list[BatchJobInfo]:
    """Get batch jobs for the specified provider using BatchProcessor"""

    # We just need the provider part for listing batches
    if provider not in PROVIDER_MODEL_MAP:
        raise ValueError(f"Unsupported provider: {provider}")

    # Create a dummy response model (not used for listing)
    from pydantic import BaseModel

    class DummyModel(BaseModel):
        dummy: str = "dummy"

    try:
        # Create BatchProcessor instance
        processor = BatchProcessor(PROVIDER_MODEL_MAP[provider], DummyModel)
        # Get batch jobs
        return processor.list_batches(limit=limit)
    except Exception as e:
        console.print(f"[red]Error listing {provider} batch jobs: {e}[/red]")
        return []


@app.command(name="list", help="See all existing batch jobs")
def watch(
    limit: int = typer.Option(10, help="Total number of batch jobs to show"),
    poll: int = typer.Option(
        10, help="Time in seconds to wait for the batch job to complete"
    ),
    screen: bool = typer.Option(False, help="Enable or disable screen output"),
    live: bool = typer.Option(
        False, help="Enable live polling to continuously update the table"
    ),
    provider: str = typer.Option(
        "openai",
        help="Provider to use (e.g., 'openai', 'anthropic', 'mistral')",
    ),
    # Deprecated flag for backward compatibility
    use_anthropic: bool = typer.Option(
        None,
        help="[DEPRECATED] Use --model instead. Use Anthropic API instead of OpenAI",
    ),
    full_id: bool = typer.Option(
        False,
        "--full-id",
        help="Show full batch IDs without truncation",
    ),
):
    """
    Monitor the status of the most recent batch jobs
    """
    # Handle deprecated flag
    if use_anthropic is not None:
        warnings.warn(
            "--use-anthropic is deprecated. Use --provider 'anthropic' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if use_anthropic:
            provider = "anthropic"

    # Check if required API key is available for the provider
    if provider in PROVIDER_API_KEYS and not os.getenv(PROVIDER_API_KEYS[provider]):
        console.print(
            f"[red]Error: {PROVIDER_API_KEYS[provider]} environment variable not set for {provider}[/red]"
        )
        return

    batch_jobs = get_jobs(limit, provider)
    table = generate_table(batch_jobs, provider, full_id=full_id)

    if not live:
        # Show table once and exit
        console.print(table)
        return

    # Live polling mode
    with Live(table, refresh_per_second=2, screen=screen) as live_table:
        while True:
            batch_jobs = get_jobs(limit, provider)
            table = generate_table(batch_jobs, provider, full_id=full_id)
            live_table.update(table)
            time.sleep(poll)


@app.command(
    help="Create a batch job from a file",
)
def create_from_file(
    file_path: str = typer.Option(help="File containing the batch job requests"),
    model: str = typer.Option(
        "openai/gpt-4o-mini",
        help="Model in format 'provider/model-name' (e.g., 'openai/gpt-4', 'anthropic/claude-3-sonnet', 'mistral/mistral-small-latest')",
    ),
    description: str = typer.Option(
        "Instructor batch job",
        help="Description/metadata for the batch job",
    ),
    completion_window: str = typer.Option(
        "24h",
        help="Completion window for the batch job (OpenAI only)",
    ),
    # Deprecated flag for backward compatibility
    use_anthropic: bool = typer.Option(
        None,
        help="[DEPRECATED] Use --model instead. Use Anthropic API instead of OpenAI",
    ),
):
    """Create a batch job from a file using the unified BatchProcessor"""
    # Handle deprecated flag
    if use_anthropic is not None:
        warnings.warn(
            "--use-anthropic is deprecated. Use --model 'anthropic/claude-3-sonnet' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if use_anthropic:
            model = "anthropic/claude-3-sonnet"

    try:
        # Create a dummy response model (not used for direct file submission)
        from pydantic import BaseModel

        class DummyModel(BaseModel):
            dummy: str = "dummy"

        # Create BatchProcessor instance
        processor = BatchProcessor(model, DummyModel)

        # Prepare metadata
        metadata = {
            "description": description,
        }

        with console.status(f"[bold green]Submitting batch job...", spinner="dots"):
            batch_id = processor.submit_batch(
                file_path, metadata=metadata, completion_window=completion_window
            )

        console.print(f"[bold green]Batch job created with ID: {batch_id}[/bold green]")

        # Show updated batch list
        provider_name = model.split("/", 1)[0]
        watch(limit=5, poll=2, screen=False, live=False, provider=provider_name)

    except Exception as e:
        console.print(f"[bold red]Error creating batch job: {e}[/bold red]")


@app.command(help="Cancel a batch job")
def cancel(
    batch_id: str = typer.Option(help="Batch job ID to cancel"),
    provider: str = typer.Option(
        "openai",
        help="Provider to use (e.g., 'openai', 'anthropic', 'mistral')",
    ),
    # Deprecated flag for backward compatibility
    use_anthropic: bool = typer.Option(
        None,
        help="[DEPRECATED] Use --provider 'anthropic' instead. Use Anthropic API instead of OpenAI",
    ),
):
    """Cancel a batch job using the unified BatchProcessor"""
    # Handle deprecated flag
    if use_anthropic is not None:
        warnings.warn(
            "--use-anthropic is deprecated. Use --provider 'anthropic' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if use_anthropic:
            provider = "anthropic"

    try:
        # Create a dummy response model (not used for cancellation)
        from pydantic import BaseModel

        class DummyModel(BaseModel):
            dummy: str = "dummy"

        if provider not in PROVIDER_MODEL_MAP:
            console.print(f"[red]Unsupported provider: {provider}[/red]")
            return

        # Create BatchProcessor instance
        processor = BatchProcessor(PROVIDER_MODEL_MAP[provider], DummyModel)

        with console.status(
            f"[bold yellow]Cancelling {provider} batch job...", spinner="dots"
        ):
            processor.cancel_batch(batch_id)

        console.print(
            f"[bold green]Batch {batch_id} cancelled successfully![/bold green]"
        )

        # Show updated status
        watch(limit=5, poll=2, screen=False, live=False, provider=provider)

    except NotImplementedError as e:
        console.print(f"[yellow]Note: {e}[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error cancelling batch {batch_id}: {e}[/bold red]")


@app.command(help="Delete a completed batch job")
def delete(
    batch_id: str = typer.Option(help="Batch job ID to delete"),
    provider: str = typer.Option(
        "openai",
        help="Provider to use (e.g., 'openai', 'anthropic', 'mistral')",
    ),
):
    """Delete a batch job using the unified BatchProcessor"""
    try:
        # Create a dummy response model (not used for deletion)
        from pydantic import BaseModel

        class DummyModel(BaseModel):
            dummy: str = "dummy"

        if provider not in PROVIDER_MODEL_MAP:
            console.print(f"[red]Unsupported provider: {provider}[/red]")
            return

        # Create BatchProcessor instance
        processor = BatchProcessor(PROVIDER_MODEL_MAP[provider], DummyModel)

        with console.status(
            f"[bold yellow]Deleting {provider} batch job...", spinner="dots"
        ):
            processor.delete_batch(batch_id)

        console.print(
            f"[bold green]Batch {batch_id} deleted successfully![/bold green]"
        )

        # Show updated status
        watch(limit=5, poll=2, screen=False, live=False, provider=provider)

    except NotImplementedError as e:
        console.print(f"[yellow]Note: {e}[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error deleting batch {batch_id}: {e}[/bold red]")


@app.command(help="Download the file associated with a batch job")
def download_file(
    batch_id: str = typer.Option(help="Batch job ID to download"),
    download_file_path: str = typer.Option(help="Path to download file to"),
    provider: str = typer.Option(
        "openai",
        help="Provider to use (e.g., 'openai', 'anthropic', 'mistral')",
    ),
):
    try:
        if provider == "anthropic":
            from anthropic import Anthropic

            client = Anthropic()
            # TODO: Remove beta fallback when stable API is available
            try:
                batches_client = client.messages.batches
            except AttributeError:
                batches_client = client.beta.messages.batches
            batch = batches_client.retrieve(batch_id)
            if batch.processing_status != "ended":
                raise ValueError("Only completed Jobs can be downloaded")

            results_url = batch.results_url
            if not results_url:
                raise ValueError("Results URL not available")

            with open(download_file_path, "w") as file:
                for result in tqdm(client.messages.batches.results(batch_id)):
                    file.write(json.dumps(result.model_dump()) + "\n")
        elif provider == "mistral":
            from instructor.batch.providers import get_provider

            get_provider("mistral").download_results(batch_id, download_file_path)
        else:
            from openai import OpenAI

            client = OpenAI()
            batch = client.batches.retrieve(batch_id=batch_id)
            status = batch.status

            if status != "completed":
                raise ValueError("Only completed Jobs can be downloaded")

            file_id = batch.output_file_id

            assert file_id, f"Equivalent Output File not found for {batch_id}"
            file_response = client.files.content(file_id)

            with open(download_file_path, "w") as file:
                file.write(file_response.text)

    except Exception as e:
        console.log(f"[bold red]Error downloading file for {batch_id}: {e}")


@app.command(help="Retrieve results from a batch job")
def results(
    batch_id: str = typer.Option(help="Batch job ID to get results from"),
    output_file: str = typer.Option(help="File to save the results to"),
    model: str = typer.Option(
        "openai/gpt-4o-mini",
        help="Model in format 'provider/model-name' (e.g., 'openai/gpt-4', 'anthropic/claude-3-sonnet', 'mistral/mistral-small-latest')",
    ),
):
    """Retrieve and save batch job results"""
    provider, _ = model.split("/", 1)

    try:
        if provider == "openai":
            from openai import OpenAI

            client = OpenAI()
            batch = client.batches.retrieve(batch_id=batch_id)

            if batch.status != "completed":
                console.print(
                    f"[yellow]Batch status is '{batch.status}', not completed[/yellow]"
                )
                return

            file_id = batch.output_file_id
            if not file_id:
                console.print("[red]No output file available[/red]")
                return

            file_response = client.files.content(file_id)
            with open(output_file, "w") as f:
                f.write(file_response.text)
            console.print(f"[bold green]Results saved to: {output_file}[/bold green]")

        elif provider == "anthropic":
            from anthropic import Anthropic

            client = Anthropic()
            batch = client.beta.messages.batches.retrieve(batch_id)

            if batch.processing_status != "ended":
                console.print(
                    f"[yellow]Batch status is '{batch.processing_status}', not ended[/yellow]"
                )
                return

            # Get results from Anthropic batch API
            results_iter = client.beta.messages.batches.results(batch_id)

            with open(output_file, "w") as f:
                for result in results_iter:
                    f.write(json.dumps(result.model_dump()) + "\n")
            console.print(f"[bold green]Results saved to: {output_file}[/bold green]")

        elif provider == "mistral":
            from instructor.batch.providers import get_provider

            get_provider("mistral").download_results(batch_id, output_file)
            console.print(f"[bold green]Results saved to: {output_file}[/bold green]")

        else:
            console.print(f"[red]Unsupported provider: {provider}[/red]")

    except Exception as e:
        console.log(f"[bold red]Error retrieving results for {batch_id}: {e}")


@app.command(help="Create batch job using BatchProcessor")
def create(
    messages_file: str = typer.Option(help="JSONL file with message conversations"),
    model: str = typer.Option(
        "openai/gpt-4o-mini",
        help="Model in format 'provider/model-name' (e.g., 'openai/gpt-4', 'anthropic/claude-3-sonnet', 'mistral/mistral-small-latest')",
    ),
    response_model: str = typer.Option(
        help="Python class path for response model (e.g., 'examples.User')"
    ),
    output_file: str = typer.Option(
        "batch_requests.jsonl", help="Output file for batch requests"
    ),
    max_tokens: int = typer.Option(1000, help="Maximum tokens per request"),
    temperature: float = typer.Option(0.1, help="Temperature for generation"),
):
    """Create a batch job using the unified BatchProcessor"""
    try:
        # Import the response model dynamically
        module_path, class_name = response_model.rsplit(".", 1)
        import importlib

        module = importlib.import_module(module_path)
        response_class = getattr(module, class_name)

        # Load messages from file
        messages_list = []
        with open(messages_file) as f:
            for line in f:
                if line.strip():
                    messages_list.append(json.loads(line))

        # Create batch processor
        processor = BatchProcessor(model, response_class)

        # Create batch file
        with console.status(
            f"[bold green]Creating batch file with {len(messages_list)} requests...",
            spinner="dots",
        ):
            processor.create_batch_from_messages(
                messages_list, output_file, max_tokens, temperature
            )

        console.print(f"[bold green]Batch file created: {output_file}[/bold green]")
        console.print(
            f"[yellow]Use 'instructor batch create-from-file --file-path {output_file}' to submit the batch[/yellow]"
        )

    except Exception as e:
        console.log(f"[bold red]Error creating batch: {e}")
