"""
OpenAI-specific batch processing implementation.

This module contains the OpenAI batch processing provider class.
"""

from typing import Any, Optional, Union
import io
import logging
from .base import BatchProvider
from ..models import BatchJobInfo

logger = logging.getLogger(__name__)


class OpenAIProvider(BatchProvider):
    """OpenAI batch processing provider"""

    def submit_batch(
        self,
        file_path_or_buffer: Union[str, io.BytesIO],
        metadata: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Submit OpenAI batch job"""
        if not isinstance(file_path_or_buffer, (str, io.BytesIO)):
            raise ValueError(
                f"Unsupported file_path_or_buffer type: {type(file_path_or_buffer)}"
            )

        try:
            from openai import OpenAI

            client = OpenAI()

            if metadata is None:
                metadata = {"description": "Instructor batch job"}

            logger.debug(f"Submitting batch job with metadata: {metadata}")

            if isinstance(file_path_or_buffer, str):
                logger.debug(f"Creating batch file from path: {file_path_or_buffer}")
                with open(file_path_or_buffer, "rb") as f:
                    batch_file = client.files.create(file=f, purpose="batch")
            else:
                logger.debug("Creating batch file from BytesIO buffer")
                file_path_or_buffer.seek(0)
                batch_file = client.files.create(
                    file=file_path_or_buffer, purpose="batch"
                )

            batch_job = client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window=kwargs.get("completion_window", "24h"),
                metadata=metadata,
            )
            logger.info(f"Successfully submitted batch job: {batch_job.id}")
            return batch_job.id
        except (ValueError, TypeError) as e:
            # Re-raise validation errors as-is
            logger.error(f"Validation error in OpenAI batch submission: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to submit OpenAI batch: {e}")
            raise RuntimeError(f"Failed to submit OpenAI batch: {e}") from e

    def get_status(self, batch_id: str) -> dict[str, Any]:
        """Get OpenAI batch status"""
        try:
            from openai import OpenAI

            client = OpenAI()
            batch = client.batches.retrieve(batch_id)
            return {
                "id": batch.id,
                "status": batch.status,
                "created_at": batch.created_at,
                "request_counts": {
                    "total": getattr(batch.request_counts, "total", 0),
                    "completed": getattr(batch.request_counts, "completed", 0),
                    "failed": getattr(batch.request_counts, "failed", 0),
                },
            }
        except Exception as e:
            raise Exception(f"Failed to get OpenAI batch status: {e}") from e

    def _get_results_text(self, batch_id: str) -> str:
        """Return success and error records for a completed OpenAI batch."""
        from openai import OpenAI
        import time

        client = OpenAI()
        batch = client.batches.retrieve(batch_id)

        if batch.status != "completed":
            raise Exception(f"Batch not completed, status: {batch.status}")

        request_counts = getattr(batch, "request_counts", None)
        if request_counts:
            completed = getattr(request_counts, "completed", 0)
            failed = getattr(request_counts, "failed", 0)
            total = getattr(request_counts, "total", 0)

            if failed > 0 and completed == 0 and not batch.error_file_id:
                raise RuntimeError(
                    f"All {total} batch requests failed. No output file will be available."
                )

        if not batch.output_file_id and not batch.error_file_id:
            max_retries = 10
            for attempt in range(max_retries):
                wait_time = min(5 + attempt, 15)
                print(
                    f"Output file not ready, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(wait_time)
                batch = client.batches.retrieve(batch_id)
                if batch.output_file_id or batch.error_file_id:
                    if batch.output_file_id:
                        print(f"Output file now available: {batch.output_file_id}")
                    else:
                        print(f"Error file now available: {batch.error_file_id}")
                    break
                if batch.status != "completed":
                    raise Exception(
                        f"Batch status changed to {batch.status} while waiting for output file"
                    )
            else:
                raise RuntimeError(
                    f"No output file available after {max_retries} retries over {sum(range(5, 5 + max_retries))} seconds. "
                    f"Batch status: {batch.status}, Request counts: {getattr(batch, 'request_counts', 'unknown')}."
                )

        contents = [
            client.files.content(file_id).text
            for file_id in (batch.output_file_id, batch.error_file_id)
            if file_id
        ]
        # A file need not end with a newline. Keep adjacent JSONL records separate.
        return "\n".join(contents)

    def retrieve_results(self, batch_id: str) -> str:
        """Retrieve OpenAI batch results"""
        try:
            return self._get_results_text(batch_id)
        except Exception as e:
            raise Exception(f"Failed to retrieve OpenAI results: {e}") from e

    def download_results(self, batch_id: str, file_path: str) -> None:
        """Download OpenAI batch results to a file"""
        try:
            results_text = self._get_results_text(batch_id)
            with open(file_path, "w") as f:
                f.write(results_text)
        except Exception as e:
            raise Exception(f"Failed to download OpenAI results: {e}") from e

    def cancel_batch(self, batch_id: str) -> dict[str, Any]:
        """Cancel OpenAI batch job"""
        try:
            from openai import OpenAI

            client = OpenAI()
            batch = client.batches.cancel(batch_id)
            return batch.model_dump()
        except Exception as e:
            raise Exception(f"Failed to cancel OpenAI batch: {e}") from e

    def delete_batch(self, batch_id: str) -> dict[str, Any]:
        """Delete OpenAI batch job"""
        try:
            from openai import OpenAI

            client = OpenAI()
            # OpenAI doesn't have a delete endpoint, so we'll return the batch info
            batch = client.batches.retrieve(batch_id)
            return {
                "id": batch.id,
                "status": batch.status,
                "message": "OpenAI does not support batch deletion",
            }
        except Exception as e:
            raise Exception(f"Failed to delete OpenAI batch: {e}") from e

    def list_batches(self, limit: int = 10) -> list[BatchJobInfo]:
        """List OpenAI batch jobs"""
        try:
            from openai import OpenAI

            client = OpenAI()
            batches = client.batches.list(limit=limit)
            return [
                BatchJobInfo.from_openai(batch.model_dump()) for batch in batches.data
            ]
        except Exception as e:
            raise Exception(f"Failed to list OpenAI batches: {e}") from e
