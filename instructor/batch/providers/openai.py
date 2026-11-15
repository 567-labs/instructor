"""
OpenAI-specific batch processing implementation.

This module contains the OpenAI batch processing provider class.
"""

from typing import Any, Optional, Union
import io
import logging
import time
from .base import BatchProvider, _atomic_result_file, _open_jsonl, _submit_errors
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
        with _submit_errors("OpenAI"), _open_jsonl(file_path_or_buffer) as f:
            from openai import OpenAI

            client = OpenAI()
            if metadata is None:
                metadata = {"description": "Instructor batch job"}
            logger.debug(f"Submitting batch job with metadata: {metadata}")
            # Name in-memory uploads so the API sees a JSONL file.
            upload = f if isinstance(file_path_or_buffer, str) else ("batch.jsonl", f)
            batch_file = client.files.create(file=upload, purpose="batch")
            batch_job = client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window=kwargs.get("completion_window", "24h"),
                metadata=metadata,
            )
            logger.info(f"Successfully submitted batch job: {batch_job.id}")
            return batch_job.id

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
        """Return success and error records for a finished OpenAI batch."""
        from openai import OpenAI
        from openai.types import Batch

        client = OpenAI()
        batch = client.batches.retrieve(batch_id)

        # Expired and cancelled batches keep records finished before they stopped.
        result_statuses = {"completed", "expired", "cancelled"}
        if batch.status not in result_statuses:
            raise Exception(f"Batch not completed, status: {batch.status}")

        def files_ready(batch: Batch) -> bool:
            counts = batch.request_counts
            if (
                batch.status in {"expired", "cancelled"}
                and counts is not None
                and not counts.completed
                and not counts.failed
            ):
                # No request finished, so no result file will ever appear.
                return True
            return bool(batch.output_file_id or batch.error_file_id) and (
                counts is None
                or (
                    (not counts.completed or bool(batch.output_file_id))
                    and (not counts.failed or bool(batch.error_file_id))
                )
            )

        if not files_ready(batch):
            max_retries = 10
            for attempt in range(max_retries):
                wait_time = min(5 + attempt, 15)
                print(
                    f"Result files not ready, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(wait_time)
                batch = client.batches.retrieve(batch_id)
                if batch.status not in result_statuses:
                    raise Exception(
                        f"Batch status changed to {batch.status} while waiting for result files"
                    )
                if files_ready(batch):
                    print("Result files now available")
                    break
            else:
                raise RuntimeError(
                    f"No result files available after {max_retries} retries over {sum(range(5, 5 + max_retries))} seconds. "
                    f"Batch status: {batch.status}, Request counts: {getattr(batch, 'request_counts', 'unknown')}."
                )

        file_ids = dict.fromkeys(
            file_id
            for file_id in (batch.output_file_id, batch.error_file_id)
            if file_id is not None
        )
        contents = [client.files.content(file_id).text for file_id in file_ids]
        if len(contents) == 1:
            return contents[0]
        # Each file is JSONL; the output file need not end with a newline.
        return "\n".join(content.rstrip("\n") for content in contents)

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
            with _atomic_result_file(file_path) as output:
                output.write(results_text.encode("utf-8"))
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
