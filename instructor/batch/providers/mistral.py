"""Mistral batch processing provider for mistralai 1.x and 2.x."""

from instructor.v2.providers._mistral_sdk import (
    load_mistral_client_type as _load_mistral_client_type,
)
import io
import logging
import os
from typing import Any, Optional, Union

from ..models import BatchJobInfo
from .base import BatchProvider, _atomic_result_file, _open_jsonl, _submit_errors

logger = logging.getLogger(__name__)

# Stopped jobs keep the records that finished before they stopped.
_STOPPED_STATUSES = frozenset({"CANCELLED", "TIMEOUT_EXCEEDED"})
# A failed job may still carry an error file with per-request errors.
_FINISHED_STATUSES = _STOPPED_STATUSES | {"SUCCESS", "FAILED"}


class MistralProvider(BatchProvider):
    """Mistral batch processing provider"""

    def _client(self) -> Any:
        """Build a Mistral client from MISTRAL_API_KEY"""
        client_type = _load_mistral_client_type()
        if client_type is None:
            raise RuntimeError(
                "Mistral batch jobs require the mistralai package. "
                'Install it with pip install "instructor[mistral]"'
            )

        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY environment variable is not set")

        return client_type(api_key=api_key)

    def _results_text(self, client: Any, batch_id: str) -> str:
        """Return success and error records for a finished Mistral job"""
        job = client.batch.jobs.get(job_id=batch_id)

        if job.status not in _FINISHED_STATUSES:
            raise RuntimeError(f"Batch not completed, status: {job.status}")

        # Unset SDK fields are falsy, so missing and null files are skipped.
        file_ids = dict.fromkeys(
            file_id
            for file_id in (job.output_file, getattr(job, "error_file", None))
            if file_id
        )
        if not file_ids:
            if job.status in _STOPPED_STATUSES and not (
                getattr(job, "completed_requests", 0) or 0
            ):
                # No request finished, so no result file will ever appear.
                return ""
            raise RuntimeError(
                f"Batch has no output or error file available, status: {job.status}"
            )

        contents = [
            client.files.download(file_id=file_id).read().decode("utf-8")
            for file_id in file_ids
        ]
        if len(contents) == 1:
            return contents[0]
        # Each file is JSONL; the output file need not end with a newline.
        return "\n".join(content.rstrip("\n") for content in contents)

    def submit_batch(
        self,
        file_path_or_buffer: Union[str, io.BytesIO],
        metadata: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Submit Mistral batch job"""
        # Mistral takes the model at job creation rather than per request
        model = kwargs.get("model")
        if not model:
            raise ValueError("Mistral batch jobs require a model")

        client = self._client()
        with _submit_errors("Mistral"), _open_jsonl(file_path_or_buffer) as f:
            logger.debug(f"Submitting batch job with metadata: {metadata}")
            # Inline batching is unavailable on SDK 1.x, so always upload a file
            input_file = client.files.upload(
                file={"file_name": "batch.jsonl", "content": f}, purpose="batch"
            )
            batch_job = client.batch.jobs.create(
                input_files=[input_file.id],
                model=model,
                endpoint=kwargs.get("endpoint", "/v1/chat/completions"),
                metadata=metadata or {},
            )
            logger.info(f"Successfully submitted batch job: {batch_job.id}")
            return batch_job.id

    def get_status(self, batch_id: str) -> dict[str, Any]:
        """Get Mistral batch status"""
        client = self._client()
        try:
            job = client.batch.jobs.get(job_id=batch_id)
            return {
                "id": job.id,
                "status": job.status,
                "created_at": job.created_at,
                "request_counts": {
                    "total": getattr(job, "total_requests", None),
                    "succeeded": getattr(job, "succeeded_requests", None),
                    "failed": getattr(job, "failed_requests", None),
                },
            }
        except Exception as e:
            raise RuntimeError(f"Failed to get Mistral batch status: {e}") from e

    def retrieve_results(self, batch_id: str) -> str:
        """Retrieve Mistral batch results"""
        client = self._client()
        try:
            return self._results_text(client, batch_id)
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve Mistral results: {e}") from e

    def download_results(self, batch_id: str, file_path: str) -> None:
        """Download Mistral batch results to a file"""
        results = self.retrieve_results(batch_id)
        with _atomic_result_file(file_path) as output:
            output.write(results.encode("utf-8"))

    def cancel_batch(self, batch_id: str) -> dict[str, Any]:
        """Cancel Mistral batch job"""
        client = self._client()
        try:
            return client.batch.jobs.cancel(job_id=batch_id).model_dump()
        except Exception as e:
            raise RuntimeError(f"Failed to cancel Mistral batch: {e}") from e

    def delete_batch(self, batch_id: str) -> dict[str, Any]:
        """Delete Mistral batch job"""
        client = self._client()

        # The delete endpoint only exists on SDK 2.x
        delete = getattr(client.batch.jobs, "delete", None)
        if delete is None:
            raise NotImplementedError(
                "Deleting batch jobs requires mistralai>=2.0.0; "
                "the installed SDK has no delete endpoint"
            )

        try:
            return delete(job_id=batch_id).model_dump()
        except Exception as e:
            raise RuntimeError(f"Failed to delete Mistral batch: {e}") from e

    def list_batches(self, limit: int = 10) -> list[BatchJobInfo]:
        """List Mistral batch jobs"""
        client = self._client()
        try:
            response = client.batch.jobs.list(page_size=limit)
            jobs = getattr(response, "data", None) or []
            return [BatchJobInfo.from_mistral(job.model_dump()) for job in jobs[:limit]]
        except Exception as e:
            raise RuntimeError(f"Failed to list Mistral batches: {e}") from e
