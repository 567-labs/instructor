"""
Mistral-specific batch processing implementation.

This module contains the Mistral batch processing provider class. It supports
both Mistral SDK export layouts: ``mistralai.client.Mistral`` on 2.x and
``mistralai.Mistral`` on 1.x.
"""

import importlib
import io
import logging
import os
from typing import Any, Optional, Union

from ..models import BatchJobInfo
from .base import BatchProvider

logger = logging.getLogger(__name__)

# SDK 2.x export first, then the SDK 1.x fallback
_CLIENT_MODULES = ("mistralai.client", "mistralai")


def _load_mistral_client_type() -> Optional[type]:
    """Load the client class from the Mistral 2.x or 1.x export"""
    for module_name in _CLIENT_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        client_type = getattr(module, "Mistral", None)
        if isinstance(client_type, type):
            return client_type
    return None


def _to_dict(obj: Any) -> dict[str, Any]:
    """Normalize an SDK response object into a plain dict"""
    if isinstance(obj, dict):
        return obj
    return obj.model_dump()


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

    def _upload_input_file(
        self, client: Any, file_path_or_buffer: Union[str, io.BytesIO]
    ) -> str:
        """Upload batch requests and return the resulting file ID"""
        if isinstance(file_path_or_buffer, str):
            logger.debug(f"Uploading batch file from path: {file_path_or_buffer}")
            with open(file_path_or_buffer, "rb") as f:
                batch_file = client.files.upload(
                    file={"file_name": "batch.jsonl", "content": f},
                    purpose="batch",
                )
        elif isinstance(file_path_or_buffer, io.BytesIO):
            logger.debug("Uploading batch file from BytesIO buffer")
            file_path_or_buffer.seek(0)
            batch_file = client.files.upload(
                file={"file_name": "batch.jsonl", "content": file_path_or_buffer},
                purpose="batch",
            )
        else:
            raise ValueError(
                f"Unsupported file_path_or_buffer type: {type(file_path_or_buffer)}"
            )

        return batch_file.id

    def _completed_job(self, client: Any, batch_id: str) -> Any:
        """Fetch a batch job, verifying it produced downloadable results"""
        job = client.batch.jobs.get(job_id=batch_id)

        if job.status != "SUCCESS":
            raise RuntimeError(f"Batch not completed, status: {job.status}")

        succeeded = getattr(job, "succeeded_requests", 0) or 0
        failed = getattr(job, "failed_requests", 0) or 0
        if failed and not succeeded:
            total = getattr(job, "total_requests", succeeded + failed)
            raise RuntimeError(
                f"All {total} batch requests failed. No results will be available."
            )

        if not job.output_file:
            raise RuntimeError("Batch has no output file ID available")

        return job

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
        try:
            logger.debug(f"Submitting batch job with metadata: {metadata}")

            # Inline batching is unavailable on SDK 1.x, so always upload a file
            input_file_id = self._upload_input_file(client, file_path_or_buffer)

            batch_job = client.batch.jobs.create(
                input_files=[input_file_id],
                model=model,
                endpoint=kwargs.get("endpoint", "/v1/chat/completions"),
                metadata=metadata or {},
            )
            logger.info(f"Successfully submitted batch job: {batch_job.id}")
            return batch_job.id
        except (ValueError, TypeError) as e:
            # Re-raise validation errors as-is
            logger.error(f"Validation error in Mistral batch submission: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to submit Mistral batch: {e}")
            raise RuntimeError(f"Failed to submit Mistral batch: {e}") from e

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
            job = self._completed_job(client, batch_id)
            response = client.files.download(file_id=job.output_file)
            return response.read().decode("utf-8")
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve Mistral results: {e}") from e

    def download_results(self, batch_id: str, file_path: str) -> None:
        """Download Mistral batch results to a file"""
        client = self._client()
        try:
            job = self._completed_job(client, batch_id)
            response = client.files.download(file_id=job.output_file)
            with open(file_path, "wb") as f:
                f.write(response.read())
        except Exception as e:
            raise RuntimeError(f"Failed to download Mistral results: {e}") from e

    def cancel_batch(self, batch_id: str) -> dict[str, Any]:
        """Cancel Mistral batch job"""
        client = self._client()
        try:
            return _to_dict(client.batch.jobs.cancel(job_id=batch_id))
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
            return _to_dict(delete(job_id=batch_id))
        except Exception as e:
            raise RuntimeError(f"Failed to delete Mistral batch: {e}") from e

    def list_batches(self, limit: int = 10) -> list[BatchJobInfo]:
        """List Mistral batch jobs"""
        client = self._client()
        try:
            response = client.batch.jobs.list(page_size=limit)
            jobs = getattr(response, "data", None) or []
            return [BatchJobInfo.from_mistral(_to_dict(job)) for job in jobs[:limit]]
        except Exception as e:
            raise RuntimeError(f"Failed to list Mistral batches: {e}") from e
