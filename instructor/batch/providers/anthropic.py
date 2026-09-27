"""
Anthropic-specific batch processing implementation.

This module contains the Anthropic batch processing provider class.
"""

import json
from collections.abc import Iterator
from typing import Any, Optional, Union
import io
import logging
from .base import BatchProvider
from ..models import BatchJobInfo

logger = logging.getLogger(__name__)


class AnthropicProvider(BatchProvider):
    """Anthropic batch processing provider"""

    def submit_batch(
        self,
        file_path_or_buffer: Union[str, io.BytesIO],
        metadata: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Submit Anthropic batch job"""
        _ = kwargs  # Unused but accepted for API consistency
        try:
            import anthropic

            client = anthropic.Anthropic()

            # Note: Anthropic doesn't support metadata in batch creation
            # but we accept it for API consistency
            if metadata:
                print(
                    f"Note: Anthropic batches don't support metadata. Ignoring: {metadata}"
                )

            # TODO(#batch-api-stable): Remove beta fallback when stable API is available
            try:
                batches_client = client.messages.batches
            except AttributeError:
                batches_client = client.beta.messages.batches

            if isinstance(file_path_or_buffer, str):
                with open(file_path_or_buffer, encoding="utf-8") as f:
                    requests = [json.loads(line) for line in f if line.strip()]
            elif isinstance(file_path_or_buffer, io.BytesIO):
                file_path_or_buffer.seek(0)
                content = file_path_or_buffer.read().decode("utf-8")
                requests = [
                    json.loads(line) for line in content.split("\n") if line.strip()
                ]
            else:
                raise ValueError(
                    f"Unsupported file_path_or_buffer type: {type(file_path_or_buffer)}"
                )

            batch = batches_client.create(requests=requests)
            return batch.id
        except (ValueError, TypeError) as e:
            # Re-raise validation errors as-is
            logger.error(f"Validation error in Anthropic batch submission: {e}")
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to submit Anthropic batch: {e}") from e

    def get_status(self, batch_id: str) -> dict[str, Any]:
        """Get Anthropic batch status"""
        try:
            import anthropic

            client = anthropic.Anthropic()

            # TODO(#batch-api-stable): Remove beta fallback when stable API is available
            try:
                batches_client = client.messages.batches
            except AttributeError:
                batches_client = client.beta.messages.batches

            batch = batches_client.retrieve(batch_id)
            return {
                "id": batch.id,
                "status": batch.processing_status,
                "created_at": batch.created_at,
                "request_counts": getattr(batch, "request_counts", {}),
            }
        except Exception as e:
            raise Exception(f"Failed to get Anthropic batch status: {e}") from e

    def _iter_result_lines(self, batch_id: str) -> Iterator[str]:
        """Yield serialized results for a completed Anthropic batch."""
        import anthropic

        client = anthropic.Anthropic()

        # TODO(#batch-api-stable): Remove beta fallback when stable API is available
        try:
            batches_client = client.messages.batches
        except AttributeError:
            batches_client = client.beta.messages.batches

        batch = batches_client.retrieve(batch_id)

        if batch.processing_status in ["failed", "cancelled", "expired"]:
            raise Exception(f"Batch job failed with status: {batch.processing_status}")

        if batch.processing_status != "ended":
            raise Exception(f"Batch not completed, status: {batch.processing_status}")

        request_counts = getattr(batch, "request_counts", None)
        if request_counts:
            succeeded = getattr(request_counts, "succeeded", 0)
            errored = getattr(request_counts, "errored", 0)
            total = getattr(request_counts, "total", 0)

            if errored > 0 and succeeded == 0:
                raise RuntimeError(
                    f"All {total} batch requests failed. No results will be available."
                )

        return (result.model_dump_json() for result in batches_client.results(batch_id))

    def retrieve_results(self, batch_id: str) -> str:
        """Retrieve Anthropic batch results"""
        try:
            return "\n".join(self._iter_result_lines(batch_id))
        except Exception as e:
            raise Exception(f"Failed to retrieve Anthropic results: {e}") from e

    def download_results(self, batch_id: str, file_path: str) -> None:
        """Download Anthropic batch results to a file"""
        try:
            result_lines = self._iter_result_lines(batch_id)
            with open(file_path, "w", encoding="utf-8") as f:
                for line in result_lines:
                    f.write(line + "\n")
        except Exception as e:
            raise Exception(f"Failed to download Anthropic results: {e}") from e

    def cancel_batch(self, batch_id: str) -> dict[str, Any]:
        """Cancel Anthropic batch job"""
        try:
            import anthropic

            client = anthropic.Anthropic()

            # TODO(#batch-api-stable): Remove beta fallback when stable API is available
            try:
                batches_client = client.messages.batches
            except AttributeError:
                batches_client = client.beta.messages.batches

            batch = batches_client.cancel(batch_id)
            return batch.model_dump()
        except Exception as e:
            raise Exception(f"Failed to cancel Anthropic batch: {e}") from e

    def delete_batch(self, batch_id: str) -> dict[str, Any]:
        """Delete Anthropic batch job"""
        try:
            import anthropic

            client = anthropic.Anthropic()

            # TODO(#batch-api-stable): Remove beta fallback when stable API is available
            try:
                batches_client = client.messages.batches
            except AttributeError:
                batches_client = client.beta.messages.batches

            batch = batches_client.retrieve(batch_id)
            return {
                "id": batch.id,
                "status": batch.processing_status,
                "message": "Anthropic does not support batch deletion",
            }
        except Exception as e:
            raise Exception(f"Failed to delete Anthropic batch: {e}") from e

    def list_batches(self, limit: int = 10) -> list[BatchJobInfo]:
        """List Anthropic batch jobs"""
        try:
            import anthropic

            client = anthropic.Anthropic()

            # TODO(#batch-api-stable): Remove beta fallback when stable API is available
            try:
                batches_client = client.messages.batches
            except AttributeError:
                batches_client = client.beta.messages.batches

            batches = batches_client.list(limit=limit)
            return [
                BatchJobInfo.from_anthropic(batch.model_dump())
                for batch in batches.data
            ]
        except Exception as e:
            raise Exception(f"Failed to list Anthropic batches: {e}") from e
