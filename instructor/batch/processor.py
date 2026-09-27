"""
Batch processor for unified batch processing across providers.

This module contains the BatchProcessor class that provides a unified interface
for batch processing across different LLM providers.
"""

from __future__ import annotations
from typing import Any, Generic
import json
import os
import io
import re
from .models import BatchResult, BatchSuccess, BatchError, BatchJobInfo, T
from .request import BatchRequest
from .providers import get_provider

_ANTHROPIC_CUSTOM_ID = re.compile(r"[a-zA-Z0-9_-]{1,64}")


class BatchProcessor(Generic[T]):
    """Unified batch processor that works across all providers"""

    def __init__(self, model: str, response_model: type[T]):
        self.model = model
        self.response_model = response_model

        # Parse provider from model string
        try:
            self.provider_name, self.model_name = model.split("/", 1)
        except ValueError as err:
            raise ValueError(
                'Model string must be in format "provider/model-name" '
                '(e.g. "openai/gpt-4" or "anthropic/claude-3-sonnet")'
            ) from err

        # Get the batch provider instance
        self.provider = get_provider(self.provider_name)

    def create_batch_from_messages(
        self,
        messages_list: list[list[dict[str, Any]]],
        file_path: str | None = None,
        max_tokens: int | None = 1000,
        temperature: float | None = 0.1,
        custom_ids: list[str] | None = None,
    ) -> str | io.BytesIO:
        """Create batch file from list of message conversations

        Args:
            messages_list: List of message conversations, each as a list of message dicts
            file_path: Path to save the batch request file. If None, returns BytesIO buffer
            max_tokens: Maximum tokens per request
            temperature: Temperature for generation
            custom_ids: Optional unique ID for each conversation, in the same order.
                Defaults to "request-0", "request-1", ...

        Returns:
            The file path where the batch was saved, or BytesIO buffer if file_path is None

        Raises:
            ValueError: If custom_ids does not match messages_list in length, contains
                duplicates, or (for Anthropic) does not match ^[a-zA-Z0-9_-]{1,64}$
        """
        if custom_ids is None:
            custom_ids = [f"request-{i}" for i in range(len(messages_list))]
        else:
            self._validate_custom_ids(custom_ids, len(messages_list))

        if file_path is not None:
            if os.path.exists(file_path):
                os.remove(file_path)

            batch_requests = []
            for custom_id, messages in zip(custom_ids, messages_list):
                batch_request = BatchRequest[T](
                    custom_id=custom_id,
                    messages=messages,
                    response_model=self.response_model,
                    model=self.model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                batch_request.save_to_file(file_path, self.provider_name)
                batch_requests.append(batch_request)

            print(f"Created batch file {file_path} with {len(batch_requests)} requests")
            return file_path
        # Create BytesIO buffer - caller is responsible for cleanup
        buffer = io.BytesIO()
        batch_requests = []
        for custom_id, messages in zip(custom_ids, messages_list):
            batch_request = BatchRequest[T](
                custom_id=custom_id,
                messages=messages,
                response_model=self.response_model,
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            batch_request.save_to_file(buffer, self.provider_name)
            batch_requests.append(batch_request)

        print(f"Created batch buffer with {len(batch_requests)} requests")
        buffer.seek(0)  # Reset buffer position for reading
        return buffer

    def _validate_custom_ids(self, custom_ids: list[str], expected: int) -> None:
        """Fail before writing requests if the provider would reject the IDs"""
        if len(custom_ids) != expected:
            raise ValueError(
                f"custom_ids has {len(custom_ids)} items but messages_list has {expected}"
            )
        if len(set(custom_ids)) != len(custom_ids):
            raise ValueError("custom_ids must be unique")
        if self.provider_name == "anthropic":
            invalid = [c for c in custom_ids if not _ANTHROPIC_CUSTOM_ID.fullmatch(c)]
            if invalid:
                raise ValueError(
                    f"Anthropic custom_ids must match {_ANTHROPIC_CUSTOM_ID.pattern}: {invalid}"
                )

    def submit_batch(
        self,
        file_path_or_buffer: str | io.BytesIO,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> str:
        """Submit batch job to the provider and return job ID

        Args:
            file_path_or_buffer: Path to the batch request file or BytesIO buffer
            metadata: Optional metadata to attach to the batch job
            **kwargs: Additional provider-specific arguments
        """
        if metadata is None:
            metadata = {"description": "Instructor batch job"}

        return self.provider.submit_batch(
            file_path_or_buffer, metadata=metadata, **kwargs
        )

    def get_batch_status(self, batch_id: str) -> dict[str, Any]:
        """Get batch job status from the provider"""
        return self.provider.get_status(batch_id)

    def retrieve_results(self, batch_id: str) -> list[BatchResult]:
        """Retrieve and parse batch results from the provider"""
        results_content = self.provider.retrieve_results(batch_id)
        return self.parse_results(results_content)

    def list_batches(self, limit: int = 10) -> list[BatchJobInfo]:
        """List batch jobs for the current provider

        Args:
            limit: Maximum number of batch jobs to return

        Returns:
            List of BatchJobInfo objects with normalized batch information
        """
        return self.provider.list_batches(limit)

    def get_results(
        self, batch_id: str, file_path: str | None = None
    ) -> list[BatchResult]:
        """Get batch results, optionally saving raw results to a file

        Args:
            batch_id: The batch job ID
            file_path: Optional file path to save raw results. If provided,
                      raw results will be saved to this file. If not provided,
                      results are only kept in memory.

        Returns:
            List of BatchResult objects (BatchSuccess[T] or BatchError)
        """
        # Retrieve results directly to memory
        results_content = self.retrieve_results(batch_id)

        # If file path is provided, save raw results to file
        if file_path is not None:
            self.provider.download_results(batch_id, file_path)

        return results_content

    def cancel_batch(self, batch_id: str) -> dict[str, Any]:
        """Cancel a batch job

        Args:
            batch_id: The batch job ID to cancel

        Returns:
            Dict containing the cancelled batch information
        """
        return self.provider.cancel_batch(batch_id)

    def delete_batch(self, batch_id: str) -> dict[str, Any]:
        """Delete a batch job (only available for completed batches)

        Args:
            batch_id: The batch job ID to delete

        Returns:
            Dict containing the deletion confirmation
        """
        return self.provider.delete_batch(batch_id)

    def parse_results(self, results_content: str) -> list[BatchResult]:
        """Parse batch results from content string into Maybe-like results with custom_id tracking"""
        results: list[BatchResult] = []

        lines = results_content.strip().split("\n")
        for line in lines:
            if not line.strip():
                continue

            try:
                data = json.loads(line)
                custom_id = data.get("custom_id", "unknown")
                extracted_data = self._extract_from_response(data)

                if extracted_data is not None:
                    try:
                        # Parse into response model
                        result = self.response_model(**extracted_data)
                        batch_result = BatchSuccess[T](
                            custom_id=custom_id, result=result
                        )
                        results.append(batch_result)
                    except Exception as e:
                        error_result = BatchError(
                            custom_id=custom_id,
                            error_type="parsing_error",
                            error_message=f"Failed to parse into {self.response_model.__name__}: {e}",
                            raw_data=extracted_data,
                        )
                        results.append(error_result)
                else:
                    # Check if this is a provider error response
                    error_message = "Unknown error"
                    error_type = "extraction_error"

                    if self.provider_name == "anthropic" and "result" in data:
                        result = data["result"]
                        if result.get("type") == "error":
                            error_info = result.get("error", {})
                            if isinstance(error_info, dict) and "error" in error_info:
                                error_details = error_info["error"]
                                error_message = error_details.get(
                                    "message", "Unknown Anthropic error"
                                )
                                error_type = error_details.get(
                                    "type", "anthropic_error"
                                )
                            else:
                                error_message = str(error_info)
                                error_type = "anthropic_error"

                    error_result = BatchError(
                        custom_id=custom_id,
                        error_type=error_type,
                        error_message=error_message,
                        raw_data=data,
                    )
                    results.append(error_result)

            except Exception as e:
                error_result = BatchError(
                    custom_id="unknown",
                    error_type="json_parse_error",
                    error_message=f"Failed to parse JSON: {e}",
                    raw_data={"raw_line": line},
                )
                results.append(error_result)

        return results

    def _extract_from_response(self, data: dict[str, Any]) -> dict[str, Any] | None:
        """Extract structured data from provider-specific response format"""
        try:
            if self.provider_name == "openai":
                # OpenAI JSON schema response
                content = data["response"]["body"]["choices"][0]["message"]["content"]
                return json.loads(content)

            if self.provider_name == "anthropic":
                # Anthropic batch response format
                if "result" not in data:
                    return None

                result = data["result"]

                # Check if result is an error
                if result.get("type") == "error":
                    # Return None to indicate error, let caller handle
                    return None

                # Handle successful message result
                if result.get("type") == "succeeded" and "message" in result:
                    content = result["message"]["content"]
                    if isinstance(content, list) and len(content) > 0:
                        # Try tool_use first
                        for item in content:
                            if item.get("type") == "tool_use":
                                return item.get("input", {})

                        # Fallback to text content and parse JSON
                        for item in content:
                            if item.get("type") == "text":
                                text = item.get("text", "")
                                try:
                                    return json.loads(text)
                                except json.JSONDecodeError:
                                    continue

                return None

        except Exception:
            return None

        return None
