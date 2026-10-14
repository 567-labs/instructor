"""
Unified Batch Processing API for Multiple Providers

This module provides a unified interface for batch processing across OpenAI and Anthropic
providers. The API uses a Maybe/Result-like pattern with custom_id
tracking for type-safe handling of batch results.

Supported Providers:
- OpenAI: 50% cost savings on batch requests
- Anthropic: 50% cost savings on batch requests (Message Batches API)

Features:
- Type-safe Maybe/Result pattern for handling successes and errors
- Custom ID tracking for correlating results to original requests
- Unified interface across all providers
- Helper functions for filtering and extracting results

Example usage:
    from instructor.batch import BatchProcessor, filter_successful, extract_results
    from pydantic import BaseModel

    class User(BaseModel):
        name: str
        age: int

    processor = BatchProcessor("openai/gpt-4o-mini", User)
    batch_id = processor.submit_batch("requests.jsonl")

    # Results are BatchSuccess[T] | BatchError union types
    all_results = processor.retrieve_results(batch_id)
    successful_results = filter_successful(all_results)
    extracted_users = extract_results(all_results)

Documentation:
- OpenAI Batch API: https://platform.openai.com/docs/guides/batch
- Anthropic Message Batches: https://docs.anthropic.com/en/api/creating-message-batches
"""

from typing import Any, Optional

# Import all public symbols from the modules
from .models import (
    BatchSuccess,
    BatchError,
    BatchStatus,
    BatchTimestamps,
    BatchRequestCounts,
    BatchErrorInfo,
    BatchFiles,
    BatchJobInfo,
    BatchResult,
    T,
)
from .utils import (
    filter_successful,
    filter_errors,
    extract_results,
    get_results_by_custom_id,
)
from .request import (
    BatchRequest,
    Function,
    Tool,
    RequestBody,
    BatchModel,
)
from .processor import BatchProcessor


class BatchJob:
    """Legacy BatchJob class for backward compatibility"""

    @classmethod
    def parse_from_file(
        cls, file_path: str, response_model: type[T]
    ) -> tuple[list[T], list[dict[Any, Any]]]:
        with open(file_path, encoding="utf-8") as file:
            content = file.read()
        return cls.parse_from_string(content, response_model)

    @classmethod
    def parse_from_string(
        cls, content: str, response_model: type[T]
    ) -> tuple[list[T], list[dict[Any, Any]]]:
        """Enhanced parser that works with all providers using JSON schema"""
        import json

        res: list[T] = []
        error_objs: list[dict[Any, Any]] = []

        lines = content.strip().split("\n")
        for line in lines:
            if not line.strip():
                continue

            try:
                data = json.loads(line)
                extracted_data = cls._extract_structured_data(data)

                if extracted_data is not None:
                    try:
                        result = response_model.model_validate(extracted_data)
                        res.append(result)
                    except Exception:
                        error_objs.append(data)
                else:
                    error_objs.append(data)

            except Exception:
                error_objs.append({"error": "Failed to parse JSON", "raw_line": line})

        return res, error_objs

    @classmethod
    def _extract_structured_data(cls, data: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Extract structured data from various provider response formats"""
        import json

        try:
            # Try OpenAI JSON schema format first
            if "response" in data and "body" in data["response"]:
                choices = data["response"]["body"].get("choices", [])
                if choices:
                    message = choices[0].get("message", {})

                    # JSON schema response
                    if "content" in message:
                        content = message["content"]
                        if isinstance(content, str):
                            return json.loads(content)

                    # Tool calls (legacy)
                    if "tool_calls" in message:
                        tool_call = message["tool_calls"][0]
                        return json.loads(tool_call["function"]["arguments"])

            # Try Anthropic format
            if "result" in data and "message" in data["result"]:
                content = data["result"]["message"]["content"]
                if isinstance(content, list) and len(content) > 0:
                    # Tool use response
                    for item in content:
                        if item.get("type") == "tool_use":
                            return item.get("input", {})
                    # Text response with JSON
                    for item in content:
                        if item.get("type") == "text":
                            text = item.get("text", "")
                            return json.loads(text)

        except Exception:
            pass

        return None


# Define what gets exported when someone does "from instructor.batch import *"
__all__ = [
    # Core types
    "T",
    "BatchResult",
    # Models
    "BatchSuccess",
    "BatchError",
    "BatchStatus",
    "BatchTimestamps",
    "BatchRequestCounts",
    "BatchErrorInfo",
    "BatchFiles",
    "BatchJobInfo",
    # Utility functions
    "filter_successful",
    "filter_errors",
    "extract_results",
    "get_results_by_custom_id",
    # Request models
    "BatchRequest",
    "Function",
    "Tool",
    "RequestBody",
    "BatchModel",
    # Main processor
    "BatchProcessor",
    # Legacy
    "BatchJob",
]
