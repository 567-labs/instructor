"""
Data models and types for batch processing.

This module contains all the Pydantic models, enums, and type definitions
used throughout the batch processing system.
"""

from __future__ import annotations
from typing import Any, Union, TypeVar, Generic
from typing_extensions import TypeAlias
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime, timezone
from enum import Enum

T = TypeVar("T", bound=BaseModel)


class BatchSuccess(BaseModel, Generic[T]):
    """Successful batch result with custom_id"""

    custom_id: str
    result: T
    success: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)


class BatchError(BaseModel):
    """Error information for failed batch requests"""

    custom_id: str
    error_type: str
    error_message: str
    success: bool = False
    raw_data: dict[str, Any] | None = None


class BatchStatus(str, Enum):
    """Normalized batch status across providers"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class BatchTimestamps(BaseModel):
    """Comprehensive timestamp tracking"""

    created_at: datetime | None = None
    started_at: datetime | None = None  # in_progress_at, processing start
    completed_at: datetime | None = None  # completed_at, ended_at
    failed_at: datetime | None = None
    cancelled_at: datetime | None = None
    expired_at: datetime | None = None
    expires_at: datetime | None = None


class BatchRequestCounts(BaseModel):
    """Unified request counts across providers"""

    total: int | None = None

    # OpenAI fields
    completed: int | None = None
    failed: int | None = None

    # Anthropic fields
    processing: int | None = None
    succeeded: int | None = None
    errored: int | None = None
    cancelled: int | None = None
    expired: int | None = None


class BatchErrorInfo(BaseModel):
    """Batch-level error information"""

    error_type: str | None = None
    error_message: str | None = None
    error_code: str | None = None


class BatchFiles(BaseModel):
    """File references for batch job"""

    input_file_id: str | None = None
    output_file_id: str | None = None
    error_file_id: str | None = None
    results_url: str | None = None  # Anthropic


class BatchJobInfo(BaseModel):
    """Enhanced unified batch job information with comprehensive provider support"""

    # Core identifiers
    id: str
    provider: str

    # Status information
    status: BatchStatus
    raw_status: str  # Original provider status

    # Timing information
    timestamps: BatchTimestamps

    # Request tracking
    request_counts: BatchRequestCounts

    # File references
    files: BatchFiles

    # Error information
    error: BatchErrorInfo | None = None

    # Provider-specific data
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw_data: dict[str, Any] | None = None

    # Additional fields
    model: str | None = None
    endpoint: str | None = None
    completion_window: str | None = None

    @classmethod
    def from_openai(cls, batch_data: dict[str, Any]) -> BatchJobInfo:
        """Create from OpenAI batch response"""
        # Normalize status
        status_map = {
            "validating": BatchStatus.PENDING,
            "in_progress": BatchStatus.PROCESSING,
            "finalizing": BatchStatus.PROCESSING,
            "completed": BatchStatus.COMPLETED,
            "failed": BatchStatus.FAILED,
            "expired": BatchStatus.EXPIRED,
            "cancelled": BatchStatus.CANCELLED,
            "cancelling": BatchStatus.CANCELLED,
        }

        # Parse timestamps
        timestamps = BatchTimestamps(
            created_at=(
                datetime.fromtimestamp(batch_data["created_at"], tz=timezone.utc)
                if batch_data.get("created_at")
                else None
            ),
            started_at=(
                datetime.fromtimestamp(batch_data["in_progress_at"], tz=timezone.utc)
                if batch_data.get("in_progress_at")
                else None
            ),
            completed_at=(
                datetime.fromtimestamp(batch_data["completed_at"], tz=timezone.utc)
                if batch_data.get("completed_at")
                else None
            ),
            failed_at=(
                datetime.fromtimestamp(batch_data["failed_at"], tz=timezone.utc)
                if batch_data.get("failed_at")
                else None
            ),
            cancelled_at=(
                datetime.fromtimestamp(batch_data["cancelled_at"], tz=timezone.utc)
                if batch_data.get("cancelled_at")
                else None
            ),
            expired_at=(
                datetime.fromtimestamp(batch_data["expired_at"], tz=timezone.utc)
                if batch_data.get("expired_at")
                else None
            ),
            expires_at=(
                datetime.fromtimestamp(batch_data["expires_at"], tz=timezone.utc)
                if batch_data.get("expires_at")
                else None
            ),
        )

        # Parse request counts
        request_counts_data = batch_data.get("request_counts", {})
        request_counts = BatchRequestCounts(
            total=request_counts_data.get("total"),
            completed=request_counts_data.get("completed"),
            failed=request_counts_data.get("failed"),
        )

        # Parse files
        files = BatchFiles(
            input_file_id=batch_data.get("input_file_id"),
            output_file_id=batch_data.get("output_file_id"),
            error_file_id=batch_data.get("error_file_id"),
        )

        # Parse error information
        error = None
        if batch_data.get("errors"):
            error_data = batch_data["errors"]
            error = BatchErrorInfo(
                error_type=error_data.get("type"),
                error_message=error_data.get("message"),
                error_code=error_data.get("code"),
            )

        return cls(
            id=batch_data["id"],
            provider="openai",
            status=status_map.get(batch_data["status"], BatchStatus.PENDING),
            raw_status=batch_data["status"],
            timestamps=timestamps,
            request_counts=request_counts,
            files=files,
            error=error,
            metadata=batch_data.get("metadata", {}),
            raw_data=batch_data,
            endpoint=batch_data.get("endpoint"),
            completion_window=batch_data.get("completion_window"),
        )

    @classmethod
    def from_anthropic(cls, batch_data: dict[str, Any]) -> BatchJobInfo:
        """Create from Anthropic batch response"""
        # Normalize status
        status_map = {
            "in_progress": BatchStatus.PROCESSING,
            "ended": BatchStatus.COMPLETED,
            "failed": BatchStatus.FAILED,
            "cancelled": BatchStatus.CANCELLED,
            "expired": BatchStatus.EXPIRED,
        }

        # Parse timestamps
        def parse_iso_timestamp(timestamp_value):
            if not timestamp_value:
                return None
            try:
                # Handle different timestamp format variations
                if isinstance(timestamp_value, datetime):
                    return timestamp_value
                if isinstance(timestamp_value, str):
                    return datetime.fromisoformat(
                        timestamp_value.replace("Z", "+00:00")
                    )
                return None
            except (ValueError, AttributeError):
                return None

        timestamps = BatchTimestamps(
            created_at=parse_iso_timestamp(batch_data.get("created_at")),
            started_at=parse_iso_timestamp(
                batch_data.get("created_at")
            ),  # Anthropic doesn't provide started_at, use created_at
            cancelled_at=parse_iso_timestamp(batch_data.get("cancel_initiated_at")),
            completed_at=parse_iso_timestamp(batch_data.get("ended_at")),
            expires_at=parse_iso_timestamp(batch_data.get("expires_at")),
        )

        # Parse request counts
        request_counts_data = batch_data.get("request_counts", {})
        request_counts = BatchRequestCounts(
            processing=request_counts_data.get("processing"),
            succeeded=request_counts_data.get("succeeded"),
            errored=request_counts_data.get("errored"),
            cancelled=request_counts_data.get(
                "canceled"
            ),  # Note: Anthropic uses "canceled"
            expired=request_counts_data.get("expired"),
            total=request_counts_data.get("processing", 0)
            + request_counts_data.get("succeeded", 0)
            + request_counts_data.get("errored", 0)
            + request_counts_data.get("canceled", 0)
            + request_counts_data.get("expired", 0),
        )

        # Parse files
        files = BatchFiles(
            results_url=batch_data.get("results_url"),
        )

        return cls(
            id=batch_data["id"],
            provider="anthropic",
            status=status_map.get(batch_data["processing_status"], BatchStatus.PENDING),
            raw_status=batch_data["processing_status"],
            timestamps=timestamps,
            request_counts=request_counts,
            files=files,
            raw_data=batch_data,
        )

    @classmethod
    def from_mistral(cls, batch_data: dict[str, Any]) -> BatchJobInfo:
        """Create from Mistral batch response"""
        # Normalize status
        status_map = {
            "QUEUED": BatchStatus.PENDING,
            "RUNNING": BatchStatus.PROCESSING,
            "SUCCESS": BatchStatus.COMPLETED,
            "FAILED": BatchStatus.FAILED,
            "TIMEOUT_EXCEEDED": BatchStatus.EXPIRED,
            "CANCELLATION_REQUESTED": BatchStatus.CANCELLED,
            "CANCELLED": BatchStatus.CANCELLED,
        }

        # Mistral reports timestamps as Unix seconds
        def parse_timestamp(timestamp_value):
            if isinstance(timestamp_value, datetime):
                return timestamp_value
            if isinstance(timestamp_value, (int, float)) and not isinstance(
                timestamp_value, bool
            ):
                try:
                    return datetime.fromtimestamp(timestamp_value, tz=timezone.utc)
                except (OverflowError, OSError, ValueError):
                    return None
            if isinstance(timestamp_value, str):
                try:
                    return datetime.fromisoformat(
                        timestamp_value.replace("Z", "+00:00")
                    )
                except ValueError:
                    return None
            return None

        timestamps = BatchTimestamps(
            created_at=parse_timestamp(batch_data.get("created_at")),
            started_at=parse_timestamp(batch_data.get("started_at")),
            completed_at=parse_timestamp(batch_data.get("completed_at")),
        )

        request_counts = BatchRequestCounts(
            total=batch_data.get("total_requests"),
            completed=batch_data.get("completed_requests"),
            succeeded=batch_data.get("succeeded_requests"),
            failed=batch_data.get("failed_requests"),
        )

        input_files = batch_data.get("input_files") or []
        files = BatchFiles(
            input_file_id=input_files[0] if input_files else None,
            output_file_id=batch_data.get("output_file"),
            error_file_id=batch_data.get("error_file"),
        )

        # Mistral returns a list of job-level errors; surface the first one
        errors = batch_data.get("errors") or []
        error = None
        if errors and isinstance(errors[0], dict):
            error = BatchErrorInfo(
                error_type="mistral_error",
                error_message=errors[0].get("message"),
            )

        raw_status = batch_data.get("status", "QUEUED")
        return cls(
            id=batch_data["id"],
            provider="mistral",
            status=status_map.get(raw_status, BatchStatus.PENDING),
            raw_status=raw_status,
            timestamps=timestamps,
            request_counts=request_counts,
            files=files,
            error=error,
            metadata=batch_data.get("metadata") or {},
            raw_data=batch_data,
            model=batch_data.get("model"),
            endpoint=batch_data.get("endpoint"),
        )


# Union type for batch results - like a Maybe/Result type
BatchResult: TypeAlias = Union[BatchSuccess[Any], BatchError]
