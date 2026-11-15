"""
Base provider class for batch processing.

This module defines the abstract base class that all batch providers must implement.
"""

from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO, cast
from typing import Any, Optional, Union
import io
import logging
import os
import stat
from ..models import BatchJobInfo

logger = logging.getLogger(__name__)


@contextmanager
def _atomic_result_file(destination: str) -> Generator[BinaryIO, None, None]:
    """Replace a file only after its new contents are written, keeping its mode."""
    handle = NamedTemporaryFile(mode="wb", dir=Path(destination).parent, delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            yield cast(BinaryIO, handle)
        with suppress(FileNotFoundError):
            existing = os.lstat(destination).st_mode
            if stat.S_ISREG(existing):
                temporary.chmod(stat.S_IMODE(existing))
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@contextmanager
def _open_jsonl(source: Union[str, io.BytesIO]) -> Generator[BinaryIO, None, None]:
    """Open a request file path, or rewind an in-memory buffer."""
    if isinstance(source, str):
        with open(source, "rb") as f:
            yield f
    elif isinstance(source, io.BytesIO):
        source.seek(0)
        yield source
    else:
        raise ValueError(f"Unsupported file_path_or_buffer type: {type(source)}")


@contextmanager
def _submit_errors(provider: str) -> Generator[None, None, None]:
    """Re-raise validation errors; wrap everything else in RuntimeError."""
    try:
        yield
    except (ValueError, TypeError) as e:
        logger.error(
            "Validation error in %s batch submission (%s)", provider, type(e).__name__
        )
        raise
    except Exception as e:
        logger.error("Failed to submit %s batch (%s)", provider, type(e).__name__)
        raise RuntimeError(f"Failed to submit {provider} batch: {e}") from e


class BatchProvider(ABC):
    """Abstract base class for batch processing providers"""

    @abstractmethod
    def submit_batch(
        self,
        file_path_or_buffer: Union[str, io.BytesIO],
        metadata: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Submit a batch job and return the job ID"""

    @abstractmethod
    def get_status(self, batch_id: str) -> dict[str, Any]:
        """Get the status of a batch job"""

    @abstractmethod
    def retrieve_results(self, batch_id: str) -> str:
        """Retrieve batch results as a string"""

    @abstractmethod
    def download_results(self, batch_id: str, file_path: str) -> None:
        """Download batch results to a file"""

    @abstractmethod
    def cancel_batch(self, batch_id: str) -> dict[str, Any]:
        """Cancel a batch job"""

    @abstractmethod
    def delete_batch(self, batch_id: str) -> dict[str, Any]:
        """Delete a batch job"""

    @abstractmethod
    def list_batches(self, limit: int = 10) -> list[BatchJobInfo]:
        """List batch jobs"""
