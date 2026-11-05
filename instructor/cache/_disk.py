"""Inline data codec for Instructor's optional SQLite response cache."""

from __future__ import annotations

import json
from typing import Any

from diskcache import Disk
from diskcache.core import MODE_RAW, UNKNOWN

_JSON_MODE = -1


class UnsafeCacheEntry(ValueError):
    """A legacy or malformed record cannot be decoded as inline data."""


class InlineDisk(Disk):
    """Keep cache values in SQLite and treat database filenames as untrusted."""

    def put(self, key: str) -> tuple[str, bool]:
        if not isinstance(key, str):
            raise TypeError("DiskCache keys must be strings")
        return key, True

    def get(self, key: Any, raw: bool) -> Any:
        if not raw:
            raise UnsafeCacheEntry("Unsupported encoded cache key")
        return key

    def store(self, value: Any, read: bool, key: Any = UNKNOWN) -> tuple:  # noqa: ARG002
        if read:
            value = value.read()
        if isinstance(value, (str, bytes)):
            return 0, MODE_RAW, None, value
        encoded = json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        if json.loads(encoded) != value:
            raise TypeError(
                "DiskCache values must round-trip as JSON, strings, or bytes"
            )
        return 0, _JSON_MODE, None, encoded

    def fetch(self, mode: int, filename: str | None, value: Any, read: bool) -> Any:  # noqa: ARG002
        if filename is not None or mode not in (MODE_RAW, _JSON_MODE):
            raise UnsafeCacheEntry("Unsupported external or encoded cache record")
        if mode == MODE_RAW:
            return value
        try:
            return json.loads(value)
        except (TypeError, ValueError, RecursionError) as exc:
            raise UnsafeCacheEntry("Invalid cached JSON") from exc

    def remove(self, file_path: str) -> None:  # noqa: ARG002
        # Inline entries own no files. Old database rows cannot authorize deletion.
        return None
