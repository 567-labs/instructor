"""List-like response wrapper.

When a response model returns a list (for example `list[User]`), we still want to
attach the provider's raw response so `create_with_completion()` can return it.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class ListResponse(list[T], Generic[T]):
    """A list that preserves the underlying provider response.

    This is used when a call returns a list of objects (e.g. `list[User]`), so
    `create_with_completion()` can still return `(result, raw_response)` without
    crashing on a plain `list`.
    """

    _raw_response: Any | None
    _total_usage: Any | None

    def __init__(
        self,
        iterable: Iterable[T] = (),
        _raw_response: Any | None = None,
        _total_usage: Any | None = None,
    ) -> None:
        super().__init__(iterable)
        self._raw_response = _raw_response
        self._total_usage = _total_usage

    @classmethod
    def from_list(
        cls,
        items: list[T],
        *,
        raw_response: Any | None,
        total_usage: Any | None = None,
    ) -> ListResponse[T]:
        return cls(
            items,
            _raw_response=raw_response,
            _total_usage=total_usage,
        )

    def get_raw_response(self) -> Any | None:
        return self._raw_response

    def get_total_usage(self) -> Any | None:
        return self._total_usage

    def __getitem__(self, key):  # type: ignore[no-untyped-def]
        value = super().__getitem__(key)
        if isinstance(key, slice):
            return type(self)(
                value,
                _raw_response=self._raw_response,
                _total_usage=self._total_usage,
            )
        return value


# Backwards-friendly alias
ResponseList = ListResponse
