from collections.abc import AsyncGenerator, Callable, Generator, Iterable
from typing import Any, Generic, TypeVar, overload

from pydantic import BaseModel
from pydantic.fields import FieldInfo

T_Model = TypeVar("T_Model", bound=BaseModel)

class MakeFieldsOptional: ...

class PartialBase(BaseModel, Generic[T_Model]):
    @staticmethod
    def extract_json(
        completion: Iterable[Any],
        stream_extractor: Callable[[Iterable[Any]], Generator[str, None, None]] | Any,
        on_event: Callable[..., Any] | None = None,
    ) -> Generator[str, None, None]: ...
    @staticmethod
    def extract_json_async(
        completion: AsyncGenerator[Any, None],
        stream_extractor: Callable[
            [AsyncGenerator[Any, None]], AsyncGenerator[str, None]
        ]
        | Any,
        on_event: Callable[..., Any] | None = None,
    ) -> AsyncGenerator[str, None]: ...

class PartialLiteralMixin: ...

def _make_field_optional(field: FieldInfo) -> tuple[Any, FieldInfo]: ...

class _PartialFactory:
    @overload
    def __getitem__(self, wrapped_class: type[T_Model]) -> type[T_Model]: ...
    @overload
    def __getitem__(
        self,
        wrapped_class: tuple[type[T_Model], type[MakeFieldsOptional]],
    ) -> type[T_Model]: ...

Partial: _PartialFactory
