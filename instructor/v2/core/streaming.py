"""Request-local stream routing with legacy direct-handler compatibility."""

from __future__ import annotations

import inspect
from typing import Any
from weakref import WeakKeyDictionary

from pydantic import BaseModel

from instructor.v2.dsl.iterable import IterableBase
from instructor.v2.dsl.parallel import ParallelBase
from instructor.v2.dsl.partial import PartialBase


class StreamingModelState:
    """Share routing policy, without sharing provider event parsing.

    Core callers pass an explicit request-local boolean. The class-keyed marker
    exists only for legacy sequential prepare_request/parse_response calls that
    omit stream. Such calls cannot disambiguate overlapping identical models.
    """

    def __init__(self) -> None:
        self._streaming_models: WeakKeyDictionary[type[Any], None] = WeakKeyDictionary()

    def _register_streaming_from_kwargs(
        self, response_model: type[BaseModel] | None, kwargs: dict[str, Any]
    ) -> None:
        self.mark_streaming_model(response_model, bool(kwargs.get("stream")))

    def mark_streaming_model(
        self, response_model: type[BaseModel] | None, stream: bool
    ) -> None:
        if (
            stream
            and inspect.isclass(response_model)
            and issubclass(response_model, (IterableBase, PartialBase))
        ):
            self._streaming_models[response_model] = None

    def _consume_streaming_flag(
        self, response_model: type[BaseModel] | ParallelBase | None
    ) -> bool:
        if inspect.isclass(response_model) and response_model in self._streaming_models:
            del self._streaming_models[response_model]
            return True
        return False

    def _should_parse_streaming(
        self,
        response_model: type[BaseModel] | ParallelBase | None,
        stream: bool | None,
    ) -> bool:
        # Always retire the legacy marker, including when explicit True wins.
        registered = self._consume_streaming_flag(response_model)
        return bool(
            inspect.isclass(response_model)
            and issubclass(response_model, (IterableBase, PartialBase))
            and (registered if stream is None else stream)
        )
