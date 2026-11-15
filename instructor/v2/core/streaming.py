"""Request-local stream routing with legacy direct-handler compatibility."""

from __future__ import annotations

import inspect
from collections.abc import AsyncGenerator, AsyncIterable, Generator, Iterable
from typing import Any, Callable, Union
from weakref import WeakKeyDictionary

from pydantic import BaseModel

from instructor.v2.core.errors import IncompleteOutputException
from instructor.v2.core.json import (
    extract_json_from_stream,
    extract_json_from_stream_async,
)
from instructor.v2.dsl.iterable import IterableBase
from instructor.v2.dsl.parallel import ParallelBase
from instructor.v2.dsl.partial import PartialBase

_MISSING = object()

ChunkFeed = Callable[[Any], Union[str, Iterable[str], None]]


def _stepper(
    feed: ChunkFeed, truncated: Callable[[Any], bool] | None
) -> tuple[Callable[[Any], list[str]], list[Any]]:
    last: list[Any] = []

    def step(chunk: Any) -> list[str]:
        try:
            out = feed(chunk)
            fragments = [out] if out is None or isinstance(out, str) else list(out)
        except (AttributeError, IndexError):
            fragments = []
        if truncated is not None and truncated(chunk):
            last[:] = [chunk]
        return [fragment for fragment in fragments if fragment]

    return step, last


def drain_stream(
    chunks: Iterable[Any],
    feed: ChunkFeed,
    *,
    truncated: Callable[[Any], bool] | None = None,
    error: Callable[[Any], Exception] = lambda c: IncompleteOutputException(
        last_completion=c
    ),
    finish: Callable[[], Iterable[str]] | None = None,
    extract_json: bool = False,
    single_model: bool = False,
) -> Generator[str, None, None]:
    """Yield `feed(chunk)` fragments; raise truncation only after they flush."""
    step, last = _stepper(feed, truncated)

    def raw() -> Generator[str, None, None]:
        for chunk in chunks:
            yield from step(chunk)
        yield from finish() if finish is not None else ()

    stream = raw()
    if extract_json:
        stream = extract_json_from_stream(stream, single_model=single_model)
    yield from stream
    if last:
        raise error(last[0])


async def adrain_stream(
    chunks: AsyncIterable[Any],
    feed: ChunkFeed,
    *,
    truncated: Callable[[Any], bool] | None = None,
    error: Callable[[Any], Exception] = lambda c: IncompleteOutputException(
        last_completion=c
    ),
    finish: Callable[[], Iterable[str]] | None = None,
    extract_json: bool = False,
    single_model: bool = False,
) -> AsyncGenerator[str, None]:
    """Async twin of `drain_stream`."""
    step, last = _stepper(feed, truncated)

    async def raw() -> AsyncGenerator[str, None]:
        async for chunk in chunks:
            for fragment in step(chunk):
                yield fragment
        for fragment in finish() if finish is not None else ():
            yield fragment

    stream = raw()
    if extract_json:
        stream = extract_json_from_stream_async(stream, single_model=single_model)
    async for fragment in stream:
        yield fragment
    if last:
        raise error(last[0])


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
        # Single atomic pop avoids a check-then-delete race when two calls
        # consume the same marker concurrently.
        return (
            inspect.isclass(response_model)
            and self._streaming_models.pop(response_model, _MISSING) is not _MISSING
        )

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
