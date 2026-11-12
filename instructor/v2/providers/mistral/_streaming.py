"""Buffer complete Mistral task-list arguments across streaming events."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterator, Generator, Iterable
from typing import Any

from instructor.v2.dsl.iterable import IterableBase, _TaskObjects


class _TaskListBuffer:
    def __init__(self) -> None:
        self.scanner = _TaskObjects(unwrap_tasks=False)
        self.saw_input = False
        self.saw_object = False

    def feed(self, chunk: str | None) -> str | None:
        # Preserve the complete-chunk parser's treatment of empty events.
        if not chunk:
            return None
        if not isinstance(chunk, str):
            raise TypeError("Mistral streaming tool arguments must be strings")
        self.saw_input = True
        if self.scanner.depth == 0:
            prefix = chunk.lstrip(" \t\r\n")
            if prefix and not prefix.startswith("{"):
                raise json.JSONDecodeError(
                    "Expected a JSON object", chunk, len(chunk) - len(prefix)
                )
            if prefix:
                # Complete events keep the fast C decoder path. An unfinished
                # object enters the scanner below, so later fragments are not
                # repeatedly decoded from the beginning.
                try:
                    _, end = json.JSONDecoder().raw_decode(prefix)
                except json.JSONDecodeError:
                    pass
                else:
                    if prefix[end:].strip(" \t\r\n"):
                        raise json.JSONDecodeError("Extra data", prefix, end)
                    self.saw_object = True
                    return prefix[:end]
        complete: str | None = None
        for value, end in self.scanner.feed(chunk):
            if chunk[end:].strip(" \t\r\n"):
                raise json.JSONDecodeError("Extra data", chunk, end)
            complete = value
        if complete is not None:
            self.saw_object = True
        return complete

    def finish(self) -> None:
        self.scanner.finish()
        if self.saw_input and not self.saw_object:
            raise json.JSONDecodeError("Expected a JSON object", "", 0)


def _complete_task_lists(chunks: Iterable[str]) -> Generator[str, None, None]:
    buffer = _TaskListBuffer()
    for chunk in chunks:
        complete = buffer.feed(chunk)
        if complete is not None:
            yield complete
    buffer.finish()


async def _complete_task_lists_async(
    chunks: AsyncIterator[str],
) -> AsyncGenerator[str, None]:
    buffer = _TaskListBuffer()
    async for chunk in chunks:
        complete = buffer.feed(chunk)
        if complete is not None:
            yield complete
    buffer.finish()


def tasks_from_mistral_chunks(
    response_model: type[IterableBase], chunks: Iterable[str], **kwargs: Any
) -> Generator[Any, None, None]:
    yield from response_model.tasks_from_task_list_chunks(
        _complete_task_lists(chunks), **kwargs
    )


async def tasks_from_mistral_chunks_async(
    response_model: type[IterableBase], chunks: AsyncIterator[str], **kwargs: Any
) -> AsyncGenerator[Any, None]:
    async for item in response_model.tasks_from_task_list_chunks_async(
        _complete_task_lists_async(chunks), **kwargs
    ):
        yield item
