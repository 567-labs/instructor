"""JSON extraction helpers owned by the v2 runtime."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Generator, Iterable


MAX_JSON_EXTRACTION_CHARS = 1024 * 1024
MAX_JSON_DEPTH = 128


def extract_json_from_codeblock(content: str) -> str:
    """Extract the last JSON object- or array-like span from a text block.

    Returns the LAST complete JSON object, not the first. The LLM's own
    structured output is the authoritative JSON and appears last; JSON that
    appeared earlier may have originated from user input embedded in the
    prompt and was referenced in the model's reasoning. Returning the first
    object allowed prompt-injection to hijack the parsed output.
    """
    return _last_json_span(content) or content


def _last_json_span(content: str) -> str | None:
    if len(content) > MAX_JSON_EXTRACTION_CHARS:
        raise ValueError("JSON extraction input exceeds the 1 MiB character limit")
    decoder = json.JSONDecoder()
    last_valid: str | None = None
    consumed = 0
    # A failed decode may inspect the entire remaining suffix (including an
    # unterminated string). Charge that upper bound, not the error position.
    # Successful spans are disjoint. This bounds total decoding work linearly
    # while retaining recovery after malformed brackets or quoted prose.
    remaining_work = 16 * len(content)
    for index, char in enumerate(content):
        if index < consumed or char not in "{[":
            continue
        if remaining_work < len(content) - index:
            raise ValueError("JSON extraction malformed-input work limit exceeded")
        try:
            value, end = decoder.raw_decode(content, index)
        except json.JSONDecodeError:
            remaining_work -= len(content) - index
            continue
        except RecursionError as exc:
            raise ValueError("JSON extraction nesting limit exceeded") from exc
        pending = [(value, 1)]
        while pending:
            node, depth = pending.pop()
            if isinstance(node, (dict, list)):
                if depth > MAX_JSON_DEPTH:
                    raise ValueError(
                        "JSON extraction nesting exceeds the 128 level limit"
                    )
                children = node.values() if isinstance(node, dict) else node
                pending.extend((child, depth + 1) for child in children)
        remaining_work -= end - index
        last_valid = content[index:end]
        consumed = end
    return last_valid


class _SingleModelStream:
    """Buffer at most one bounded response so the final JSON span wins."""

    def __init__(self) -> None:
        self.chunks: list[str] = []
        self.size = 0

    def append(self, chunk: str) -> None:
        self.size += len(chunk)
        if self.size > MAX_JSON_EXTRACTION_CHARS:
            raise ValueError("JSON extraction input exceeds the 1 MiB character limit")
        if chunk:
            self.chunks.append(chunk)

    def result(self) -> Generator[str, None, None]:
        content = "".join(self.chunks)
        span = _last_json_span(content)
        if span is None:
            # Retain incomplete input so a truncated stream can produce a partial model.
            yield from extract_json_from_stream(self.chunks)
        elif span == content.strip():
            # Preserve partial snapshots at the original chunk boundaries.
            yield from self.chunks
        else:
            yield span


class _MultiModelStream:
    """Separate fenced output from prose while scanning each input character once."""

    def __init__(self, prefer_fenced: bool) -> None:
        self.prefer_fenced = prefer_fenced
        self.in_fence = False
        self.saw_fence = False
        self.ticks = 0
        self.quoted = False
        self.escaped = False
        self.stack: list[str] = []
        self.buffer: list[str] = []
        self.deferred: list[str] = []
        self.last_invalid: str | None = None
        self.has_valid = False
        self.size = 0

    def feed(self, chunk: str) -> Generator[str, None, None]:
        self.size += len(chunk)
        if self.size > MAX_JSON_EXTRACTION_CHARS:
            raise ValueError("JSON extraction input exceeds the 1 MiB character limit")
        for char in chunk:
            if not self.quoted and char == "`":
                self.ticks += 1
                if self.ticks == 3:
                    self.ticks = 0
                    if self.in_fence and self.buffer:
                        yield from self.buffer
                    if not self.in_fence and not self.saw_fence and self.prefer_fenced:
                        self.deferred.clear()
                        self.last_invalid = None
                        self.has_valid = False
                    self.in_fence = not self.in_fence
                    self.saw_fence = True
                    self.stack.clear()
                    self.buffer.clear()
                continue
            self.ticks = 0
            if self.prefer_fenced and self.saw_fence and not self.in_fence:
                continue
            if not self.stack and char not in "{[":
                continue

            self.buffer.append(char)
            if self.escaped:
                self.escaped = False
            elif char == "\\" and self.quoted:
                self.escaped = True
            elif char == '"':
                self.quoted = not self.quoted
            elif not self.quoted:
                if char in "{[":
                    self.stack.append("}" if char == "{" else "]")
                    if len(self.stack) > MAX_JSON_DEPTH:
                        raise ValueError(
                            "JSON extraction nesting exceeds the 128 level limit"
                        )
                elif char == self.stack[-1]:
                    self.stack.pop()
                    if not self.stack:
                        candidate = "".join(self.buffer)
                        self.buffer.clear()
                        try:
                            json.loads(candidate)
                        except ValueError:
                            self.last_invalid = candidate
                            continue
                        self.has_valid = True
                        self.last_invalid = None
                        if self.in_fence or not self.prefer_fenced:
                            yield from candidate
                        else:
                            self.deferred.append(candidate)

    def finish(self) -> Generator[str, None, None]:
        for candidate in self.deferred:
            yield from candidate
        if self.buffer:
            yield from self.buffer
        elif not self.has_valid and self.last_invalid is not None:
            yield from self.last_invalid


def extract_json_from_stream(
    chunks: Iterable[str], *, single_model: bool = False, prefer_fenced: bool = True
) -> Generator[str, None, None]:
    """Prefer fenced Markdown output; set prefer_fenced=False for eager bare JSON."""
    if single_model:
        selection = _SingleModelStream()
        for chunk in chunks:
            selection.append(chunk)
        yield from selection.result()
        return
    selection_multi = _MultiModelStream(prefer_fenced)
    for chunk in chunks:
        yield from selection_multi.feed(chunk)
    yield from selection_multi.finish()


async def extract_json_from_stream_async(
    chunks: AsyncGenerator[str, None],
    *,
    single_model: bool = False,
    prefer_fenced: bool = True,
) -> AsyncGenerator[str, None]:
    """Asynchronous counterpart of extract_json_from_stream."""
    if single_model:
        selection = _SingleModelStream()
        async for chunk in chunks:
            selection.append(chunk)
        for part in selection.result():
            yield part
        return
    selection_multi = _MultiModelStream(prefer_fenced)
    async for chunk in chunks:
        for part in selection_multi.feed(chunk):
            yield part
    for part in selection_multi.finish():
        yield part
