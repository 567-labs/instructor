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
    return last_valid if last_valid is not None else content


def extract_json_from_stream(chunks: Iterable[str]) -> Generator[str, None, None]:
    """Extract JSON characters from a plain-text or fenced streaming response.

    Mirrors the prompt-injection posture of :func:`extract_json_from_codeblock`:
    JSON that completes outside a fenced codeblock is held back, and if a
    fence appears later the held-back prose objects are discarded so only the
    fenced object is emitted. Streams that never contain a fence flush the
    held-back objects in order at the end, preserving unfenced behavior.
    """
    in_codeblock = False
    codeblock_delimiter_count = 0
    json_started = False
    in_string = False
    escape_next = False
    delimiter_stack: list[str] = []
    buffer: list[str] = []
    codeblock_buffer: list[str] = []
    last_invalid_candidate: str | None = None
    emitted_valid_candidate = False
    total_chars = 0
    fence_seen = False
    deferred_prose: list[str] = []

    for chunk in chunks:
        total_chars += len(chunk)
        if total_chars > MAX_JSON_EXTRACTION_CHARS:
            raise ValueError("JSON extraction input exceeds the 1 MiB character limit")
        for char in chunk:
            if not in_codeblock and char == "`" and not (json_started and in_string):
                codeblock_buffer.append(char)
                if len(codeblock_buffer) == 3:
                    in_codeblock = True
                    fence_seen = True
                    deferred_prose.clear()
                    codeblock_delimiter_count = 0
                    codeblock_buffer = []
                continue
            if codeblock_buffer and char != "`":
                codeblock_buffer = []

            if in_codeblock and not json_started:
                if char == "`":
                    codeblock_delimiter_count += 1
                    if codeblock_delimiter_count == 3:
                        in_codeblock = False
                        codeblock_delimiter_count = 0
                    continue
                if codeblock_delimiter_count > 0:
                    codeblock_delimiter_count = 0

                if char in "{[":
                    json_started = True
                    delimiter_stack.append("}" if char == "{" else "]")
                    buffer.append(char)
                continue

            if json_started:
                if escape_next:
                    escape_next = False
                elif char == "\\" and in_string:
                    escape_next = True
                    buffer.append(char)
                    continue
                elif char == '"':
                    in_string = not in_string

                if in_codeblock and not in_string:
                    if char == "`":
                        codeblock_delimiter_count += 1
                        if codeblock_delimiter_count == 3:
                            in_codeblock = False
                            yield from buffer
                            buffer = []
                            json_started = False
                            break
                        continue
                    if codeblock_delimiter_count > 0:
                        codeblock_delimiter_count = 0

                if not in_string:
                    if char in "{[":
                        delimiter_stack.append("}" if char == "{" else "]")
                        if len(delimiter_stack) > MAX_JSON_DEPTH:
                            raise ValueError(
                                "JSON extraction nesting exceeds the 128 level limit"
                            )
                    elif delimiter_stack and char == delimiter_stack[-1]:
                        delimiter_stack.pop()
                        if not delimiter_stack:
                            buffer.append(char)
                            candidate = "".join(buffer)
                            buffer = []
                            json_started = False
                            try:
                                json.loads(candidate)
                            except ValueError:
                                last_invalid_candidate = candidate
                                continue
                            if in_codeblock:
                                emitted_valid_candidate = True
                                last_invalid_candidate = None
                                yield from candidate
                            elif fence_seen:
                                # Prose JSON after a fence: untrusted context,
                                # never emitted (matches codeblock extraction).
                                last_invalid_candidate = None
                            else:
                                # Prose object before any fence: defer so it
                                # can be dropped if a fence shows up later.
                                deferred_prose.append(candidate)
                                last_invalid_candidate = None
                            continue

                buffer.append(char)
                continue

            if not in_codeblock and not json_started and char in "{[":
                json_started = True
                delimiter_stack.append("}" if char == "{" else "]")
                buffer.append(char)

    if fence_seen:
        # Fenced responses ignore prose JSON entirely, including trailing
        # unterminated text outside the fence. Unterminated text inside an
        # open fence still streams so Partial consumers see cut-off data.
        if in_codeblock and json_started and buffer:
            yield from buffer
    elif json_started and buffer:
        for candidate in deferred_prose:
            emitted_valid_candidate = True
            yield from candidate
        yield from buffer
    else:
        for candidate in deferred_prose:
            emitted_valid_candidate = True
            yield from candidate
        if not emitted_valid_candidate and last_invalid_candidate is not None:
            yield from last_invalid_candidate


async def extract_json_from_stream_async(
    chunks: AsyncGenerator[str, None],
) -> AsyncGenerator[str, None]:
    """Async counterpart to :func:`extract_json_from_stream`."""
    in_codeblock = False
    codeblock_delimiter_count = 0
    json_started = False
    in_string = False
    escape_next = False
    delimiter_stack: list[str] = []
    buffer: list[str] = []
    codeblock_buffer: list[str] = []
    last_invalid_candidate: str | None = None
    emitted_valid_candidate = False
    total_chars = 0
    fence_seen = False
    deferred_prose: list[str] = []

    async for chunk in chunks:
        total_chars += len(chunk)
        if total_chars > MAX_JSON_EXTRACTION_CHARS:
            raise ValueError("JSON extraction input exceeds the 1 MiB character limit")
        for char in chunk:
            if not in_codeblock and char == "`" and not (json_started and in_string):
                codeblock_buffer.append(char)
                if len(codeblock_buffer) == 3:
                    in_codeblock = True
                    fence_seen = True
                    deferred_prose.clear()
                    codeblock_delimiter_count = 0
                    codeblock_buffer = []
                continue
            if codeblock_buffer and char != "`":
                codeblock_buffer = []

            if in_codeblock and not json_started:
                if char == "`":
                    codeblock_delimiter_count += 1
                    if codeblock_delimiter_count == 3:
                        in_codeblock = False
                        codeblock_delimiter_count = 0
                    continue
                if codeblock_delimiter_count > 0:
                    codeblock_delimiter_count = 0

                if char in "{[":
                    json_started = True
                    delimiter_stack.append("}" if char == "{" else "]")
                    buffer.append(char)
                continue

            if json_started:
                if escape_next:
                    escape_next = False
                elif char == "\\" and in_string:
                    escape_next = True
                    buffer.append(char)
                    continue
                elif char == '"':
                    in_string = not in_string

                if in_codeblock and not in_string:
                    if char == "`":
                        codeblock_delimiter_count += 1
                        if codeblock_delimiter_count == 3:
                            in_codeblock = False
                            for buffered_char in buffer:
                                yield buffered_char
                            buffer = []
                            json_started = False
                            break
                        continue
                    if codeblock_delimiter_count > 0:
                        codeblock_delimiter_count = 0

                if not in_string:
                    if char in "{[":
                        delimiter_stack.append("}" if char == "{" else "]")
                        if len(delimiter_stack) > MAX_JSON_DEPTH:
                            raise ValueError(
                                "JSON extraction nesting exceeds the 128 level limit"
                            )
                    elif delimiter_stack and char == delimiter_stack[-1]:
                        delimiter_stack.pop()
                        if not delimiter_stack:
                            buffer.append(char)
                            candidate = "".join(buffer)
                            buffer = []
                            json_started = False
                            try:
                                json.loads(candidate)
                            except ValueError:
                                last_invalid_candidate = candidate
                                continue
                            if in_codeblock:
                                emitted_valid_candidate = True
                                last_invalid_candidate = None
                                for buffered_char in candidate:
                                    yield buffered_char
                            elif fence_seen:
                                # Prose JSON after a fence: untrusted context,
                                # never emitted (matches codeblock extraction).
                                last_invalid_candidate = None
                            else:
                                # Prose object before any fence: defer so it
                                # can be dropped if a fence shows up later.
                                deferred_prose.append(candidate)
                                last_invalid_candidate = None
                            continue

                buffer.append(char)
                continue

            if not in_codeblock and not json_started and char in "{[":
                json_started = True
                delimiter_stack.append("}" if char == "{" else "]")
                buffer.append(char)

    if fence_seen:
        # Fenced responses ignore prose JSON entirely, including trailing
        # unterminated text outside the fence. Unterminated text inside an
        # open fence still streams so Partial consumers see cut-off data.
        if in_codeblock and json_started and buffer:
            for buffered_char in buffer:
                yield buffered_char
    elif json_started and buffer:
        for candidate in deferred_prose:
            emitted_valid_candidate = True
            for buffered_char in candidate:
                yield buffered_char
        for buffered_char in buffer:
            yield buffered_char
    else:
        for candidate in deferred_prose:
            emitted_valid_candidate = True
            for buffered_char in candidate:
                yield buffered_char
        if not emitted_valid_candidate and last_invalid_candidate is not None:
            for buffered_char in last_invalid_candidate:
                yield buffered_char
