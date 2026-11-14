"""GenAI multimodal encoding with real ``google.genai.types`` objects."""

from __future__ import annotations

import base64
from typing import Any

import pytest

from instructor.v2.core.multimodal import Image
from instructor.v2.providers.genai import multimodal

types = pytest.importorskip("google.genai.types")

PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"


def test_gcs_image_decodes_base64_data(monkeypatch: pytest.MonkeyPatch) -> None:
    # Recent google-genai releases decode base64 strings themselves, which hid
    # the bug; older supported releases stored the base64 text as raw bytes.
    # Spy on the real constructor to check instructor passes decoded bytes.
    real_from_bytes = types.Part.from_bytes
    received: list[Any] = []

    def spy(*, data: Any, mime_type: str) -> Any:
        received.append(data)
        return real_from_bytes(data=data, mime_type=mime_type)

    monkeypatch.setattr(types.Part, "from_bytes", spy)
    image = Image(
        source="gs://bucket/image.png",
        media_type="image/png",
        data=base64.b64encode(PNG).decode(),
    )

    part = image.to_genai()

    assert received == [PNG]
    assert part.inline_data.data == PNG
    assert part.inline_data.mime_type == "image/png"


def _file(state: Any) -> Any:
    return types.File(
        name="files/doc",
        uri="https://generativelanguage.googleapis.com/v1beta/files/doc",
        mime_type="application/pdf",
        state=state,
    )


class Poller:
    def __init__(self, *states: Any) -> None:
        self.states = list(states)
        self.calls: list[str] = []

    def __call__(self, *, name: str) -> Any:
        self.calls.append(name)
        return _file(self.states.pop(0))


def test_wait_for_active_file_fails_fast_on_failed_state() -> None:
    poll = Poller()

    with pytest.raises(ValueError, match="processing failed"):
        multimodal._wait_for_active_file(
            _file(types.FileState.FAILED), poll, retry_delay=0
        )

    assert poll.calls == []


def test_wait_for_active_file_raises_when_processing_turns_failed() -> None:
    poll = Poller(types.FileState.FAILED, types.FileState.ACTIVE)

    with pytest.raises(ValueError, match="processing failed"):
        multimodal._wait_for_active_file(
            _file(types.FileState.PROCESSING), poll, retry_delay=0
        )

    assert poll.calls == ["files/doc"]


def test_wait_for_active_file_returns_active_file() -> None:
    poll = Poller(types.FileState.ACTIVE)

    file = multimodal._wait_for_active_file(
        _file(types.FileState.PROCESSING), poll, retry_delay=0
    )

    assert file.state == types.FileState.ACTIVE
    assert poll.calls == ["files/doc"]
