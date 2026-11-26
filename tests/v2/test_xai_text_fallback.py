"""Regression tests for xAI text-fallback content extraction.

xAI can return structured-output text in ``content`` as a list of content
blocks rather than a plain string. The fallback path must read the ``text``
out of those blocks; stringifying the block objects produces a Python repr
that is not the JSON the model returned.

All fakes in this module are offline -- no provider calls or credentials.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from instructor import Mode, Provider
from instructor.v2.core.registry import mode_registry

FALLBACK_MODES = [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON]
CLIENT_FALLBACK_MODES = [Mode.TOOLS, Mode.MD_JSON]


class Answer(BaseModel):
    answer: float


class _Response:
    """Minimal tool-call-free response in the shape the fallback expects."""

    def __init__(self, content: Any = None, text: str | None = None) -> None:
        self.content = content
        self.text = text
        self.tool_calls = []


def _text_blocks(*chunks: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": chunk} for chunk in chunks]


def _image_block() -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": "https://example.invalid/x.png"}}


# --- handlers --------------------------------------------------------------


@pytest.mark.parametrize("mode", FALLBACK_MODES)
def test_single_text_block_parses(mode: Mode) -> None:
    """A single text block must expose its JSON, not the block's repr."""
    handlers = mode_registry.get_handlers(Provider.XAI, mode)

    result = handlers.response_parser(
        _Response(content=_text_blocks('{"answer": 5.0}')), Answer
    )

    assert isinstance(result, Answer)
    assert result.answer == 5.0


@pytest.mark.parametrize("mode", FALLBACK_MODES)
def test_split_text_blocks_join_into_one_payload(mode: Mode) -> None:
    """Multiple text blocks must be joined before JSON validation."""
    handlers = mode_registry.get_handlers(Provider.XAI, mode)

    result = handlers.response_parser(
        _Response(content=_text_blocks('{"ans', 'wer": 6.0}')), Answer
    )

    assert isinstance(result, Answer)
    assert result.answer == 6.0


@pytest.mark.parametrize("mode", FALLBACK_MODES)
def test_text_block_after_non_text_block_parses(mode: Mode) -> None:
    """Opaque blocks must be skipped, not treated as the text payload."""
    handlers = mode_registry.get_handlers(Provider.XAI, mode)

    result = handlers.response_parser(
        _Response(content=[_image_block(), *_text_blocks('{"answer": 7.0}')]), Answer
    )

    assert isinstance(result, Answer)
    assert result.answer == 7.0


@pytest.mark.parametrize("mode", FALLBACK_MODES)
def test_string_content_behavior_is_unchanged(mode: Mode) -> None:
    """Plain string content and legacy string lists keep parsing as before."""
    handlers = mode_registry.get_handlers(Provider.XAI, mode)

    results = [
        handlers.response_parser(_Response(content=shape), Answer)
        for shape in ('{"answer": 8.0}', ['{"answer": 8.0}'])
    ]

    assert all(isinstance(result, Answer) and result.answer == 8.0 for result in results)


# --- client factory --------------------------------------------------------


class _Proto:
    def __init__(self) -> None:
        self.tools: list[Any] = []
        self.tool_choice = SimpleNamespace(CopyFrom=lambda *_: None)
        self.response_format = SimpleNamespace(CopyFrom=lambda *_: None)


class _SyncChat:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.proto = _Proto()

    def sample(self) -> _Response:
        return self.response


class _AsyncChat(_SyncChat):
    async def sample(self) -> _Response:
        return self.response


class _ChatFactory:
    def __init__(self, response: _Response, chat_cls: type[_SyncChat]) -> None:
        self.response = response
        self.chat_cls = chat_cls

    def create(self, **_kwargs: Any) -> _SyncChat:
        return self.chat_cls(self.response)


class _FakeSyncClient:
    def __init__(self, response: _Response) -> None:
        self.chat = _ChatFactory(response, _SyncChat)


class _FakeAsyncClient:
    def __init__(self, response: _Response) -> None:
        self.chat = _ChatFactory(response, _AsyncChat)


@pytest.fixture
def fake_xai_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[type[_FakeSyncClient], type[_FakeAsyncClient]]:
    """Install offline xAI SDK stand-ins for the client factory."""
    from instructor.v2.providers.xai import client as xai_client

    xchat = SimpleNamespace(
        text=lambda value: ("text", value),
        user=lambda value: ("user", value),
        system=lambda value: ("system", value),
        tool=lambda **kwargs: SimpleNamespace(
            function=SimpleNamespace(name=kwargs["name"])
        ),
        required_tool=lambda name: name,
    )
    monkeypatch.setattr(xai_client, "xchat", xchat)
    monkeypatch.setattr(xai_client, "SyncClient", _FakeSyncClient)
    monkeypatch.setattr(xai_client, "AsyncClient", _FakeAsyncClient)
    return _FakeSyncClient, _FakeAsyncClient


@pytest.mark.parametrize("mode", CLIENT_FALLBACK_MODES)
def test_sync_client_fallback_joins_text_blocks(
    mode: Mode,
    fake_xai_clients: tuple[type[_FakeSyncClient], type[_FakeAsyncClient]],
) -> None:
    from instructor.v2.providers.xai.client import from_xai

    sync_client_cls, _ = fake_xai_clients
    response = _Response(content=_text_blocks('{"ans', 'wer": 9.0}'))
    instructor_client = from_xai(sync_client_cls(response), mode=mode)

    result = instructor_client.create_fn(
        Answer, [{"role": "user", "content": "hi"}], model="fake-model"
    )

    assert isinstance(result, Answer)
    assert result.answer == 9.0


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", CLIENT_FALLBACK_MODES)
async def test_async_client_fallback_joins_text_blocks(
    mode: Mode,
    fake_xai_clients: tuple[type[_FakeSyncClient], type[_FakeAsyncClient]],
) -> None:
    from instructor.v2.providers.xai.client import from_xai

    _, async_client_cls = fake_xai_clients
    response = _Response(content=_text_blocks('{"ans', 'wer": 10.0}'))
    instructor_client = from_xai(async_client_cls(response), mode=mode)

    result = await instructor_client.create_fn(
        Answer, [{"role": "user", "content": "hi"}], model="fake-model"
    )

    assert isinstance(result, Answer)
    assert result.answer == 10.0
