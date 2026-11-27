"""Offline regression tests for xAI prompt templating via ``context``."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel
from xai_sdk import chat as xchat

from instructor.v2.core.mode import Mode
from instructor.v2.providers.xai import client as xai_client


class Answer(BaseModel):
    answer: float


class _SyncChat:
    def __init__(self) -> None:
        self.proto = xchat.chat_pb2.GetCompletionsRequest()

    def sample(self) -> Any:
        return SimpleNamespace(
            tool_calls=[
                SimpleNamespace(function=SimpleNamespace(arguments='{"answer": 4.0}'))
            ]
        )


class _AsyncChat(_SyncChat):
    async def sample(self) -> Any:
        return _SyncChat().sample()


class _ChatFactory:
    def __init__(self, chat: Any) -> None:
        self.chat = chat
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.chat


class _SyncClient:
    def __init__(self, factory: _ChatFactory) -> None:
        self.chat = factory


class _AsyncClient:
    def __init__(self, factory: _ChatFactory) -> None:
        self.chat = factory


def _install_stubs(monkeypatch: Any) -> _ChatFactory:
    factory = _ChatFactory(_SyncChat())
    monkeypatch.setattr(xai_client, "SyncClient", _SyncClient)
    monkeypatch.setattr(xai_client, "AsyncClient", _AsyncClient)
    return factory


def _install_async_stubs(monkeypatch: Any) -> _ChatFactory:
    factory = _ChatFactory(_AsyncChat())
    monkeypatch.setattr(xai_client, "SyncClient", _SyncClient)
    monkeypatch.setattr(xai_client, "AsyncClient", _AsyncClient)
    return factory


def _messages() -> list[dict[str, str]]:
    return [{"role": "user", "content": "What is {{ a }} + {{ b }}?"}]


def test_from_xai_renders_context_before_sending_sync(monkeypatch: Any) -> None:
    factory = _install_stubs(monkeypatch)
    wrapped = xai_client.from_xai(_SyncClient(factory), mode=Mode.TOOLS)
    messages = _messages()

    result = wrapped.create(
        response_model=Answer,
        messages=messages,
        model="grok-test",
        context={"a": 2, "b": 2},
    )

    assert result.answer == 4.0
    assert factory.calls[0]["messages"][0].content[0].text == "What is 2 + 2?"
    assert messages == _messages()


def test_from_xai_renders_context_before_sending_async(monkeypatch: Any) -> None:
    factory = _install_async_stubs(monkeypatch)
    wrapped = xai_client.from_xai(_AsyncClient(factory), mode=Mode.TOOLS)
    messages = _messages()

    result = asyncio.run(
        wrapped.create(
            response_model=Answer,
            messages=messages,
            model="grok-test",
            context={"a": 2, "b": 2},
        )
    )

    assert result.answer == 4.0
    assert factory.calls[0]["messages"][0].content[0].text == "What is 2 + 2?"
    assert messages == _messages()
