"""Behavior tests for the small Anthropic v2 support modules."""

from __future__ import annotations

import base64
import builtins
import runpy
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Union, cast

import anthropic
import pytest

# These tests inject an HTTP client into the Anthropic SDK, which performs an
# `isinstance(http_client, httpx.Client)` guard. We pin the real httpx here
# so the guard passes against the pinned anthropic 0.93; the migration to
# httpx2 is handled in the production code path (`instructor.v2.auto_client`).
import httpx
from anthropic.types import Usage
from anthropic.types.cache_creation import CacheCreation
from anthropic.types.server_tool_usage import ServerToolUsage
from pydantic import BaseModel, ValidationInfo, field_validator

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.errors import ClientError, ModeError
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.providers import anthropic as anthropic_provider
from instructor.v2.providers.anthropic import client as anthropic_client
from instructor.v2.providers.anthropic import multimodal, templating
from instructor.v2.providers.anthropic.parallel import (
    AnthropicParallelModel,
    handle_parallel_model,
)
from instructor.v2.providers.anthropic.usage import initialize_usage, update_total_usage


def test_optional_anthropic_imports_fail_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__
    client_path = Path(anthropic_client.__file__)
    provider_path = Path(anthropic_provider.__file__)

    with monkeypatch.context() as context:
        context.setattr(
            builtins,
            "__import__",
            lambda name, *args, **kwargs: (
                (_ for _ in ()).throw(ImportError("anthropic is unavailable"))
                if name == "anthropic"
                else real_import(name, *args, **kwargs)
            ),
        )
        namespace = runpy.run_path(str(client_path), run_name="anthropic_missing_sdk")
        assert namespace["anthropic"] is None
        with pytest.raises(ClientError, match="anthropic is not installed"):
            namespace["from_anthropic"](object())

    with monkeypatch.context() as context:
        context.setattr(
            builtins,
            "__import__",
            lambda name, *args, **kwargs: (
                (_ for _ in ()).throw(ImportError("client is unavailable"))
                if name == "instructor.v2.providers.anthropic.client"
                else real_import(name, *args, **kwargs)
            ),
        )
        namespace = runpy.run_path(
            str(provider_path), run_name="anthropic_missing_client"
        )
        assert namespace["from_anthropic"] is None
        assert namespace["__all__"] == ["from_anthropic"]


def test_anthropic_factory_validates_mode_and_client() -> None:
    client = anthropic.Anthropic(
        api_key="test-key",
        base_url="https://anthropic.invalid",
        http_client=httpx.Client(trust_env=False),
    )

    with pytest.raises(ModeError) as error:
        anthropic_client.from_anthropic(client, mode=Mode.TOOLS_STRICT)
    assert error.value.mode == Mode.TOOLS_STRICT.value
    assert error.value.provider == Provider.ANTHROPIC.value
    assert Mode.TOOLS.value in error.value.valid_modes

    with pytest.raises(ClientError, match="Got: object") as error:
        anthropic_client.from_anthropic(
            cast(anthropic.Anthropic, object()), mode=Mode.TOOLS
        )
    assert "Anthropic" in str(error.value)
    assert "AsyncAnthropicVertex" in str(error.value)
    client.close()


@pytest.mark.asyncio
async def test_anthropic_factory_uses_beta_sync_and_regular_async_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    regular_sync = lambda **_kwargs: {"source": "sync"}
    beta_sync = lambda **_kwargs: {"source": "beta"}

    async def regular_async(**_kwargs: Any) -> dict[str, str]:
        return {"source": "async"}

    sync_client = anthropic.Anthropic(
        api_key="test-key",
        base_url="https://anthropic.invalid",
        http_client=httpx.Client(trust_env=False),
    )
    raw_async = anthropic.AsyncAnthropic(
        api_key="test-key",
        base_url="https://anthropic.invalid",
        http_client=httpx.AsyncClient(trust_env=False),
    )
    monkeypatch.setattr(sync_client.messages, "create", regular_sync)
    monkeypatch.setattr(sync_client.beta.messages, "create", beta_sync)
    monkeypatch.setattr(raw_async.messages, "create", regular_async)
    patch_calls: list[dict[str, Any]] = []

    def fake_patch(**kwargs: Any) -> Any:
        patch_calls.append(kwargs)
        return kwargs["func"]

    monkeypatch.setattr(anthropic_client, "patch_v2", fake_patch)
    with pytest.warns(DeprecationWarning, match="ANTHROPIC_JSON is deprecated"):
        sync = anthropic_client.from_anthropic(
            sync_client,
            mode=Mode.ANTHROPIC_JSON,
            beta=True,
            model="claude-sync",
            temperature=0,
        )
    async_client = anthropic_client.from_anthropic(
        raw_async, mode=Mode.TOOLS, model="claude-async", max_tokens=512
    )

    assert isinstance(sync, Instructor)
    assert not isinstance(sync, AsyncInstructor)
    assert sync.provider is Provider.ANTHROPIC
    assert sync.mode is Mode.JSON
    assert sync.create_fn is beta_sync
    assert sync.kwargs == {"temperature": 0}
    assert isinstance(async_client, AsyncInstructor)
    assert async_client.mode is Mode.TOOLS
    assert async_client.create_fn is regular_async
    assert async_client.kwargs == {"max_tokens": 512}
    assert patch_calls == [
        {
            "func": beta_sync,
            "provider": Provider.ANTHROPIC,
            "mode": Mode.JSON,
            "default_model": "claude-sync",
        },
        {
            "func": regular_async,
            "provider": Provider.ANTHROPIC,
            "mode": Mode.TOOLS,
            "default_model": "claude-async",
        },
    ]
    sync_client.close()
    await raw_async.close()


def test_anthropic_multimodal_encodes_remote_image_and_local_pdf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_calls: list[str] = []
    image = SimpleNamespace(
        source="https://example.test/diagram.png",
        media_type="image/png",
        data=None,
        url_to_base64=lambda url: image_calls.append(url) or "aW1hZ2U=",
    )
    assert multimodal.image_to_anthropic(image) == {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "aW1hZ2U=",
        },
    }
    assert image_calls == ["https://example.test/diagram.png"]
    assert image.data == "aW1hZ2U="

    requests: list[tuple[str, int]] = []

    def fake_get(url: str, *, timeout: int) -> Any:
        requests.append((url, timeout))
        return SimpleNamespace(content=b"%PDF-1.7\nexample")

    monkeypatch.setattr(multimodal.requests, "get", fake_get)
    pdf = SimpleNamespace(
        source=Path("/tmp/example.pdf"), media_type="application/pdf", data=None
    )
    expected = base64.b64encode(b"%PDF-1.7\nexample").decode()
    assert multimodal.pdf_to_anthropic(pdf) == {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": expected,
        },
    }
    assert requests == [("/tmp/example.pdf", 30)]
    assert pdf.data == expected


def test_anthropic_multimodal_handles_pdf_urls_cache_control_and_audio() -> None:
    remote_pdf = SimpleNamespace(
        source="https://example.test/handbook.pdf",
        media_type="application/pdf",
        data=None,
    )
    assert multimodal.pdf_to_anthropic(remote_pdf) == {
        "type": "document",
        "source": {"type": "url", "url": "https://example.test/handbook.pdf"},
    }

    image = SimpleNamespace(
        source="data:image/png;base64,aW1hZ2U=",
        media_type="image/png",
        data="aW1hZ2U=",
        cache_control={"type": "ephemeral", "ttl": "5m"},
    )
    assert multimodal.image_with_cache_control_to_anthropic(image) == {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "aW1hZ2U=",
        },
        "cache_control": {"type": "ephemeral", "ttl": "5m"},
    }
    image.cache_control = None
    assert "cache_control" not in multimodal.image_with_cache_control_to_anthropic(
        image
    )

    cached_pdf = SimpleNamespace(
        source="data:application/pdf;base64,cGRm",
        media_type="application/pdf",
        data="cGRm",
    )
    assert multimodal.pdf_with_cache_control_to_anthropic(cached_pdf) == {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": "cGRm",
        },
        "cache_control": {"type": "ephemeral"},
    }
    with pytest.raises(NotImplementedError, match="Anthropic is not supported yet"):
        multimodal.audio_to_anthropic(SimpleNamespace(data="YXVkaW8="))


class Contact(BaseModel):
    """A person mentioned in the message."""

    name: str
    score: int

    @field_validator("score", mode="after")
    @classmethod
    def add_context_bonus(cls, value: int, info: ValidationInfo) -> int:
        return value + info.context.get("bonus", 0) if info.context else value


class Reminder(BaseModel):
    """A reminder requested by the user."""

    text: str


def test_anthropic_parallel_schema_and_response_filtering() -> None:
    typehint = Iterable[Union[Contact, Reminder]]
    schemas = handle_parallel_model(typehint)

    assert [schema["name"] for schema in schemas] == ["Contact", "Reminder"]
    assert schemas[0]["description"] == "A person mentioned in the message."
    assert schemas[0]["input_schema"]["required"] == ["name", "score"]
    assert schemas[1]["input_schema"]["required"] == ["text"]

    parser = AnthropicParallelModel(typehint)
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="I found two items."),
            SimpleNamespace(
                type="tool_use", name="Contact", input={"name": "Ada", "score": 4}
            ),
            SimpleNamespace(type="tool_use", name="Unknown", input={"ignored": True}),
            SimpleNamespace(
                type="tool_use", name="Reminder", input={"text": "Send notes"}
            ),
            SimpleNamespace(text="no block type"),
        ]
    )

    assert list(
        parser.from_response(
            response, Mode.PARALLEL_TOOLS, validation_context={"bonus": 3}, strict=True
        )
    ) == [Contact(name="Ada", score=7), Reminder(text="Send notes")]
    assert list(parser.from_response(None, Mode.PARALLEL_TOOLS)) == []
    assert list(parser.from_response(object(), Mode.PARALLEL_TOOLS)) == []


def test_anthropic_templating_renders_only_text_blocks() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def apply_template(value: str, context: dict[str, Any]) -> str:
        calls.append((value, context))
        return value.replace("{{ name }}", context["name"])

    context = {"name": "Ada"}
    message: dict[str, Any] = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Hello {{ name }}"},
            {"type": "image", "source": {"type": "url", "url": "image.invalid"}},
            {"type": "text", "text": 42},
            "plain text",
        ],
    }

    assert templating.process_message(message, context, apply_template) is message
    assert message["content"] == [
        {"type": "text", "text": "Hello Ada"},
        {"type": "image", "source": {"type": "url", "url": "image.invalid"}},
        {"type": "text", "text": 42},
        "plain text",
    ]
    assert calls == [("Hello {{ name }}", context)]
    assert templating.process_message({"role": "user"}, context, apply_template) == {
        "role": "user"
    }


def test_anthropic_usage_initializes_and_accumulates_sdk_usage() -> None:
    total = initialize_usage()
    assert isinstance(total, Usage)
    assert total.model_dump(
        include={
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        }
    ) == {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }

    total.cache_creation_input_tokens = None
    total.cache_read_input_tokens = None
    response = Usage(
        input_tokens=8,
        output_tokens=3,
        cache_creation_input_tokens=2,
        cache_read_input_tokens=5,
    )
    assert update_total_usage(response, total) is True
    assert (
        total.input_tokens,
        total.output_tokens,
        total.cache_creation_input_tokens,
        total.cache_read_input_tokens,
    ) == (8, 3, 2, 5)
    assert response.model_dump(
        include={
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        }
    ) == {
        "input_tokens": 8,
        "output_tokens": 3,
        "cache_creation_input_tokens": 2,
        "cache_read_input_tokens": 5,
    }

    second = Usage(
        input_tokens=4,
        output_tokens=7,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
    )
    assert update_total_usage(second, total) is True
    assert (
        second.input_tokens,
        second.output_tokens,
        second.cache_creation_input_tokens,
        second.cache_read_input_tokens,
    ) == (12, 10, 2, 5)
    assert update_total_usage(object(), total) is False
    assert update_total_usage(second, object()) is False


def test_anthropic_usage_accumulates_nested_provider_fields() -> None:
    total = initialize_usage()
    first = Usage(
        input_tokens=100,
        output_tokens=50,
        cache_creation=CacheCreation(
            ephemeral_1h_input_tokens=10, ephemeral_5m_input_tokens=20
        ),
        server_tool_use=ServerToolUsage(web_fetch_requests=2, web_search_requests=3),
    )
    second = Usage(
        input_tokens=100,
        output_tokens=50,
        cache_creation=CacheCreation(
            ephemeral_1h_input_tokens=10, ephemeral_5m_input_tokens=20
        ),
        server_tool_use=ServerToolUsage(web_fetch_requests=2, web_search_requests=3),
    )

    assert update_total_usage(first, total) is True
    assert update_total_usage(second, total) is True
    assert total.cache_creation == CacheCreation(
        ephemeral_1h_input_tokens=20, ephemeral_5m_input_tokens=40
    )
    assert total.server_tool_use == ServerToolUsage(
        web_fetch_requests=4, web_search_requests=6
    )
    assert second.cache_creation == total.cache_creation
    assert second.server_tool_use == total.server_tool_use
