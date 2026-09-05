from __future__ import annotations

import builtins
import json
import runpy
from pathlib import Path
from typing import Any, cast

# These tests inject an HTTP client into the OpenAI SDK, which performs an
# `isinstance(http_client, httpx.Client)` guard. We deliberately pin the real
# httpx here so the guard passes against `openai<3`; the migration to httpx2
# is handled in the production code path (`instructor.v2.auto_client`).
import httpx
import openai
import pytest
from openai.types.responses import Response
from pydantic import BaseModel, Field

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.errors import ClientError, ModeError
from instructor.v2.core.mode import Mode
from instructor.v2.core.multimodal import Audio, Image, PDF
from instructor.v2.core.providers import Provider
from instructor.v2.core.remote import RemoteContent
from instructor.v2.providers.openai.client import (
    async_map_chat_completion_to_response,
    from_openai,
    map_chat_completion_to_response,
)
from instructor.v2.providers.openai.multimodal import (
    audio_to_openai,
    image_to_openai,
    pdf_to_openai,
)
from instructor.v2.providers.openai import multimodal as openai_multimodal
from instructor.v2.providers.openai.schema import generate_openai_schema
from instructor.v2.providers.openai.templating import process_message
from instructor.v2.providers.openrouter.client import from_openrouter
from instructor.v2.providers.perplexity.client import from_perplexity


def _response_payload(model: str) -> dict[str, Any]:
    return {
        "id": "resp_coverage",
        "object": "response",
        "created_at": 1,
        "model": model,
        "output": [],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }


def test_openai_sync_response_mapping_uses_real_sdk_types() -> None:
    seen: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/responses")
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json=_response_payload(body["model"]))

    http_client = httpx.Client(transport=httpx.MockTransport(handle), trust_env=False)
    client = openai.OpenAI(
        api_key="test-key",
        base_url="https://openai.invalid/v1",
        http_client=http_client,
    )
    messages = [{"role": "user", "content": "hello from sync"}]

    response = map_chat_completion_to_response(
        messages, client, model="gpt-test-sync", metadata={"lane": "sync"}
    )

    assert isinstance(response, Response)
    assert response.id == "resp_coverage"
    assert seen == [
        {
            "input": messages,
            "model": "gpt-test-sync",
            "metadata": {"lane": "sync"},
        }
    ]
    client.close()


@pytest.mark.asyncio
async def test_openai_async_response_mapping_uses_real_sdk_types() -> None:
    seen: list[dict[str, Any]] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/responses")
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json=_response_payload(body["model"]))

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handle), trust_env=False
    )
    client = openai.AsyncOpenAI(
        api_key="test-key",
        base_url="https://openai.invalid/v1",
        http_client=http_client,
    )
    messages = [{"role": "user", "content": "hello from async"}]

    response = await async_map_chat_completion_to_response(
        messages, client, model="gpt-test-async", metadata={"lane": "async"}
    )

    assert isinstance(response, Response)
    assert response.model == "gpt-test-async"
    assert seen == [
        {
            "input": messages,
            "model": "gpt-test-async",
            "metadata": {"lane": "async"},
        }
    ]
    await client.close()


def test_openai_factory_rejects_an_unregistered_mode() -> None:
    client = openai.OpenAI(
        api_key="test-key",
        base_url="https://openai.invalid/v1",
        http_client=httpx.Client(trust_env=False),
    )

    with pytest.raises(ModeError) as error:
        from_openai(client, mode=Mode.GEMINI_TOOLS)

    assert error.value.mode == Mode.GEMINI_TOOLS.value
    assert error.value.provider == Provider.OPENAI.value
    assert Mode.TOOLS.value in error.value.valid_modes
    client.close()


def test_openai_factory_rejects_an_invalid_client() -> None:
    with pytest.raises(ClientError, match="Got: object") as error:
        from_openai(cast(openai.OpenAI, object()), mode=Mode.TOOLS)

    assert "OpenAI, AsyncOpenAI" in str(error.value)


@pytest.mark.asyncio
async def test_openai_compatible_wrappers_keep_provider_mode_and_client() -> None:
    sync_client = openai.OpenAI(
        api_key="test-key",
        base_url="https://openrouter.invalid/api/v1",
        http_client=httpx.Client(trust_env=False),
    )
    async_client = openai.AsyncOpenAI(
        api_key="test-key",
        base_url="https://perplexity.invalid",
        http_client=httpx.AsyncClient(trust_env=False),
    )

    openrouter = from_openrouter(
        sync_client, mode=Mode.JSON_SCHEMA, model="router-model"
    )
    perplexity = from_perplexity(async_client, model="perplexity-model")

    assert isinstance(openrouter, Instructor)
    assert openrouter.client is sync_client
    assert openrouter.provider is Provider.OPENROUTER
    assert openrouter.mode is Mode.JSON_SCHEMA
    assert isinstance(perplexity, AsyncInstructor)
    assert perplexity.client is async_client
    assert perplexity.provider is Provider.PERPLEXITY
    assert perplexity.mode is Mode.MD_JSON
    sync_client.close()
    await async_client.close()


@pytest.mark.asyncio
async def test_openai_response_wrappers_normalize_the_inbuilt_tools_mode() -> None:
    sync_client = openai.OpenAI(
        api_key="test-key",
        base_url="https://openai.invalid/v1",
        http_client=httpx.Client(trust_env=False),
    )
    async_client = openai.AsyncOpenAI(
        api_key="test-key",
        base_url="https://openai.invalid/v1",
        http_client=httpx.AsyncClient(trust_env=False),
    )

    sync_instructor = from_openai(
        sync_client, mode=Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS
    )
    async_instructor = from_openai(
        async_client, mode=Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS
    )

    assert isinstance(sync_instructor, Instructor)
    assert sync_instructor.client is sync_client
    assert sync_instructor.mode is Mode.RESPONSES_TOOLS
    assert isinstance(async_instructor, AsyncInstructor)
    assert async_instructor.client is async_client
    assert async_instructor.mode is Mode.RESPONSES_TOOLS
    sync_client.close()
    await async_client.close()


def test_openai_multimodal_encoders_cover_response_and_error_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_url = Image(
        source="https://cdn.example.invalid/image.png", media_type="image/png"
    )
    assert image_to_openai(image_url, Mode.RESPONSES_TOOLS) == {
        "type": "input_image",
        "image_url": "https://cdn.example.invalid/image.png",
    }
    assert image_to_openai(image_url, Mode.TOOLS) == {
        "type": "image_url",
        "image_url": {"url": "https://cdn.example.invalid/image.png"},
    }

    image_data = Image.from_base64("data:image/png;base64,aW1hZ2U=")
    assert image_to_openai(image_data, Mode.RESPONSES_TOOLS) == {
        "type": "input_image",
        "image_url": "data:image/png;base64,aW1hZ2U=",
    }
    assert image_to_openai(image_data, Mode.TOOLS) == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
    }

    with pytest.raises(ValueError, match="Image data is missing"):
        image_to_openai(
            Image(source="missing-image", media_type="image/png"), Mode.TOOLS
        )

    audio = Audio(source="clip.wav", media_type="audio/wav", data="YXVkaW8=")
    assert audio_to_openai(audio, Mode.TOOLS) == {
        "type": "input_audio",
        "input_audio": {"data": "YXVkaW8=", "format": "wav"},
    }
    with pytest.raises(ValueError, match="Responses doesn't support audio"):
        audio_to_openai(audio, Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS)

    requested: list[tuple[str, int]] = []

    def fetch(url: str, *, max_bytes: int, timeout: int) -> RemoteContent:
        requested.append((url, timeout))
        assert max_bytes > 0
        return RemoteContent(
            content=b"%PDF-1.7\ncoverage",
            content_type="application/pdf",
            url=url,
        )

    monkeypatch.setattr(openai_multimodal, "fetch_remote_content", fetch)
    pdf_url = PDF(source="https://cdn.example.invalid/report.pdf")
    assert pdf_to_openai(pdf_url, Mode.RESPONSES_TOOLS) == {
        "type": "input_file",
        "filename": "https://cdn.example.invalid/report.pdf",
        "file_data": "data:application/pdf;base64,JVBERi0xLjcKY292ZXJhZ2U=",
    }
    assert pdf_to_openai(pdf_url, Mode.TOOLS) == {
        "type": "file",
        "file": {
            "filename": "https://cdn.example.invalid/report.pdf",
            "file_data": "data:application/pdf;base64,JVBERi0xLjcKY292ZXJhZ2U=",
        },
    }
    assert requested == [
        ("https://cdn.example.invalid/report.pdf", 30),
        ("https://cdn.example.invalid/report.pdf", 30),
    ]

    pdf_data = PDF.from_base64("data:application/pdf;base64,cGRm")
    assert pdf_to_openai(pdf_data, Mode.RESPONSES_TOOLS) == {
        "type": "input_file",
        "filename": "data:application/pdf;base64,cGRm",
        "file_data": "data:application/pdf;base64,cGRm",
    }
    assert pdf_to_openai(pdf_data, Mode.TOOLS) == {
        "type": "file",
        "file": {
            "filename": "data:application/pdf;base64,cGRm",
            "file_data": "data:application/pdf;base64,cGRm",
        },
    }

    with pytest.raises(ValueError, match="PDF data is missing"):
        pdf_to_openai(PDF(source="missing-pdf"), Mode.TOOLS)


def test_openai_schema_merges_only_missing_parameter_descriptions() -> None:
    class Contact(BaseModel):
        """A contact extracted from a note.

        Args:
            name: The contact's full name.
            nickname: A short display name.
            unknown: This parameter is intentionally not a model field.
        """

        name: str
        nickname: str = Field(description="Keep the explicit field description.")
        visits: int = 0

    schema = generate_openai_schema(Contact)
    properties = schema["parameters"]["properties"]

    assert schema["name"] == "Contact"
    assert schema["description"].startswith("A contact extracted from a note.")
    assert properties["name"]["description"] == "The contact's full name."
    assert (
        properties["nickname"]["description"] == "Keep the explicit field description."
    )
    assert "unknown" not in properties
    assert schema["parameters"]["required"] == ["name", "nickname"]


@pytest.mark.parametrize(
    "message",
    [
        {"role": "user"},
        {"role": "user", "content": [{"type": "text", "text": "{{ name }}"}]},
    ],
)
def test_openai_templating_leaves_non_string_content_unchanged(
    message: dict[str, Any],
) -> None:
    calls: list[str] = []

    def apply_template(text: str, _context: dict[str, Any]) -> str:
        calls.append(text)
        return "rendered"

    assert process_message(message, {"name": "Ada"}, apply_template) is message
    assert calls == []


def test_openai_templating_renders_string_content_in_place() -> None:
    message = {"role": "user", "content": "Hello {{ name }}"}
    contexts: list[dict[str, Any]] = []

    def apply_template(text: str, context: dict[str, Any]) -> str:
        contexts.append(context)
        return text.replace("{{ name }}", context["name"])

    assert process_message(message, {"name": "Ada"}, apply_template) is message
    assert message["content"] == "Hello Ada"
    assert contexts == [{"name": "Ada"}]


def test_openai_schema_supplies_a_description_when_the_model_has_no_help_text() -> None:
    class Count(BaseModel):
        value: int

    Count.__doc__ = None
    schema = generate_openai_schema(Count)

    assert schema["description"] == (
        "Correctly extracted `Count` with all the required parameters with correct types"
    )
    assert schema["parameters"]["required"] == ["value"]


def test_openai_package_stays_importable_when_client_dependency_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__
    blocked: list[str] = []

    def import_without_client(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "instructor.v2.providers.openai.client":
            blocked.append(name)
            raise ImportError("openai dependency is unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_client)
    init_path = Path(__file__).parents[2] / "instructor/v2/providers/openai/__init__.py"
    namespace = runpy.run_path(str(init_path), run_name="openai_optional_import_test")

    assert blocked == ["instructor.v2.providers.openai.client"]
    assert namespace["from_openai"] is None
    assert namespace["__all__"] == ["from_openai"]
