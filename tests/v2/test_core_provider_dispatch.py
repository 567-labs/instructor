"""Tests for shared-core dispatch into provider-owned helpers."""

from __future__ import annotations

from typing import Any, cast

import pytest

from instructor import Mode, Provider
from instructor.v2.core import retry, templating
from instructor.v2.core.multimodal import (
    Audio,
    Image,
    ImageWithCacheControl,
    PDF,
    PDFWithCacheControl,
    PDFWithGenaiFile,
    extract_genai_multimodal_content,
)


@pytest.mark.parametrize(
    ("provider", "message", "target"),
    [
        (
            Provider.OPENAI,
            {"content": "Hello {{ name }}"},
            "instructor.v2.providers.openai.templating.process_message",
        ),
        (
            Provider.ANTHROPIC,
            {"content": [{"type": "text", "text": "Hello {{ name }}"}]},
            "instructor.v2.providers.anthropic.templating.process_message",
        ),
        (
            Provider.GEMINI,
            {"parts": ["Hello {{ name }}"]},
            "instructor.v2.providers.gemini.templating.process_message",
        ),
        (
            Provider.COHERE,
            {"message": "Hello {{ name }}"},
            "instructor.v2.providers.cohere.templating.process_message",
        ),
    ],
)
def test_process_message_dispatches_to_provider_modules(
    monkeypatch: pytest.MonkeyPatch,
    provider: Provider,
    message: dict[str, Any],
    target: str,
) -> None:
    calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def fake_process_message(
        value: dict[str, Any],
        context: dict[str, Any],
        _apply_template: Any,
    ) -> dict[str, str]:
        calls.append((value, context))
        return {"provider": provider.value}

    monkeypatch.setattr(target, fake_process_message)

    assert templating.process_message(message, {"name": "Ada"}, provider) == {
        "provider": provider.value
    }
    assert calls == [(message, {"name": "Ada"})]


def test_handle_templating_cohere_message_without_chat_history() -> None:
    result = templating.handle_templating(
        {"message": "Hello {{ name }}"},
        Mode.TOOLS,
        provider=Provider.COHERE,
        context={"name": "Ada"},
    )

    assert result == {"message": "Hello Ada", "chat_history": []}


def test_initialize_usage_dispatches_anthropic_to_provider_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    calls: list[str] = []

    def fake_initialize_usage() -> object:
        calls.append("anthropic")
        return sentinel

    monkeypatch.setattr(
        "instructor.v2.providers.anthropic.usage.initialize_usage",
        fake_initialize_usage,
    )

    assert retry._initialize_usage(Mode.ANTHROPIC_TOOLS) is sentinel
    assert calls == ["anthropic"]


def test_update_total_usage_dispatches_anthropic_to_provider_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from instructor.v2.core import usage

    class Response:
        usage = object()

    response = Response()
    total_usage = cast(Any, object())
    calls: list[tuple[Any, Any]] = []

    def fake_update_total_usage(response_usage: Any, running_total: Any) -> bool:
        calls.append((response_usage, running_total))
        return True

    monkeypatch.setattr(
        "instructor.v2.providers.anthropic.usage.update_total_usage",
        fake_update_total_usage,
    )

    assert usage.update_total_usage(response, total_usage) is response
    assert calls == [(response.usage, total_usage)]


@pytest.mark.parametrize(
    ("instance", "method_name", "target", "args"),
    [
        (
            Image(
                source="data:image/png;base64,AA==", media_type="image/png", data="AA=="
            ),
            "to_openai",
            "instructor.v2.providers.openai.multimodal.image_to_openai",
            (Mode.TOOLS,),
        ),
        (
            Image(
                source="data:image/png;base64,AA==", media_type="image/png", data="AA=="
            ),
            "to_anthropic",
            "instructor.v2.providers.anthropic.multimodal.image_to_anthropic",
            (),
        ),
        (
            Audio(source="audio.wav", media_type="audio/wav", data="AA=="),
            "to_genai",
            "instructor.v2.providers.genai.multimodal.audio_to_genai",
            (),
        ),
        (
            PDF(
                source="https://example.com/file.pdf",
                media_type="application/pdf",
                data=None,
            ),
            "to_mistral",
            "instructor.v2.providers.mistral.multimodal.pdf_to_mistral",
            (),
        ),
        (
            ImageWithCacheControl(
                source="data:image/png;base64,AA==",
                media_type="image/png",
                data="AA==",
                cache_control={"type": "ephemeral"},
            ),
            "to_anthropic",
            "instructor.v2.providers.anthropic.multimodal.image_with_cache_control_to_anthropic",
            (),
        ),
        (
            PDFWithCacheControl(
                source="data:application/pdf;base64,AA==",
                media_type="application/pdf",
                data="AA==",
            ),
            "to_anthropic",
            "instructor.v2.providers.anthropic.multimodal.pdf_with_cache_control_to_anthropic",
            (),
        ),
        (
            PDFWithGenaiFile(
                source="https://generativelanguage.googleapis.com/v1beta/files/123",
                media_type="application/pdf",
                data=None,
            ),
            "to_genai",
            "instructor.v2.providers.genai.multimodal.uploaded_pdf_to_genai",
            (),
        ),
    ],
)
def test_multimodal_methods_delegate_to_provider_modules(
    monkeypatch: pytest.MonkeyPatch,
    instance: Any,
    method_name: str,
    target: str,
    args: tuple[Any, ...],
) -> None:
    sentinel = {"delegated": True}
    calls: list[tuple[Any, ...]] = []

    def fake_encoder(*encoder_args: Any) -> dict[str, bool]:
        calls.append(encoder_args)
        return sentinel

    monkeypatch.setattr(target, fake_encoder)

    assert getattr(instance, method_name)(*args) is sentinel
    assert calls == [(instance, *args)]


def test_genai_multimodal_extractor_delegates_to_provider_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = [object()]
    calls: list[tuple[list[Any], bool]] = []

    def fake_extract_multimodal_content(
        contents: list[Any], autodetect_images: bool = True
    ) -> list[Any]:
        calls.append((contents, autodetect_images))
        return sentinel

    monkeypatch.setattr(
        "instructor.v2.providers.genai.multimodal.extract_multimodal_content",
        fake_extract_multimodal_content,
    )

    contents = [object()]
    assert extract_genai_multimodal_content(contents, False) is sentinel
    assert calls == [(contents, False)]
