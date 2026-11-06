from __future__ import annotations

import pytest

from instructor.v2.providers.bedrock.handlers import (
    _prepare_bedrock_converse_kwargs_internal,
)


def test_prepare_bedrock_kwargs_openai_text_plus_image(tiny_png_data_url: str):
    call_kwargs = {
        "model": "anthropic.claude-3-5-sonnet",
        "temperature": 0.3,
        "max_tokens": 256,
        "top_p": 0.9,
        "stop": ["<END>"],
        "system": [{"text": "You are helpful."}],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "image_url", "image_url": {"url": tiny_png_data_url}},
                ],
            },
        ],
    }

    out = _prepare_bedrock_converse_kwargs_internal(call_kwargs)

    assert out["modelId"] == "anthropic.claude-3-5-sonnet"
    inf = out["inferenceConfig"]
    assert inf["temperature"] == 0.3
    assert inf["maxTokens"] == 256
    assert inf["topP"] == 0.9
    assert inf["stopSequences"] == ["<END>"]
    assert out["system"][0]["text"] == "You are helpful."

    parts = out["messages"][0]["content"]
    assert parts[0] == {"text": "hi"}
    assert parts[1]["image"]["format"] in {"jpeg", "png", "gif", "webp"}
    assert isinstance(parts[1]["image"]["source"]["bytes"], (bytes, bytearray))
    assert len(parts[1]["image"]["source"]["bytes"]) > 0


def test_prepare_bedrock_kwargs_openai_image_url_rejects_http():
    """Remote HTTP(S) URLs are not fetched (SSRF prevention) — must use data: URLs."""
    call_kwargs = {
        "model": "anthropic.claude-3-5-sonnet",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.jpg"},
                    }
                ],
            }
        ],
    }
    with pytest.raises(ValueError, match="Unsupported image_url scheme for Bedrock"):
        _prepare_bedrock_converse_kwargs_internal(call_kwargs)


def test_prepare_bedrock_kwargs_routes_top_k_to_additional_fields():
    """top_k is model-specific and must go to additionalModelRequestFields.

    AWS InferenceConfiguration only supports maxTokens/stopSequences/
    temperature/topP. A leftover top_k would otherwise reach
    client.converse(top_k=...) and boto3 raises ParamValidationError.
    """
    call_kwargs = {
        "model": "anthropic.claude-3-5-sonnet",
        "temperature": 0.3,
        "top_k": 200,
        "messages": [{"role": "user", "content": "hi"}],
    }

    out = _prepare_bedrock_converse_kwargs_internal(call_kwargs)

    assert out["additionalModelRequestFields"] == {"top_k": 200}
    assert "top_k" not in out
    assert "top_k" not in out["inferenceConfig"]
    assert "topK" not in out["inferenceConfig"]


def test_prepare_bedrock_kwargs_top_k_camel_case():
    """topK (camelCase) is also routed to additionalModelRequestFields."""
    call_kwargs = {
        "model": "anthropic.claude-3-5-sonnet",
        "topK": 100,
        "messages": [{"role": "user", "content": "hi"}],
    }

    out = _prepare_bedrock_converse_kwargs_internal(call_kwargs)

    assert out["additionalModelRequestFields"] == {"top_k": 100}
    assert "topK" not in out


def test_prepare_bedrock_kwargs_top_k_preserves_user_additional_fields():
    """A user-provided additionalModelRequestFields must not be clobbered."""
    call_kwargs = {
        "model": "anthropic.claude-3-5-sonnet",
        "top_k": 200,
        "additionalModelRequestFields": {"anthropic_beta": ["foo"]},
        "messages": [{"role": "user", "content": "hi"}],
    }

    out = _prepare_bedrock_converse_kwargs_internal(call_kwargs)

    assert out["additionalModelRequestFields"]["anthropic_beta"] == ["foo"]
    assert out["additionalModelRequestFields"]["top_k"] == 200
