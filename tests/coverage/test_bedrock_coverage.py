"""Focused offline coverage for the Bedrock v2 client and handlers."""

from __future__ import annotations

import builtins
import runpy
from pathlib import Path
from typing import Any

import pytest
from botocore.client import BaseClient
from pydantic import BaseModel

from instructor import Mode, Provider
from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.errors import (
    ClientError,
    ConfigurationError,
    ResponseParsingError,
)
from instructor.v2.providers.bedrock import client as bedrock_client
from instructor.v2.providers.bedrock.handlers import (
    BedrockMDJSONHandler,
    BedrockToolsHandler,
    _extract_bedrock_text,
    _extract_bedrock_tool_input,
    _openai_image_part_to_bedrock,
    _prepare_bedrock_converse_kwargs_internal,
    _to_bedrock_content_items,
    handle_bedrock_json,
    reask_bedrock_tools,
)


class Answer(BaseModel):
    value: int


def _tool_response(value: Any, *, name: str = "Answer") -> dict[str, Any]:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tool-123",
                            "name": name,
                            "input": {"value": value},
                        }
                    }
                ],
            }
        }
    }


def test_package_and_client_report_missing_optional_bedrock_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__
    provider_dir = Path(bedrock_client.__file__).parent

    def no_client(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "instructor.v2.providers.bedrock.client":
            raise ImportError("bedrock client unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_client)
    package_namespace = runpy.run_path(str(provider_dir / "__init__.py"))
    assert package_namespace["from_bedrock"] is None

    def no_botocore(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "botocore.client":
            raise ImportError("botocore unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_botocore)
    client_namespace = runpy.run_path(str(provider_dir / "client.py"))
    assert client_namespace["BaseClient"] is None
    with pytest.raises(ClientError, match="botocore is not installed"):
        client_namespace["from_bedrock"](None)


def test_sync_bedrock_client_sends_native_converse_request_and_parses_tool() -> None:
    calls: list[dict[str, Any]] = []

    def converse(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return _tool_response(7)

    client = BaseClient.__new__(BaseClient)
    client.converse = converse
    wrapped = bedrock_client.from_bedrock(
        client, mode=Mode.TOOLS, model="anthropic.claude-test"
    )

    assert isinstance(wrapped, Instructor)
    assert wrapped.provider is Provider.BEDROCK
    result = wrapped.create(
        response_model=Answer,
        messages=[{"role": "user", "content": "Give me seven"}],
        max_retries=1,
    )

    assert result.model_dump() == {"value": 7}
    assert calls == [
        {
            "modelId": "anthropic.claude-test",
            "messages": [{"role": "user", "content": [{"text": "Give me seven"}]}],
            "toolConfig": {
                "tools": [
                    {
                        "toolSpec": {
                            "name": "Answer",
                            "description": (
                                "Correctly extracted `Answer` with all the required "
                                "parameters with correct types"
                            ),
                            "inputSchema": {"json": Answer.model_json_schema()},
                        }
                    }
                ],
                "toolChoice": {"tool": {"name": "Answer"}},
            },
        }
    ]


@pytest.mark.asyncio
async def test_async_bedrock_client_wraps_sync_converse_and_parses_tool() -> None:
    calls: list[dict[str, Any]] = []

    def converse(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return _tool_response(11)

    client = BaseClient.__new__(BaseClient)
    client.converse = converse
    wrapped = bedrock_client.from_bedrock(
        client,
        mode=Mode.TOOLS,
        async_client=True,
        model="anthropic.claude-async-test",
    )

    assert isinstance(wrapped, AsyncInstructor)
    assert wrapped.provider is Provider.BEDROCK
    result = await wrapped.create(
        response_model=Answer,
        messages=[{"role": "user", "content": "Give me eleven"}],
        max_retries=1,
    )

    assert result.model_dump() == {"value": 11}
    assert calls[0]["modelId"] == "anthropic.claude-async-test"
    assert calls[0]["messages"] == [
        {"role": "user", "content": [{"text": "Give me eleven"}]}
    ]
    assert calls[0]["toolConfig"]["toolChoice"] == {"tool": {"name": "Answer"}}


def test_tools_reask_without_tool_invocation_adds_plain_correction() -> None:
    original = {"messages": [{"role": "user", "content": [{"text": "extract"}]}]}
    response = {
        "output": {
            "message": {"role": "assistant", "content": [{"text": "not a tool"}]}
        }
    }

    result = reask_bedrock_tools(original, response, ValueError("value is required"))

    assert result["messages"] == [
        {"role": "user", "content": [{"text": "extract"}]},
        {"role": "assistant", "content": [{"text": "not a tool"}]},
        {
            "role": "user",
            "content": [
                {
                    "text": (
                        "Validation Error due to no tool invocation:\n"
                        "value is required\n"
                        "Recall the function correctly, fix the errors"
                    )
                }
            ],
        },
    ]


def test_tools_reask_without_content_adds_plain_correction() -> None:
    original = {"messages": [{"role": "user", "content": [{"text": "extract"}]}]}
    response = {"output": {"message": {"role": "assistant"}}}

    result = reask_bedrock_tools(original, response, ValueError("value is required"))

    assert result["messages"] == [
        {"role": "user", "content": [{"text": "extract"}]},
        {"role": "assistant"},
        {
            "role": "user",
            "content": [
                {
                    "text": (
                        "Validation Error due to no tool invocation:\n"
                        "value is required\n"
                        "Recall the function correctly, fix the errors"
                    )
                }
            ],
        },
    ]


@pytest.mark.parametrize(
    ("part", "message"),
    [
        ({"type": "image_url", "image_url": {}}, "image_url.url is required"),
        (
            {"type": "image_url", "image_url": {"url": "data:image/png;base64"}},
            "Invalid data URL",
        ),
        (
            {"type": "image_url", "image_url": {"url": "data:image/png,abc"}},
            "Only base64 data URLs are supported",
        ),
        (
            {"type": "image_url", "image_url": {"url": "s3://bucket/image.png"}},
            "Unsupported image_url scheme",
        ),
    ],
)
def test_openai_image_conversion_rejects_invalid_inputs(
    part: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _openai_image_part_to_bedrock(part)


def test_openai_data_image_uses_filename_when_mime_type_is_missing() -> None:
    result = _openai_image_part_to_bedrock(
        {
            "type": "image_url",
            "image_url": {"url": 'data:;name="scan.png";base64,aW1hZ2UtYnl0ZXM='},
        }
    )

    assert result == {"image": {"format": "png", "source": {"bytes": b"image-bytes"}}}


def test_openai_data_image_without_mime_or_known_filename_defaults_to_jpeg() -> None:
    result = _openai_image_part_to_bedrock(
        {
            "type": "image_url",
            "image_url": {
                "url": "data:;charset=utf-8;name=scan;base64,aW1hZ2UtYnl0ZXM="
            },
        }
    )

    assert result == {"image": {"format": "jpeg", "source": {"bytes": b"image-bytes"}}}


def test_content_conversion_keeps_native_cache_and_string_parts() -> None:
    cache_point = {"cachePoint": {"type": "default"}}

    assert _to_bedrock_content_items([cache_point, "tail"]) == [
        cache_point,
        {"text": "tail"},
    ]


def test_content_conversion_keeps_native_reasoning_content() -> None:
    """Assistant turns returned by Converse can be sent back unchanged."""
    redacted = {"reasoningContent": {"redactedContent": b"rsn_opaque"}}
    text = {"reasoningContent": {"reasoningText": {"text": "thinking"}}}

    assert _to_bedrock_content_items([redacted, text, {"text": "done"}]) == [
        redacted,
        text,
        {"text": "done"},
    ]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ([{"type": "audio", "audio": "..."}], "Unsupported OpenAI-style part type"),
        ([{"text": 3}], "Unsupported dict content"),
        ([17], "Unsupported content part"),
    ],
)
def test_content_conversion_rejects_unsupported_parts(
    content: list[Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _to_bedrock_content_items(content)


def test_prepare_converse_handles_native_system_without_messages() -> None:
    result = _prepare_bedrock_converse_kwargs_internal(
        {"system": [{"text": "Keep answers short"}], "model": "model-a"}
    )

    assert result == {
        "modelId": "model-a",
        "system": [{"text": "Keep answers short"}],
        "messages": [],
    }


def test_prepare_converse_drops_an_empty_native_system_list() -> None:
    result = _prepare_bedrock_converse_kwargs_internal(
        {"system": [], "model": "model-a"}
    )

    assert result == {"modelId": "model-a"}


@pytest.mark.parametrize(
    ("stop_kwargs", "expected"),
    [
        ({"stop": "END"}, ["END"]),
        ({"stop_sequences": ["DONE"]}, ["DONE"]),
        ({"stopSequences": ["HALT"]}, ["HALT"]),
    ],
)
def test_prepare_converse_maps_stop_variants(
    stop_kwargs: dict[str, Any], expected: list[str]
) -> None:
    result = _prepare_bedrock_converse_kwargs_internal(
        {"messages": [{"role": "user", "content": "go"}], **stop_kwargs}
    )

    assert result["inferenceConfig"]["stopSequences"] == expected
    assert result["messages"] == [{"role": "user", "content": [{"text": "go"}]}]


def test_prepare_converse_drops_an_unset_stop_value() -> None:
    result = _prepare_bedrock_converse_kwargs_internal(
        {"messages": [{"role": "user", "content": "go"}], "stop": None}
    )

    assert result == {"messages": [{"role": "user", "content": [{"text": "go"}]}]}


def test_prepare_converse_keeps_existing_model_specific_top_k() -> None:
    result = _prepare_bedrock_converse_kwargs_internal(
        {
            "top_k": 12,
            "additionalModelRequestFields": {"top_k": 7, "custom": "keep"},
            "messages": [{"role": "user", "content": "go"}],
        }
    )

    assert result == {
        "additionalModelRequestFields": {"top_k": 7, "custom": "keep"},
        "messages": [{"role": "user", "content": [{"text": "go"}]}],
    }


def test_prepare_converse_maps_inference_values_without_messages() -> None:
    result = _prepare_bedrock_converse_kwargs_internal(
        {"modelId": "model-a", "temperature": 0.3}
    )

    assert result == {
        "modelId": "model-a",
        "inferenceConfig": {"temperature": 0.3},
    }


def test_prepare_converse_merges_camel_case_inference_values_and_non_dict_messages() -> (
    None
):
    result = _prepare_bedrock_converse_kwargs_internal(
        {
            "maxTokens": 120,
            "topP": 0.2,
            "temperature": 0.7,
            "inferenceConfig": {"temperature": 0.1},
            "messages": ["already serialized"],
        }
    )

    assert result == {
        "inferenceConfig": {"temperature": 0.1, "maxTokens": 120, "topP": 0.2},
        "messages": ["already serialized"],
    }


def test_prepare_converse_rejects_non_string_system_message() -> None:
    with pytest.raises(ValueError, match="System message content must be a string"):
        _prepare_bedrock_converse_kwargs_internal(
            {"messages": [{"role": "system", "content": [{"text": "nested"}]}]}
        )


def test_json_request_appends_schema_instruction_to_existing_system_message() -> None:
    model, result = handle_bedrock_json(
        Answer,
        {"messages": [{"role": "system", "content": "Keep answers exact"}]},
    )

    assert model is Answer
    assert result["messages"] == []
    assert result["system"][0] == {"text": "Keep answers exact"}
    assert "parsed objects in json" in result["system"][1]["text"]
    assert '"value"' in result["system"][1]["text"]


def test_json_request_rejects_invalid_native_system_shape() -> None:
    with pytest.raises(ValueError, match="system must be a list of SystemMessage"):
        handle_bedrock_json(Answer, {"system": "not-a-list", "messages": []})


def test_response_extractors_accept_text_object_and_reject_bad_tool_shapes() -> None:
    class TextResponse:
        text = '{"value": 5}'

    assert _extract_bedrock_text(TextResponse()) == '{"value": 5}'
    with pytest.raises(ResponseParsingError, match="no text attribute found"):
        _extract_bedrock_text(object())
    with pytest.raises(ResponseParsingError, match="expected dict response"):
        _extract_bedrock_tool_input(object(), Answer)
    with pytest.raises(ResponseParsingError, match="Tool name mismatch"):
        _extract_bedrock_tool_input(_tool_response(5, name="Other"), Answer)
    with pytest.raises(ResponseParsingError, match="No tool use found"):
        _extract_bedrock_tool_input(
            {"output": {"message": {"content": [{"text": "plain response"}]}}},
            Answer,
        )


@pytest.mark.parametrize(
    ("handler", "response", "mode"),
    [
        (BedrockToolsHandler(), _tool_response(1), "TOOLS"),
        (
            BedrockMDJSONHandler(),
            {"output": {"message": {"content": [{"text": '{"value": 1}'}]}}},
            "MD_JSON",
        ),
    ],
)
def test_bedrock_handlers_reject_unsupported_streaming(
    handler: BedrockToolsHandler | BedrockMDJSONHandler,
    response: dict[str, Any],
    mode: str,
) -> None:
    with pytest.raises(ConfigurationError, match=f"Bedrock in {mode} mode"):
        handler.parse_response(response, Answer, stream=True)
