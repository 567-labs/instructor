import json
import logging
from typing import Any, Callable, Optional, cast

import pytest
from openai.types.chat import ChatCompletion
from typing_extensions import NotRequired, TypedDict
from pydantic import BaseModel, ValidationError
from instructor import Mode
from instructor.processing.response import (
    handle_response_model,
    process_response,
    process_response_async,
)
from instructor.v2.providers.bedrock.handlers import (
    _prepare_bedrock_converse_kwargs_internal,
)


def test_typed_dict_conversion() -> None:
    class User(TypedDict):
        name: str
        age: int

    _, user_tool_definition = handle_response_model(User)

    class User(BaseModel):
        name: str
        age: int

    _, pydantic_user_tool_definition = handle_response_model(User)
    assert user_tool_definition == pydantic_user_tool_definition


@pytest.mark.parametrize(
    "keys",
    [
        ["_id"],
        ["_"],
        ["__dunder__"],
        ["__config__"],
        ["_id", "field_id", "field_id_"],
        ["_", "__", "field_", "field__"],
    ],
)
def test_typed_dict_underscore_keys_survive_tool_requests_and_responses(
    keys: list[str],
) -> None:
    typed_dict_factory = cast(Callable[..., type[Any]], TypedDict)
    record = typed_dict_factory("Record", {key: str for key in keys})
    model, request = handle_response_model(record, mode=Mode.TOOLS)
    assert isinstance(model, type) and issubclass(model, BaseModel)
    schema = request["tools"][0]["function"]["parameters"]
    assert set(schema["properties"]) == set(keys)
    assert set(schema["required"]) == set(keys)

    payload = {key: f"value-{index}" for index, key in enumerate(keys)}
    completion = ChatCompletion.model_validate(
        {
            "id": "offline-completion",
            "object": "chat.completion",
            "created": 0,
            "model": "offline-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "Record",
                                    "arguments": json.dumps(payload),
                                },
                            }
                        ],
                    },
                }
            ],
        }
    )
    result = process_response(
        completion, response_model=model, mode=Mode.TOOLS, stream=False
    )
    assert result.model_dump(by_alias=True) == payload
    assert json.loads(result.model_dump_json(by_alias=True)) == payload

    missing_required = dict(payload)
    del missing_required[keys[0]]
    with pytest.raises(ValidationError):
        model.model_validate(missing_required)


def test_iterable_typed_dict_preserves_optional_underscore_key() -> None:
    class Record(TypedDict):
        name: str
        _id: NotRequired[str]

    model, _ = handle_response_model(list[Record], mode=Mode.TOOLS)
    assert isinstance(model, type) and issubclass(model, BaseModel)
    result = model.model_validate(
        {"tasks": [{"name": "first", "_id": "doc-1"}, {"name": "second"}]}
    )
    assert result.model_dump(by_alias=True, exclude_unset=True) == {
        "tasks": [{"name": "first", "_id": "doc-1"}, {"name": "second"}]
    }


def test_request_log_omits_provider_options(caplog: pytest.LogCaptureFixture) -> None:
    class User(BaseModel):
        name: str

    headers = {"Cookie": "session-canary", "X-Custom-Auth": "gateway-canary"}
    with caplog.at_level(logging.DEBUG, logger="instructor"):
        _, kwargs = handle_response_model(
            User, messages=[], extra_headers=headers, api_key="provider-canary"
        )

    assert kwargs["extra_headers"] == headers
    assert kwargs["api_key"] == "provider-canary"
    record = next(
        r for r in caplog.records if r.getMessage().startswith("Instructor Request")
    )
    assert record.__dict__["mode"] == Mode.TOOLS.value
    assert record.__dict__["response_model"] == "User"
    for value in (*headers.values(), kwargs["api_key"]):
        assert value not in record.getMessage()
        assert value not in repr(vars(record))


@pytest.mark.asyncio
@pytest.mark.parametrize("opt_in", [None, "0", "1"])
async def test_raw_response_logging_requires_opt_in(
    opt_in: Optional[str],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    if opt_in is None:
        monkeypatch.delenv("INSTRUCTOR_LOG_RAW_RESPONSES", raising=False)
    else:
        monkeypatch.setenv("INSTRUCTOR_LOG_RAW_RESPONSES", opt_in)

    canary = "private-response-canary"
    response = ChatCompletion.model_validate(
        {
            "id": "test",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": canary},
                }
            ],
        }
    )
    with caplog.at_level(logging.DEBUG, logger="instructor"):
        assert process_response(response, response_model=None, stream=False) is response
        assert await process_response_async(response, response_model=None) is response

    assert sum(canary in record.getMessage() for record in caplog.records) == (
        2 if opt_in == "1" else 0
    )
    if opt_in != "1":
        assert all(canary not in repr(vars(record)) for record in caplog.records)


def test_openai_to_bedrock_conversion() -> None:
    """OpenAI-style input should be fully converted to Bedrock format."""
    call_kwargs = {
        "model": "anthropic.claude-3-haiku-20240307-v1:0",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Extract: Jason is 22 years old"},
            {"role": "assistant", "content": "Sure! Jason is 22."},
        ],
    }
    result = _prepare_bedrock_converse_kwargs_internal(call_kwargs)
    assert "model" not in result
    assert result["modelId"] == "anthropic.claude-3-haiku-20240307-v1:0"
    assert result["system"] == [{"text": "You are a helpful assistant."}]
    assert len(result["messages"]) == 2
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][0]["content"] == [
        {"text": "Extract: Jason is 22 years old"}
    ]
    assert result["messages"][1]["role"] == "assistant"
    assert result["messages"][1]["content"] == [{"text": "Sure! Jason is 22."}]


def test_bedrock_native_preserved() -> None:
    """Bedrock-native input should be preserved as-is."""
    call_kwargs = {
        "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
        "system": [{"text": "You are a helpful assistant."}],
        "messages": [
            {"role": "user", "content": [{"text": "Extract: Jason is 22 years old"}]},
            {"role": "assistant", "content": [{"text": "Sure! Jason is 22."}]},
        ],
    }
    result = _prepare_bedrock_converse_kwargs_internal(call_kwargs)
    assert result["system"] == [{"text": "You are a helpful assistant."}]
    assert len(result["messages"]) == 2
    assert result["messages"][0]["content"] == [
        {"text": "Extract: Jason is 22 years old"}
    ]
    assert result["messages"][1]["content"] == [{"text": "Sure! Jason is 22."}]


def test_mixed_openai_and_bedrock() -> None:
    """Mixed input: OpenAI-style is converted, Bedrock-native is preserved."""
    call_kwargs = {
        "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
        "system": [{"text": "You are a helpful assistant."}],
        "messages": [
            {
                "role": "user",
                "content": "Extract: Jason is 22 years old",
            },  # OpenAI style
            {
                "role": "assistant",
                "content": [{"text": "Sure! Jason is 22."}],
            },  # Bedrock style
        ],
    }
    result = _prepare_bedrock_converse_kwargs_internal(call_kwargs)
    assert result["modelId"] == "anthropic.claude-3-haiku-20240307-v1:0"
    assert result["system"] == [{"text": "You are a helpful assistant."}]
    assert len(result["messages"]) == 2
    # OpenAI-style user message converted
    assert result["modelId"] == "anthropic.claude-3-haiku-20240307-v1:0"
    assert result["messages"][0]["content"] == [
        {"text": "Extract: Jason is 22 years old"}
    ]
    # Bedrock-style assistant message preserved
    assert result["messages"][1]["content"] == [{"text": "Sure! Jason is 22."}]


def test_bedrock_round_trip() -> None:
    """Bedrock input should be unchanged after round-trip through the function."""
    call_kwargs = {
        "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
        "system": [{"text": "Bedrock system."}],
        "messages": [
            {"role": "user", "content": [{"text": "Bedrock user message."}]},
        ],
    }
    import copy

    original = copy.deepcopy(call_kwargs)
    result = _prepare_bedrock_converse_kwargs_internal(call_kwargs)
    assert result == original


def test_empty_and_missing_content() -> None:
    """Empty messages and missing content should be handled gracefully."""
    # Empty messages
    call_kwargs = {"messages": []}
    result = _prepare_bedrock_converse_kwargs_internal(call_kwargs)
    assert result["messages"] == []
    # Message with no content
    call_kwargs = {"messages": [{"role": "user"}]}
    result = _prepare_bedrock_converse_kwargs_internal(call_kwargs)
    assert result["messages"][0]["role"] == "user"
    # Should not add a content key if not present
    assert "content" not in result["messages"][0]


def test_bedrock_invalid_content_format() -> None:
    """Invalid content types should raise ValueError."""
    call_kwargs = {
        "messages": [{"role": "user", "content": 12345}]  # Invalid content type
    }
    try:
        _prepare_bedrock_converse_kwargs_internal(call_kwargs)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "Unsupported message content type for Bedrock" in str(e)
