from typing_extensions import TypedDict
from pydantic import BaseModel
from instructor.processing.response import handle_response_model
from instructor.v2.core.response import _redact_kwargs
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


def test_typed_dict_underscore_prefixed_keys() -> None:
    import pytest
    from typing_extensions import NotRequired
    from pydantic import ValidationError

    class Record(TypedDict):
        _id: str
        name: str
        _opt: NotRequired[int]
        field_id: NotRequired[str]

    model, tool_def = handle_response_model(Record)
    params = tool_def["tools"][0]["function"]["parameters"]

    # 1. Schema must include _id and field_id in properties, and only required keys in required
    assert "_id" in params["properties"]
    assert "name" in params["properties"]
    assert "_opt" in params["properties"]
    assert "field_id" in params["properties"]
    assert set(params["required"]) == {"_id", "name"}

    # 2. Validation preserves _id with by_alias=True and dict()
    inst = model.model_validate(
        {"_id": "doc-1", "name": "Report", "_opt": 42, "field_id": "cust-9"}
    )
    assert inst.model_dump(by_alias=True) == {
        "_id": "doc-1",
        "name": "Report",
        "_opt": 42,
        "field_id": "cust-9",
    }
    assert dict(inst) == {
        "_id": "doc-1",
        "name": "Report",
        "_opt": 42,
        "field_id": "cust-9",
    }
    attr_key = "_id"
    assert getattr(inst, attr_key) == "doc-1"
    assert inst._id == "doc-1"
    assert inst.field_id == "cust-9"

    # Optional _opt omitted
    inst_no_opt = model.model_validate({"_id": "doc-2", "name": "Doc"})
    assert inst_no_opt._opt is None
    assert inst_no_opt.model_dump(by_alias=True, exclude_unset=True) == {
        "_id": "doc-2",
        "name": "Doc",
    }

    # 3. Missing required _id must raise ValidationError
    with pytest.raises(ValidationError):
        model.model_validate({"name": "Report"})


def test_typed_dict_iterable_with_underscore_prefixed_keys() -> None:
    class Item(TypedDict):
        _id: str
        label: str

    model, tool_def = handle_response_model(list[Item])
    params = tool_def["tools"][0]["function"]["parameters"]
    item_schema = (
        params.get("$defs", {}).get("Item") or params["properties"]["tasks"]["items"]
    )
    item_props = item_schema["properties"]
    item_req = item_schema["required"]

    assert "_id" in item_props
    assert "label" in item_props
    assert "_id" in item_req

    inst = model.model_validate(
        {"tasks": [{"_id": "1", "label": "first"}, {"_id": "2", "label": "second"}]}
    )
    assert inst.tasks[0]._id == "1"
    assert dict(inst.tasks[1]) == {"_id": "2", "label": "second"}


def test_redact_kwargs_hides_nested_sensitive_fields() -> None:
    kwargs = {
        "api_key": "top-level",
        "headers": {
            "Authorization": "Bearer secret",
            "x-api-key": "nested secret",
            "safe": "visible",
        },
        "messages": [{"token": "inner secret", "content": "hello"}],
    }

    assert _redact_kwargs(kwargs) == {
        "api_key": "[redacted]",
        "headers": {
            "Authorization": "[redacted]",
            "x-api-key": "[redacted]",
            "safe": "visible",
        },
        "messages": [{"token": "[redacted]", "content": "hello"}],
    }


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
