from __future__ import annotations

import json
from typing import Any, cast

import pytest
from openai.types.chat import ChatCompletion
from pydantic import ValidationError
from typing_extensions import NotRequired, TypedDict

from instructor import Mode
from instructor.cache import AutoCache
from instructor.processing.response import handle_response_model, process_response
from instructor.v2.core.cache_response import (
    load_cached_response,
    store_cached_response,
)
from instructor.v2.core.response_model import prepare_response_model


@pytest.mark.parametrize(
    "key",
    [
        "model_validate",
        "model_dump",
        "model_config",
        "model_fields",
        "schema",
        "dict",
        "openai_schema",
        "parse_json",
    ],
)
def test_reserved_keys_survive_tool_request_and_response(key: str) -> None:
    record = cast(Any, TypedDict)("Record", {key: str})
    model, request = handle_response_model(record, mode=Mode.TOOLS)
    model = cast(Any, model)
    schema = request["tools"][0]["function"]["parameters"]
    assert set(schema["properties"]) == {key}
    assert schema["required"] == [key]

    payload = {key: "value"}
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
    round_trip = result.model_dump_json(round_trip=True)
    assert json.loads(round_trip) == payload
    assert model.model_validate_json(round_trip).model_dump(by_alias=True) == payload
    with pytest.raises(ValidationError):
        model.model_validate({})


def test_reserved_keys_do_not_collide_with_generated_aliases() -> None:
    class Record(TypedDict):
        model_dump: str
        fieldmodel_dump: str
        fieldmodel_dump_: str
        _id: str
        field_id: str

    model = cast(Any, prepare_response_model(Record))
    payload = {
        key: f"value-{index}" for index, key in enumerate(Record.__annotations__)
    }
    assert model.model_validate(payload).model_dump(by_alias=True) == payload
    assert set(model.model_json_schema()["required"]) == set(payload)


def test_iterable_reserved_keys_preserve_optional_fields_and_round_trips() -> None:
    class Record(TypedDict):
        model_dump: str
        schema: NotRequired[str]

    model = cast(Any, prepare_response_model(list[Record]))
    payload = {
        "tasks": [{"model_dump": "first", "schema": "x"}, {"model_dump": "second"}]
    }
    result = model.model_validate(payload)
    assert result.model_dump(by_alias=True, exclude_unset=True) == payload
    serialized = result.model_dump_json(round_trip=True)
    assert json.loads(serialized) == payload
    assert (
        model.model_validate_json(serialized).model_dump(
            by_alias=True, exclude_unset=True
        )
        == payload
    )
    assert set(model.task_type.model_json_schema()["required"]) == {"model_dump"}


def test_reserved_keys_survive_response_cache() -> None:
    class Record(TypedDict):
        model_dump: str
        schema: NotRequired[str]

    model = cast(Any, prepare_response_model(Record))
    payload = {"model_dump": "value"}
    cache = AutoCache()
    store_cached_response(cache, "record", model.model_validate(payload))
    restored = load_cached_response(cache, "record", model)
    assert restored.model_dump(by_alias=True, exclude_unset=True) == payload
