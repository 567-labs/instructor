"""Offline regression tests for public model defaults."""

from __future__ import annotations

from inspect import signature
import json
from typing import Any

import httpx
import pytest
from openai import OpenAI
from pydantic import BaseModel

import instructor
from instructor.batch.request import BatchRequest
from instructor.distil import Instructions
from instructor.v2.validation.llm_validators import llm_validator


CURRENT_DEFAULT_MODEL = "gpt-6-luna"


def test_llm_validator_uses_current_default_model() -> None:
    assert signature(llm_validator).parameters["model"].default == CURRENT_DEFAULT_MODEL


def test_distil_uses_current_default_model() -> None:
    assert (
        signature(Instructions.distil).parameters["model"].default
        == CURRENT_DEFAULT_MODEL
    )


class Result(BaseModel):
    value: int


@pytest.mark.parametrize(
    ("mode", "request_options", "expected"),
    [
        (instructor.Mode.TOOLS, {}, "none"),
        (instructor.Mode.JSON_SCHEMA, {"temperature": 0}, "none"),
        (instructor.Mode.JSON_SCHEMA, {"top_p": 0.9}, "none"),
        (instructor.Mode.JSON_SCHEMA, {}, None),
        (instructor.Mode.TOOLS, {"reasoning_effort": "low"}, "low"),
    ],
)
def test_luna_chat_reasoning_defaults(
    mode: instructor.Mode, request_options: dict[str, Any], expected: str | None
) -> None:
    bodies = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        message: dict[str, Any] = {"role": "assistant", "content": '{"value":7}'}
        if "tools" in body:
            message["tool_calls"] = [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "Result", "arguments": '{"value":7}'},
                }
            ]
        return httpx.Response(
            200,
            json={
                "id": "local",
                "created": 0,
                "model": CURRENT_DEFAULT_MODEL,
                "object": "chat.completion",
                "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            },
        )

    transport = httpx.MockTransport(respond)
    with OpenAI(api_key="local", http_client=httpx.Client(transport=transport)) as sdk:
        client = instructor.from_openai(sdk, mode=mode, model=CURRENT_DEFAULT_MODEL)
        result = client.create(
            response_model=Result,
            messages=[{"role": "user", "content": "Return 7."}],
            max_retries=0,
            **request_options,
        )
    assert result.model_dump() == {"value": 7}
    assert len(bodies) == 1
    assert bodies[0].get("reasoning_effort") == expected
    assert bodies[0]["model"] == CURRENT_DEFAULT_MODEL


def test_new_batch_defaults_use_supported_parameters() -> None:
    request = BatchRequest[Result](
        custom_id="record",
        messages=[{"role": "user", "content": "Return 7."}],
        response_model=Result,
        model=CURRENT_DEFAULT_MODEL,
        max_tokens=128,
    )
    openai_body = request.to_openai_format()["body"]
    assert openai_body["max_completion_tokens"] == 128
    assert openai_body["reasoning_effort"] == "none"
    assert "max_tokens" not in openai_body
    anthropic_body = request.model_copy(
        update={"model": "claude-haiku-5-5"}
    ).to_anthropic_format()["params"]
    assert anthropic_body["max_tokens"] == 128
    assert "temperature" not in anthropic_body
    assert anthropic_body["tool_choice"] == {"type": "tool", "name": "extract_data"}
