"""Offline regression tests for public model defaults."""

from __future__ import annotations

from functools import partial
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
from instructor.v2.core.patch import patch_v2
from instructor.v2.core.providers import Provider
from instructor.v2.providers.openai.client import map_chat_completion_to_response
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


def _luna_response_payload(mode: instructor.Mode) -> dict[str, Any]:
    if mode in (
        instructor.Mode.RESPONSES_TOOLS,
        instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
    ):
        return {
            "id": "resp_local",
            "object": "response",
            "created_at": 1,
            "model": CURRENT_DEFAULT_MODEL,
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "Result",
                    "arguments": '{"value":7}',
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
    return {
        "id": "chat_local",
        "object": "chat.completion",
        "created": 1,
        "model": CURRENT_DEFAULT_MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "Result", "arguments": '{"value":7}'},
                        }
                    ],
                },
            }
        ],
    }


@pytest.mark.parametrize(
    ("mode", "expects_reasoning_effort"),
    [
        (instructor.Mode.TOOLS, True),
        (instructor.Mode.RESPONSES_TOOLS, False),
        (instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS, False),
    ],
)
def test_luna_reasoning_default_excluded_for_responses_modes(
    mode: instructor.Mode, expects_reasoning_effort: bool
) -> None:
    # Patch the exact mode here: from_openai normalizes the inbuilt mode to
    # RESPONSES_TOOLS, which would hide a regression in the mode check.
    bodies: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_luna_response_payload(mode))

    is_responses_mode = mode in (
        instructor.Mode.RESPONSES_TOOLS,
        instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
    )
    with OpenAI(
        api_key="local", http_client=httpx.Client(transport=httpx.MockTransport(respond))
    ) as sdk:
        create = (
            partial(map_chat_completion_to_response, client=sdk)
            if is_responses_mode
            else sdk.chat.completions.create
        )
        patched_create = patch_v2(
            create, Provider.OPENAI, mode, default_model=CURRENT_DEFAULT_MODEL
        )
        result = patched_create(
            response_model=Result,
            messages=[{"role": "user", "content": "Return 7."}],
            temperature=0,
            max_retries=0,
        )

    assert result.model_dump() == {"value": 7}
    assert len(bodies) == 1
    if expects_reasoning_effort:
        assert bodies[0]["reasoning_effort"] == "none"
    else:
        assert "reasoning_effort" not in bodies[0]


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
