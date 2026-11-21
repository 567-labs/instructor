"""Regression tests for tool schemas of self-referential response models."""

from __future__ import annotations

import json
from typing import Any, Optional

from openai import OpenAI
from pydantic import BaseModel

import instructor
from instructor import Mode
from instructor.v2.providers.anthropic.schema import generate_anthropic_schema
from instructor.v2.providers.openai.schema import generate_openai_schema
from tests import openai_http as httpx


class Node(instructor.OpenAISchema):
    """A tree node that can nest another node."""

    name: str
    child: Optional["Node"] = None


def _tool_names(payload: dict[str, Any]) -> str:
    return payload["tools"][0]["function"]["name"]


def test_recursive_model_tool_schema_keeps_root_properties() -> None:
    # A self-referential model is serialized with its body under ``$defs`` and a
    # single ``$ref`` at the root, so the resolved root has no ``title``.
    schema = generate_openai_schema(Node)

    assert schema["name"] == "Node"
    assert schema["description"] == "A tree node that can nest another node."
    assert schema["parameters"]["type"] == "object"
    assert list(schema["parameters"]["properties"]) == ["name", "child"]
    assert schema["parameters"]["required"] == ["name"]
    assert schema["parameters"]["properties"]["child"]["anyOf"][0] == {
        "$ref": "#/$defs/Node"
    }
    assert list(schema["parameters"]["$defs"]) == ["Node"]


def test_recursive_model_tools_request_is_built() -> None:
    requests: list[dict[str, Any]] = []

    def transport(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "model": "test",
                "created": 1,
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": _tool_names(payload),
                                        "arguments": json.dumps(
                                            {"name": "root", "child": None}
                                        ),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    sdk = OpenAI(
        api_key="test",
        http_client=httpx.Client(transport=httpx.MockTransport(transport)),
    )
    client = instructor.from_openai(sdk, mode=Mode.TOOLS)

    result = client.create(
        response_model=Node, model="test", messages=[{"role": "user", "content": "tree"}]
    )

    assert isinstance(result, Node)
    assert result.name == "root"
    assert result.child is None
    assert requests[0]["tools"][0]["function"]["parameters"]["required"] == ["name"]
    sdk.close()


def test_wrapped_recursive_model_schema_matches_base_model() -> None:
    # Non-recursive models must keep resolving to the same tool schema shape.
    class Plain(BaseModel):
        """Plain model."""

        name: str

    wrapped = instructor.openai_schema(Plain)
    assert generate_openai_schema(wrapped)["parameters"]["properties"]["name"]
    assert generate_anthropic_schema(Node)["input_schema"]["$ref"] == "#/$defs/Node"
