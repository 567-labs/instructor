from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel

import instructor
from instructor import Mode
from instructor.batch import BatchProcessor


class Contact(BaseModel):
    name: str
    nickname: str | None = None


def capture(requests: list[dict[str, Any]]):
    def transport(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        arguments = json.dumps({"name": "Ada", "nickname": None})
        message: dict[str, Any] = {"role": "assistant", "content": arguments}
        if "tools" in payload:
            message["tool_calls"] = [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "Contact", "arguments": arguments},
                }
            ]
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "model": "test",
                "created": 1,
                "object": "chat.completion",
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            },
        )

    return transport


@pytest.mark.parametrize("mode", [Mode.TOOLS, Mode.JSON_SCHEMA])
@pytest.mark.parametrize("explicit", [None, True, False])
def test_public_strict_wire_format(mode: Mode, explicit: bool | None) -> None:
    requests: list[dict[str, Any]] = []
    sdk = OpenAI(
        api_key="test",
        http_client=httpx.Client(transport=httpx.MockTransport(capture(requests))),
    )
    client = instructor.from_openai(sdk, mode=mode)
    options: dict[str, Any] = {} if explicit is None else {"strict": explicit}
    result = client.create(response_model=Contact, model="test", messages=[], **options)
    assert result.name == "Ada"
    payload = requests[0]
    assert "strict" not in payload
    definition = (
        payload["tools"][0]["function"]
        if mode == Mode.TOOLS
        else payload["response_format"]["json_schema"]
    )
    strict = (mode == Mode.JSON_SCHEMA) if explicit is None else explicit
    schema = definition.get("parameters", definition.get("schema"))
    assert definition.get("strict", False) is strict
    assert schema["required"] == (["name", "nickname"] if strict else ["name"])
    sdk.close()


@pytest.mark.asyncio
async def test_async_strict_tools_preserve_cached_permissive_schema() -> None:
    requests: list[dict[str, Any]] = []
    sdk = AsyncOpenAI(
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(capture(requests))),
    )
    client = instructor.from_openai(sdk, mode=Mode.TOOLS)
    calls: list[dict[str, Any]] = [{"strict": True}, {}]
    for options in calls:
        result = await client.create(
            response_model=Contact, model="test", messages=[], **options
        )
        assert result.name == "Ada"
    strict, default = [r["tools"][0]["function"] for r in requests]
    assert strict["strict"] is True
    assert strict["parameters"]["additionalProperties"] is False
    assert "strict" not in default
    assert "additionalProperties" not in default["parameters"]
    await sdk.close()


@pytest.mark.parametrize("write_file", [False, True])
def test_permissive_batch_preserves_mapping_schema(
    tmp_path: Path, write_file: bool
) -> None:
    class Stock(BaseModel):
        counts: dict[str, int]

    batch = BatchProcessor("openai/test", Stock)
    path = str(tmp_path / "batch.jsonl") if write_file else None
    output = batch.create_batch_from_messages([[]], path, strict=False)
    text = (
        output.getvalue()
        if isinstance(output, io.BytesIO)
        else Path(output).read_bytes()
    )
    definition = json.loads(text)["body"]["response_format"]["json_schema"]
    assert definition["strict"] is False
    assert definition["schema"]["properties"]["counts"]["additionalProperties"] == {
        "type": "integer"
    }
