"""Legacy batch parsing rejects token-limited records without changing its tuple API."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from instructor.batch import BatchJob


class Count(BaseModel):
    count: int


def native_record(provider: str, tool: bool, limited: bool) -> dict[str, Any]:
    payload = {"count": 3}
    if provider == "anthropic":
        pytest.importorskip("anthropic")
        from anthropic.types.messages import MessageBatchIndividualResponse

        block = (
            {"type": "tool_use", "id": "tool_1", "name": "Count", "input": payload}
            if tool
            else {"type": "text", "text": json.dumps(payload)}
        )
        return MessageBatchIndividualResponse.model_validate(
            {
                "custom_id": "target",
                "result": {
                    "type": "succeeded",
                    "message": {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "model": "local",
                        "content": [block],
                        "stop_reason": "max_tokens" if limited else "end_turn",
                        "stop_sequence": None,
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    },
                },
            }
        ).model_dump(mode="json", exclude_none=True)

    from openai.types.chat import ChatCompletion

    response_type: Any = ChatCompletion
    if provider == "mistral":
        pytest.importorskip("mistralai")
        from mistralai.client.models import ChatCompletionResponse

        response_type: Any = ChatCompletionResponse
    message: dict[str, Any] = {"role": "assistant"}
    if tool:
        message["tool_calls"] = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "Count", "arguments": json.dumps(payload)},
            }
        ]
    else:
        message["content"] = json.dumps(payload)
    body = response_type.model_validate(
        {
            "id": "msg_1",
            "object": "chat.completion",
            "created": 1,
            "model": "local",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "length" if limited else "stop",
                    "message": message,
                }
            ],
        }
    ).model_dump(mode="json", exclude_none=True, by_alias=True)
    return {"custom_id": "target", "response": {"status_code": 200, "body": body}}


@pytest.mark.parametrize("provider", ["openai", "anthropic", "mistral"])
@pytest.mark.parametrize("tool", [False, True])
@pytest.mark.parametrize("limited", [False, True])
@pytest.mark.parametrize("file_backed", [False, True])
def test_batch_job_token_limit(
    provider: str, tool: bool, limited: bool, file_backed: bool, tmp_path: Path
) -> None:
    target = native_record(provider, tool, limited)
    normal = native_record(provider, tool, False)
    normal["custom_id"] = "normal"
    # Missing stop metadata remains accepted for backward compatibility.
    if provider == "anthropic":
        normal["result"]["message"].pop("stop_reason")
    else:
        normal["response"]["body"]["choices"][0].pop("finish_reason")
    provider_error = {"custom_id": "failed", "error": {"message": "request failed"}}
    invalid = native_record(provider, False, False)
    invalid["custom_id"] = "invalid"
    if provider == "anthropic":
        invalid["result"]["message"]["content"][0]["text"] = "{bad"
    else:
        invalid["response"]["body"]["choices"][0]["message"]["content"] = "{bad"
    content = (
        "\n".join(json.dumps(r) for r in [target, provider_error, invalid, normal])
        + "\n{broken\n"
    )
    if file_backed:
        path = tmp_path / "results.jsonl"
        path.write_text(content, encoding="utf-8")
        result = BatchJob.parse_from_file(str(path), Count)
    else:
        result = BatchJob.parse_from_string(content, Count)
    assert isinstance(result, tuple) and len(result) == 2
    parsed, errors = result
    assert parsed == [Count(count=3)] * (1 if limited else 2)
    assert errors == ([target] if limited else []) + [
        provider_error,
        invalid,
        {"error": "Failed to parse JSON", "raw_line": "{broken"},
    ]
