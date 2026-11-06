"""Small, SDK-validated OpenAI response builders for offline coverage tests."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageFunctionToolCall,
)

FinishReason = Literal[
    "stop", "length", "tool_calls", "content_filter", "function_call"
]


def tool_call(
    name: str,
    arguments: Mapping[str, Any] | str,
    call_id: str = "call_1",
) -> ChatCompletionMessageFunctionToolCall:
    serialized = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return ChatCompletionMessageFunctionToolCall.model_validate(
        {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": serialized},
        }
    )


def chat_completion(
    *,
    content: str | None = None,
    tool_calls: Sequence[ChatCompletionMessageFunctionToolCall] | None = None,
    function_call: tuple[str, str] | None = None,
    refusal: str | None = None,
    finish_reason: FinishReason | None = None,
    usage: bool = False,
) -> ChatCompletion:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = list(tool_calls)
    if function_call is not None:
        name, arguments = function_call
        message["function_call"] = {"name": name, "arguments": arguments}
    if refusal is not None:
        message["refusal"] = refusal

    completion: dict[str, Any] = {
        "id": "chatcmpl-coverage",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-test",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason
                if finish_reason is not None
                else "tool_calls"
                if tool_calls
                else "stop",
                "message": message,
            }
        ],
    }
    if usage:
        completion["usage"] = {
            "prompt_tokens": 8,
            "completion_tokens": 4,
            "total_tokens": 12,
        }
    return ChatCompletion.model_validate(completion)


def chat_chunk(
    delta: Mapping[str, Any],
    *,
    finish_reason: FinishReason | None = None,
) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-stream-coverage",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-test",
            "choices": [
                {"index": 0, "finish_reason": finish_reason, "delta": dict(delta)}
            ],
        }
    )


def tool_chunks(*parts: str, name: str = "User") -> list[ChatCompletionChunk]:
    return [
        chat_chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": f"call-{name.lower()}",
                        "type": "function",
                        "function": {"name": name, "arguments": part},
                    }
                ]
            },
            finish_reason="stop" if index == len(parts) - 1 else None,
        )
        for index, part in enumerate(parts)
    ]
