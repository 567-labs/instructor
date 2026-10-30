from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import AsyncExitStack
from copy import deepcopy
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel, PrivateAttr, ValidationError

from instructor.core import retry_async, retry_sync
from instructor.processing import handle_reask_kwargs
from instructor.v2.core.errors import InstructorRetryException
from instructor.v2.core.messages import isolate_retry_kwargs
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider


class Answer(BaseModel):
    name: str
    age: int
    _total_usage: CompletionUsage = PrivateAttr()


def _completion(call_id: str, *, valid: bool) -> ChatCompletion:
    answer = {"name": "Ada", "age": 37} if valid else {"name": "Ada"}
    return ChatCompletion(
        id=f"chatcmpl-{call_id}",
        created=0,
        model="local",
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="tool_calls",
                message=ChatCompletionMessage(
                    role="assistant",
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id=call_id,
                            type="function",
                            function=Function(
                                name="Answer", arguments=json.dumps(answer)
                            ),
                        )
                    ],
                ),
            )
        ],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


@dataclass
class RetryEndpoint:
    url: str = ""
    responses: list[ChatCompletion] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)


@pytest.fixture
def retry_endpoint() -> Iterator[RetryEndpoint]:
    endpoint = RetryEndpoint()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            endpoint.calls.append(
                {
                    "path": self.path,
                    "body": json.loads(
                        self.rfile.read(int(self.headers["Content-Length"]))
                    ),
                }
            )
            if not endpoint.responses:
                self.send_error(500, "No response queued for this request")
                return
            body = endpoint.responses.pop(0).model_dump_json().encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002, ARG002
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint.url = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        yield endpoint
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("succeeds", [False, True], ids=["exhausted", "recovered"])
async def test_public_retries_isolate_reused_caller_history_over_http(
    retry_endpoint: RetryEndpoint, is_async: bool, succeeds: bool
) -> None:
    retry_endpoint.responses = [
        _completion(f"call_{index}", valid=succeeds and index % 2 == 0)
        for index in range(1, 5)
    ]
    messages = [
        {"role": "system", "content": "Extract the person."},
        {"role": "user", "content": [{"type": "text", "text": "Ada is 37 years old."}]},
    ]
    kwargs: dict[str, Any] = {
        "model": "local",
        "messages": messages,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "Answer",
                    "parameters": Answer.model_json_schema(),
                },
            }
        ],
    }
    snapshot = deepcopy(kwargs)
    failures: list[InstructorRetryException] = []
    async with AsyncExitStack() as stack:
        if is_async:
            client = await stack.enter_async_context(
                AsyncOpenAI(
                    api_key="local-test-key",
                    base_url=retry_endpoint.url,
                    max_retries=0,
                    http_client=httpx.AsyncClient(trust_env=False),
                )
            )
        else:
            client = stack.enter_context(
                OpenAI(
                    api_key="local-test-key",
                    base_url=retry_endpoint.url,
                    max_retries=0,
                    http_client=httpx.Client(trust_env=False),
                )
            )
        for _ in range(2):
            if succeeds:
                if is_async:
                    result = await retry_async(
                        client.chat.completions.create,
                        Answer,
                        (),
                        kwargs,
                        mode=Mode.TOOLS,
                        provider=Provider.OPENAI,
                        max_retries=1,
                    )
                else:
                    result = retry_sync(
                        client.chat.completions.create,
                        Answer,
                        (),
                        kwargs,
                        mode=Mode.TOOLS,
                        provider=Provider.OPENAI,
                        max_retries=1,
                    )
                assert isinstance(result, Answer)
                assert result.model_dump() == {"name": "Ada", "age": 37}
                assert result._total_usage.total_tokens == 30
            else:
                with pytest.raises(InstructorRetryException) as failure:
                    if is_async:
                        await retry_async(
                            client.chat.completions.create,
                            Answer,
                            (),
                            kwargs,
                            mode=Mode.TOOLS,
                            provider=Provider.OPENAI,
                            max_retries=1,
                        )
                    else:
                        retry_sync(
                            client.chat.completions.create,
                            Answer,
                            (),
                            kwargs,
                            mode=Mode.TOOLS,
                            provider=Provider.OPENAI,
                            max_retries=1,
                        )
                failures.append(failure.value)
                assert failure.value.n_attempts == 2
                assert failure.value.failed_attempts is not None
                assert len(failure.value.failed_attempts) == 2
                assert failure.value.create_kwargs is not None
                assert failure.value.create_kwargs["messages"] is not messages
                assert len(failure.value.create_kwargs["messages"]) == len(messages) + 4
            assert kwargs == snapshot
            assert kwargs["messages"] is messages

    assert len(retry_endpoint.calls) == 4
    assert retry_endpoint.responses == []
    for index, call in enumerate(retry_endpoint.calls):
        assert call["path"] == "/v1/chat/completions"
        body = call["body"]
        assert body["tools"] == snapshot["tools"]
        if index % 2 == 0:
            assert body["messages"] == snapshot["messages"]
        else:
            assert body["messages"][:2] == snapshot["messages"]
            assert len(body["messages"]) == 4
            assistant, feedback = body["messages"][-2:]
            assert assistant["role"] == "assistant"
            assert assistant["tool_calls"][0]["id"] == f"call_{index}"
            assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {
                "name": "Ada"
            }
            assert feedback["role"] == "tool"
            assert feedback["tool_call_id"] == f"call_{index}"
            assert "age" in feedback["content"]
            assert "Field required" in feedback["content"]
    if failures:
        assert failures[0].create_kwargs is not None
        assert failures[1].create_kwargs is not None
        assert (
            failures[0].create_kwargs["messages"]
            is not failures[1].create_kwargs["messages"]
        )


def test_handle_reask_isolates_all_history_lists_with_real_sdk_response() -> None:
    response = _completion("call_missing_age", valid=False)
    tool_calls = response.choices[0].message.tool_calls
    assert tool_calls is not None
    assert tool_calls[0].type == "function"
    with pytest.raises(ValidationError) as failure:
        Answer.model_validate_json(tool_calls[0].function.arguments)
    messages = [{"role": "user", "content": "Ada is 37 years old."}]
    contents = [{"role": "user", "parts": [{"text": "Original contents"}]}]
    chat_history = [{"role": "USER", "message": "Original chat history"}]
    kwargs = {"messages": messages, "contents": contents, "chat_history": chat_history}
    snapshot = deepcopy(kwargs)

    reasked = handle_reask_kwargs(
        kwargs, Mode.TOOLS, response, failure.value, provider=Provider.OPENAI
    )

    assert kwargs == snapshot
    assert reasked["messages"] is not messages
    assert reasked["contents"] is not contents
    assert reasked["chat_history"] is not chat_history
    assert reasked["messages"][:1] == messages
    assert len(reasked["messages"]) == 3
    assert reasked["messages"][-1]["tool_call_id"] == "call_missing_age"
    assert "age" in reasked["messages"][-1]["content"]
    reasked["contents"].append({"role": "model", "parts": []})
    reasked["chat_history"].append({"role": "CHATBOT", "message": "Retry"})
    assert kwargs == snapshot


def test_isolate_retry_kwargs_copies_only_mutable_history_containers() -> None:
    messages = [{"role": "user", "content": "Original message"}]
    contents = ["Original contents"]
    chat_history = [{"role": "USER", "message": "Original chat history"}]
    tools = [{"name": "Answer"}]
    kwargs = {
        "messages": messages,
        "contents": contents,
        "chat_history": chat_history,
        "tools": tools,
    }
    isolated = isolate_retry_kwargs(kwargs)

    isolated["messages"].clear()
    isolated["contents"].append("Retry contents")
    isolated["chat_history"].append({"role": "CHATBOT", "message": "Retry"})
    assert messages == [{"role": "user", "content": "Original message"}]
    assert contents == ["Original contents"]
    assert chat_history == [{"role": "USER", "message": "Original chat history"}]
    assert isolated["tools"] is tools
    assert isolated is not kwargs
