"""Preserve supported assistant histories through public JSON-mode requests."""

import asyncio
import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from instructor import Mode, from_provider


@pytest.mark.parametrize("async_client", [False, True])
@pytest.mark.parametrize(
    "kind", ["legacy_omitted", "legacy_null", "modern_omitted", "modern_null", "plain"]
)
def test_assistant_history_preserves_content(kind: str, async_client: bool) -> None:
    assistant: dict = {"role": "assistant"}
    messages: list[dict] = [{"role": "user", "content": "Look up the prior value."}]
    if kind.startswith("legacy"):
        assistant["function_call"] = {"name": "lookup", "arguments": '{"key":"prior"}'}
        result = {"role": "function", "name": "lookup", "content": "prior value"}
    elif kind.startswith("modern"):
        assistant["tool_calls"] = [
            {
                "id": "prior-call",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"key":"prior"}'},
            }
        ]
        result = {
            "role": "tool",
            "tool_call_id": "prior-call",
            "content": "prior value",
        }
    else:
        assistant["content"] = "prior value"
        result = None
    if kind.endswith("null"):
        assistant["content"] = None
    messages.append(assistant)
    if result is not None:
        messages.append(result)
    messages.append({"role": "user", "content": "Return a JSON object."})
    original = copy.deepcopy(messages)
    requests: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_POST(self) -> None:
            requests.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            body = json.dumps(
                {
                    "id": "chatcmpl-local",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "local",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": '{"ok":true}'},
                        }
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client: Any = from_provider(
        "openai/local",
        mode=Mode.JSON,
        async_client=async_client,
        api_key="local-synthetic-key",
        base_url=f"http://127.0.0.1:{server.server_port}",
    )
    kwargs: dict[str, Any] = {
        "response_model": None,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "max_retries": 1,
    }

    async def request_async() -> None:
        try:
            response = await client.chat.completions.create(**kwargs)
            assert response.choices[0].message.content == '{"ok":true}'
        finally:
            await client.client.close()

    try:
        if async_client:
            asyncio.run(request_async())
        else:
            try:
                response = client.chat.completions.create(**kwargs)
                assert response.choices[0].message.content == '{"ok":true}'
            finally:
                client.client.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert len(requests) == 1
    assert requests[0]["messages"] == original
    assert messages == original
