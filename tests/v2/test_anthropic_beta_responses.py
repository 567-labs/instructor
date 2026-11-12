"""Stable and beta Anthropic responses through native localhost HTTP."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
from typing import Any, cast

import pytest
from pydantic import BaseModel
import instructor
from instructor.v2.core.errors import (
    IncompleteOutputException,
    InstructorRetryException,
)
from instructor.v2.core.hooks import Hooks

anthropic = pytest.importorskip("anthropic")


class Result(BaseModel):
    count: int


@pytest.mark.parametrize(
    "mode, payload",
    [
        (instructor.Mode.TOOLS, "normal"),
        (instructor.Mode.JSON, "normal"),
        (instructor.Mode.JSON_SCHEMA, "normal"),
        (instructor.Mode.PARALLEL_TOOLS, "normal"),
        (instructor.Mode.PARALLEL_TOOLS, "empty"),
        (instructor.Mode.PARALLEL_TOOLS, "invalid"),
        (instructor.Mode.PARALLEL_TOOLS, "raw"),
    ],
)
@pytest.mark.parametrize("beta", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("truncated", [False, True])
def test_native_beta_response(
    mode: instructor.Mode,
    payload: str,
    beta: bool,
    asynchronous: bool,
    truncated: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []
    responses: list[Any] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(req)
            content: list[dict[str, Any]] = [{"type": "text", "text": '{"count":3}'}]
            stop = "end_turn"
            if mode in (instructor.Mode.TOOLS, instructor.Mode.PARALLEL_TOOLS):
                content = [
                    {
                        "type": "tool_use",
                        "id": "toolu_local",
                        "name": "Result",
                        "input": {"count": 3},
                    }
                ]
                stop = "tool_use"
            if payload == "empty":
                content = []
            elif payload == "invalid":
                # A valid first item must not escape before the second fails.
                content.append(
                    {
                        "type": "tool_use",
                        "id": "toolu_invalid",
                        "name": "Result",
                        "input": {"count": "bad"},
                    }
                )
            body = json.dumps(
                {
                    "id": "msg_local",
                    "type": "message",
                    "role": "assistant",
                    "model": req["model"],
                    "content": content,
                    "stop_reason": "max_tokens" if truncated else stop,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "context_management": {"applied_edits": []},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    client: Any = None
    try:
        client = cast(
            Any,
            instructor.from_provider(
                "anthropic/claude-local",
                api_key="local-not-secret",
                mode=mode,
                beta=beta,
                async_client=asynchronous,
            ),
        )
        hooks = Hooks()
        hooks.on("completion:response", responses.append)

        async def invoke() -> Any:
            value = client.create(
                response_model=None
                if payload == "raw"
                else (
                    Iterable[Result]
                    if mode == instructor.Mode.PARALLEL_TOOLS
                    else Result
                ),
                max_retries=2,
                messages=[{"role": "user", "content": "Extract count3"}],
                hooks=hooks,
            )
            if asynchronous:
                value = await value
            return (
                list(value)
                if mode == instructor.Mode.PARALLEL_TOOLS and payload != "raw"
                else value
            )

        if payload == "raw":
            value = asyncio.run(invoke())
            assert value.stop_reason == ("max_tokens" if truncated else "tool_use")
            assert responses == []
            responses.append(value)
        elif truncated:
            with pytest.raises(IncompleteOutputException) as error:
                asyncio.run(invoke())
            assert error.value.last_completion is responses[0]
        elif payload == "invalid":
            with pytest.raises(InstructorRetryException):
                asyncio.run(invoke())
        else:
            value = asyncio.run(invoke())
            if payload == "empty":
                assert value == []
            elif mode == instructor.Mode.PARALLEL_TOOLS:
                assert value[0].count == 3
            else:
                assert value.count == 3
                assert value._raw_response is responses[0]
        assert (
            len(requests)
            == len(responses)
            == (3 if payload == "invalid" and not truncated else 1)
        )
        if mode == instructor.Mode.PARALLEL_TOOLS:
            if payload == "raw":
                assert "tools" not in requests[0]
            else:
                assert [tool["name"] for tool in requests[0]["tools"]] == ["Result"]
                assert requests[0]["tool_choice"] == {"type": "auto"}
        expected = anthropic.types.beta.BetaMessage if beta else anthropic.types.Message
        assert isinstance(responses[0], expected)
        if beta:
            assert responses[0].context_management.applied_edits == []
    finally:
        if client is not None:
            if asynchronous:
                asyncio.run(client.client.close())
            else:
                client.client.close()
        server.shutdown()
        server.server_close()
        thread.join()
