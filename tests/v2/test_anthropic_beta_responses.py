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
from instructor.v2.core.errors import IncompleteOutputException
from instructor.v2.core.hooks import Hooks

anthropic = pytest.importorskip("anthropic")


class Result(BaseModel):
    count: int


@pytest.mark.parametrize(
    "mode",
    [
        instructor.Mode.TOOLS,
        instructor.Mode.JSON,
        instructor.Mode.JSON_SCHEMA,
        instructor.Mode.PARALLEL_TOOLS,
    ],
)
@pytest.mark.parametrize("beta", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("truncated", [False, True])
def test_native_beta_response(
    mode: instructor.Mode,
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
            content = [{"type": "text", "text": '{"count":3}'}]
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
                response_model=Iterable[Result]
                if mode == instructor.Mode.PARALLEL_TOOLS
                else Result,
                max_retries=1,
                messages=[{"role": "user", "content": "Extract count3"}],
                hooks=hooks,
            )
            if asynchronous:
                value = await value
            return list(value) if mode == instructor.Mode.PARALLEL_TOOLS else value

        if truncated and mode != instructor.Mode.PARALLEL_TOOLS:
            with pytest.raises(IncompleteOutputException) as error:
                asyncio.run(invoke())
            assert error.value.last_completion is responses[0]
        else:
            value = asyncio.run(invoke())
            if mode == instructor.Mode.PARALLEL_TOOLS:
                # Parallel terminal handling is unchanged and outside this fix.
                assert value[0].count == 3
            else:
                assert value.count == 3
                assert value._raw_response is responses[0]
        assert len(requests) == len(responses) == 1
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
