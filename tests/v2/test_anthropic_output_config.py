"""Native Anthropic transport coverage for the stable structured-output API."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import inspect
import json
from threading import Thread
from typing import Any, cast
import warnings

import pytest
from pydantic import BaseModel
import instructor

anthropic = pytest.importorskip("anthropic")


class Result(BaseModel):
    count: int


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "config",
    [
        None,
        anthropic.omit,
        {"effort": "low"},
        {
            "effort": "low",
            "format": {"type": "json_schema", "schema": {"type": "string"}},
        },
    ],
)
@pytest.mark.parametrize("native", [False, True])
def test_output_config_native_transport(
    asynchronous: bool,
    config: Any,
    native: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if (
        "output_config"
        not in inspect.signature(
            anthropic.resources.messages.Messages.create
        ).parameters
    ):
        pytest.skip("The installed SDK does not expose stable output_config")
    requests: list[dict[str, Any]] = []
    beta_headers: list[str | None] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(req)
            beta_headers.append(self.headers.get("anthropic-beta"))
            value = "bad" if len(requests) % 2 else 3
            body = json.dumps(
                {
                    "id": "msg_local",
                    "type": "message",
                    "role": "assistant",
                    "model": req["model"],
                    "content": [{"type": "text", "text": json.dumps({"count": value})}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
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
    kwargs: dict[str, Any] = (
        {"output_config": deepcopy(config)} if config is not None else {}
    )
    original = deepcopy(kwargs)
    if isinstance(config, anthropic.Omit):
        kwargs["output_config"] = config
        original["output_config"] = config
    client: Any = None
    try:
        client = cast(
            Any,
            instructor.from_provider(
                "anthropic/claude-local",
                api_key="local-not-secret",
                mode=instructor.Mode.JSON_SCHEMA if native else instructor.Mode.JSON,
                async_client=asynchronous,
            ),
        )

        async def call() -> None:
            for _ in range(2):
                result = client.create(
                    response_model=Result,
                    max_retries=1,
                    messages=[{"role": "user", "content": "Count is three."}],
                    **kwargs,
                )
                if asynchronous:
                    result = await result
                assert result.count == 3
                assert kwargs == original

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            asyncio.run(call())
        assert not any("falling back" in str(w.message) for w in caught)
        assert len(requests) == 4
        assert requests[0] == requests[2]
        assert beta_headers == [None] * 4
        for req in requests:
            assert "output_format" not in req and "betas" not in req
            if native:
                assert req["output_config"]["format"] == {
                    "type": "json_schema",
                    "schema": anthropic.transform_schema(Result),
                }
                assert req["output_config"].get("effort") == (config or {}).get(
                    "effort"
                )
            else:
                assert req.get("output_config") == (
                    None if isinstance(config, anthropic.Omit) else config
                )
    finally:
        if client is not None:
            if asynchronous:
                asyncio.run(client.client.close())
            else:
                client.client.close()
        server.shutdown()
        server.server_close()
        thread.join()
