"""Exercise Bedrock completeness through boto3's real localhost transport."""

from __future__ import annotations

import asyncio
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any, cast

import pytest
from pydantic import BaseModel

import instructor
from instructor.v2.core.errors import IncompleteOutputException

pytest.importorskip("boto3")


class Result(BaseModel):
    count: int


@pytest.mark.parametrize(
    "mode",
    [
        instructor.Mode.TOOLS,
        instructor.Mode.TOOLS_STRICT,
        instructor.Mode.MD_JSON,
        instructor.Mode.JSON_SCHEMA,
    ],
)
@pytest.mark.parametrize("async_client", [False, True])
@pytest.mark.parametrize(
    "truncated,payload,raw",
    [
        (False, "valid", False),
        (True, "valid", False),
        (True, "malformed", False),
        (True, "empty", False),
        (True, "valid", True),
        (False, "valid", True),
    ],
)
def test_bedrock_token_limit(
    mode: instructor.Mode,
    async_client: bool,
    truncated: bool,
    payload: str,
    raw: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            content: list[dict[str, Any]] = [{"text": '{"count": 3}'}]
            stop = "end_turn"
            if "toolConfig" in request:
                content = [
                    {
                        "toolUse": {
                            "toolUseId": "local",
                            "name": "Result",
                            "input": {"count": 3},
                        }
                    }
                ]
                stop = "tool_use"
            if payload == "empty":
                content = []
            elif payload == "malformed":
                if "toolConfig" in request:
                    cast(dict[str, Any], content[0]["toolUse"])["input"] = {
                        "count": "invalid"
                    }
                else:
                    content = [{"text": '{"count":'}]
            response = {
                "output": {"message": {"role": "assistant", "content": content}},
                "stopReason": "max_tokens" if truncated else stop,
                "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
                "metrics": {"latencyMs": 1},
            }
            responses.append(response)
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv(
        "AWS_ENDPOINT_URL_BEDROCK_RUNTIME", f"http://127.0.0.1:{server.server_port}"
    )
    monkeypatch.setenv("AWS_CONFIG_FILE", os.devnull)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", os.devnull)
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    client: Any = None
    try:
        client = cast(
            Any,
            instructor.from_provider(
                "bedrock/anthropic.claude-local",
                mode=mode,
                async_client=async_client,
                region="us-east-1",
                aws_access_key_id="local-not-secret",
                aws_secret_access_key="local-not-secret",
                aws_session_token="local-not-secret",
            ),
        )

        def invoke() -> Any:
            value = client.create(
                response_model=None if raw else Result,
                max_retries=2,
                messages=[{"role": "user", "content": "Extract the count."}],
            )
            return asyncio.run(value) if async_client else value

        if raw:
            result = invoke()
            assert result["output"] == responses[0]["output"]
            assert result["stopReason"] == responses[0]["stopReason"]
        elif truncated:
            with pytest.raises(IncompleteOutputException) as exc:
                invoke()
            assert exc.value.last_completion is not None
            assert exc.value.last_completion["stopReason"] == "max_tokens"
            assert exc.value.last_completion["output"] == responses[0]["output"]
        else:
            assert invoke().count == 3
        assert len(requests) == 1
    finally:
        if client is not None:
            client.client.close()
        server.shutdown()
        server.server_close()
        thread.join()
