"""Writer terminal metadata must win over otherwise valid JSON content."""

import json
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any

import pytest
from pydantic import BaseModel

import instructor
from instructor.v2.core.errors import IncompleteOutputException
from instructor.v2.core.client import Instructor, AsyncInstructor

pytest.importorskip("writerai")


class Result(BaseModel):
    count: int


@pytest.fixture
def writer_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[tuple[dict[str, str], list[dict[str, Any]]], None, None]:
    requests: list[dict[str, Any]] = []
    state = {"finish_reason": "stop", "content": '{"count":3}'}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            message: dict[str, Any] = {"role": "assistant", "content": state["content"]}
            if "tools" in request:
                message["tool_calls"] = [
                    {
                        "id": "call1",
                        "type": "function",
                        "function": {"name": "Result", "arguments": state["content"]},
                    }
                ]
            body = json.dumps(
                {
                    "id": "local",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "palmyra-x5",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": state["finish_reason"],
                            "message": message,
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
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("WRITER_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(key, raising=False)
    try:
        yield state, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.asyncio
@pytest.mark.parametrize("async_client", [False, True])
@pytest.mark.parametrize(
    "mode",
    [instructor.Mode.TOOLS, instructor.Mode.MD_JSON, instructor.Mode.JSON_SCHEMA],
)
@pytest.mark.parametrize(
    "finish_reason,content",
    [
        ("length", '{"count":3}'),
        ("length", '{"count":'),
        ("length", ""),
        ("stop", '{"count":3}'),
    ],
)
async def test_writer_json_completion_respects_token_limit(
    writer_endpoint: tuple[dict[str, str], list[dict[str, Any]]],
    async_client: bool,
    mode: instructor.Mode,
    finish_reason: str,
    content: str,
) -> None:
    state, requests = writer_endpoint
    state["finish_reason"] = finish_reason
    state["content"] = content
    if mode == instructor.Mode.MD_JSON and content == '{"count":3}':
        state["content"] = f"```json\n{content}\n```"
    client = instructor.from_provider(
        "writer/palmyra-x5",
        mode=mode,
        api_key="local-not-secret",
        async_client=async_client,
    )

    assert isinstance(client, (Instructor, AsyncInstructor))

    async def create() -> Result:
        if isinstance(client, AsyncInstructor):
            return await client.create(
                response_model=Result,
                messages=[{"role": "user", "content": "Extract counts"}],
                max_retries=3,
            )
        return client.create(
            response_model=Result,
            messages=[{"role": "user", "content": "Extract counts"}],
            max_retries=3,
        )

    try:
        if finish_reason == "length":
            with pytest.raises(IncompleteOutputException) as error:
                await create()
            assert error.value.last_completion is not None
            assert error.value.last_completion.choices[0].finish_reason == "length"
        else:
            assert (await create()).count == 3
        assert len(requests) == 1
    finally:
        assert client.client is not None
        if async_client:
            await client.client.close()
        else:
            client.client.close()
