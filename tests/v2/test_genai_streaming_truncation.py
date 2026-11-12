"""Exercise GenAI terminal metadata through real SDK HTTP streaming."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any

import pytest
from pydantic import BaseModel

import instructor
from instructor import Mode
from instructor.v2.core.errors import IncompleteOutputException
from instructor.v2.core.client import Instructor, AsyncInstructor

types = pytest.importorskip("google.genai.types")


class Person(BaseModel):
    name: str


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("mode", [Mode.TOOLS, Mode.JSON])
@pytest.mark.parametrize("partial_model", [False, True])
@pytest.mark.parametrize("finish_reason", ["MAX_TOKENS", "STOP"])
@pytest.mark.parametrize(
    "terminal_content", ["same", "same_no_usage", "missing", "no_parts", "empty_parts"]
)
async def test_genai_stream_completion_metadata(
    is_async: bool,
    mode: Mode,
    partial_model: bool,
    finish_reason: str,
    terminal_content: str,
) -> None:
    requests: list[dict[str, Any]] = []

    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            payload: dict[str, Any] = {"name": "Ada"}
            if not partial_model:
                payload = {"tasks": [payload]}
            if mode == Mode.TOOLS:
                name = request["tools"][0]["functionDeclarations"][0]["name"]
                parts = [{"functionCall": {"name": name, "args": payload}}]
            else:
                parts = [{"text": json.dumps(payload)}]
            candidate: dict[str, Any] = {
                "index": 0,
                "content": {"role": "model", "parts": parts},
            }
            events: list[dict[str, Any]] = [{"candidates": [candidate]}]
            if terminal_content in ("same", "same_no_usage"):
                candidate["finishReason"] = finish_reason
            else:
                terminal: dict[str, Any] = {"finishReason": finish_reason}
                if terminal_content == "no_parts":
                    terminal["content"] = {"role": "model"}
                elif terminal_content == "empty_parts":
                    terminal["content"] = {"role": "model", "parts": []}
                events.append({"candidates": [terminal]})
            # Empty candidate/usage chunks must neither erase nor invent a limit.
            if terminal_content != "same_no_usage" and mode == Mode.TOOLS:
                events.append(
                    {"candidates": [], "usageMetadata": {"totalTokenCount": 10}}
                )
            if terminal_content != "same_no_usage":
                events.append({"usageMetadata": {"totalTokenCount": 10}})
            body = "".join(
                f"data: {json.dumps(event)}\n\n" for event in events
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    thread = Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    client: Any = None
    actual: list[Any] = []
    try:
        client = instructor.from_provider(
            "google/gemini-2.5-flash",
            api_key="synthetic-local",
            mode=mode,
            async_client=is_async,
            http_options=types.HttpOptions(
                base_url=f"http://127.0.0.1:{server.server_port}",
                client_args={"trust_env": False},
                async_client_args={"trust_env": False},
            ),
        )
        assert isinstance(client, (Instructor, AsyncInstructor))
        create = client.create_partial if partial_model else client.create_iterable
        stream: Any = create(
            messages=[{"role": "user", "content": "Extract all people"}],
            response_model=Person,
            max_retries=3,
        )

        async def consume() -> None:
            if is_async:
                async for item in stream:
                    actual.append(item)
            else:
                actual.extend(stream)

        if finish_reason == "MAX_TOKENS" and not partial_model:
            with pytest.raises(IncompleteOutputException) as error:
                await consume()
            completion = error.value.last_completion
            assert isinstance(completion, types.GenerateContentResponse)
            assert completion.candidates
            assert (
                completion.candidates[0].finish_reason == types.FinishReason.MAX_TOKENS
            )
        else:
            await consume()
        assert actual
        assert actual[-1].name == "Ada"
        if not partial_model:
            assert actual == [Person(name="Ada")]
        assert len(requests) == 1
    finally:
        if isinstance(client, (Instructor, AsyncInstructor)):
            assert client.client is not None
            if is_async:
                await client.client.aio.aclose()
            client.client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
