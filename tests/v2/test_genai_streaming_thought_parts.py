from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from pydantic import BaseModel

import instructor
from instructor import Mode

types = pytest.importorskip("google.genai.types")


class Person(BaseModel):
    name: str


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    "placement", ["none", "separate", "before", "after", "both", "no_parts"]
)
async def test_iterable_tool_call_with_thought_parts(
    is_async: bool, placement: str
) -> None:
    requests: list[dict[str, Any]] = []

    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            name = request["tools"][0]["functionDeclarations"][0]["name"]
            thought = {"thought": True, "text": "checking"}
            parts = [
                {"functionCall": {"name": name, "args": {"tasks": [{"name": "Ada"}]}}}
            ]
            if placement in ("before", "both"):
                parts.insert(0, thought)
            if placement in ("after", "both"):
                parts.append(thought)
            events = []
            if placement == "separate":
                events.append(
                    {"candidates": [{"content": {"role": "model", "parts": [thought]}}]}
                )
            events.append(
                {
                    "candidates": [
                        {
                            "index": 0,
                            "content": {"role": "model", "parts": parts},
                            "finishReason": "STOP",
                        }
                    ]
                }
            )
            if placement == "no_parts":
                events[0]["candidates"][0]["content"] = {"role": "model"}
            events.extend(
                [
                    {
                        "candidates": [
                            {
                                "content": {"role": "model", "parts": []},
                                "finishReason": "STOP",
                            }
                        ]
                    },
                    {"candidates": [{"finishReason": "STOP"}]},
                    {"candidates": []},
                ]
            )
            body = "".join(
                f"data: {json.dumps(event)}\n\n" for event in events
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    client: Any = instructor.from_provider(
        "google/gemini-2.5-flash",
        api_key="synthetic-local",
        mode=Mode.TOOLS,
        async_client=is_async,
        http_options=types.HttpOptions(
            base_url=f"http://127.0.0.1:{server.server_port}",
            client_args={"trust_env": False},
            async_client_args={"trust_env": False},
        ),
    )
    try:
        result: Any = client.create_iterable(
            messages=[{"role": "user", "content": "synthetic"}],
            response_model=Person,
            max_retries=0,
        )
        actual = [item async for item in result] if is_async else list(result)
        assert actual == ([] if placement == "no_parts" else [Person(name="Ada")])
        assert len(requests) == 1
    finally:
        if is_async:
            await client.client.aio.aclose()
        client.client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
