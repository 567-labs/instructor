import asyncio, json, os
import pytest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from pydantic import BaseModel
import instructor


class Item(BaseModel):
    value: int


requests = []


class Endpoint(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        q = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        requests.append(q)
        value = {"value": "bad" if len(requests) == 1 else 7}
        msg = {"role": "assistant", "content": json.dumps(value)}
        if "tools" in q:
            msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_local",
                        "type": "function",
                        "function": {"name": "Item", "arguments": json.dumps(value)},
                    }
                ],
            }
        result = {
            "id": "local",
            "object": "chat.completion",
            "created": 0,
            "model": "local",
            "choices": [
                {
                    "index": 0,
                    "message": msg,
                    "finish_reason": "tool_calls" if "tools" in q else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "queue_time": 0.01,
                "prompt_time": 0.02,
                "completion_time": 0.03,
                "total_time": 0.06,
            },
            "x_groq": {"id": "req_local"},
        }
        if getattr(self.server, "usage_gap", False) and len(requests) == 1:
            result["usage"] = None
        b = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


def dump(x):
    return x.model_dump() if hasattr(x, "model_dump") else x


@pytest.mark.parametrize("gap", [False, True])
@pytest.mark.parametrize("provider", ["openai", "groq"])
@pytest.mark.parametrize("async_", [False, True])
@pytest.mark.parametrize("mode", [instructor.Mode.TOOLS, instructor.Mode.MD_JSON])
@pytest.mark.parametrize("budget", [None, 100, 10])
def test_native_public_retry_budget(provider, async_, mode, budget, gap, monkeypatch):
    if provider == "groq":
        pytest.importorskip("groq")
    for k in list(os.environ):
        if k.lower() in ["http_proxy", "https_proxy", "all_proxy"]:
            monkeypatch.delenv(k)
    s = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    s.usage_gap = gap
    t = Thread(target=s.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    t.start()
    url = f"http://127.0.0.1:{s.server_port}"
    monkeypatch.setenv("GROQ_BASE_URL", url)
    monkeypatch.setenv("OPENAI_BASE_URL", url)
    requests.clear()

    async def call():
        client = instructor.from_provider(
            provider + "/local",
            api_key="local-not-secret",
            mode=mode,
            async_client=async_,
        )
        try:
            try:
                result = client.create(
                    response_model=Item,
                    messages=[{"role": "user", "content": "value"}],
                    max_retries=2,
                    token_budget=budget,
                )
                if async_:
                    result = await result
            except Exception as exc:
                assert budget is not None
                if gap:
                    assert type(exc).__name__ == "TokenUsageUnavailableError"
                else:
                    assert budget == 10
                    assert type(exc).__name__ == "TokenBudgetExceeded"
                    assert exc.total_usage.total_tokens == 15
                assert len(requests) == 1
            else:
                assert budget != 10
                assert result.value == 7
                if gap:
                    assert not hasattr(result, "_total_usage")
                    assert len(requests) == 2
                    return
                assert result._total_usage.total_tokens == 30
                assert result._total_usage.prompt_tokens == 20
                assert result._total_usage.completion_tokens == 10
                assert len(requests) == 2
                if provider == "groq":
                    assert type(result._total_usage).__module__.startswith("groq.")
                    assert result._total_usage.queue_time == 0.01
                    assert dump(result._raw_response.x_groq)["id"] == "req_local"
        finally:
            if async_:
                await client.client.close()
            else:
                client.client.close()

    try:
        asyncio.run(call())
    finally:
        s.shutdown()
        s.server_close()
        t.join()
