"""Known tasks wrappers must not turn ignored metadata into extra models."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
from typing import Any, cast

import pytest
from pydantic import BaseModel

import instructor
from instructor.v2.dsl.iterable import IterableModel, _ToolCallStart


class Item(BaseModel):
    name: str


PAYLOADS = {
    "object": {"tasks": [{"name": "real"}], "metadata": {"name": "phantom"}},
    "array": {
        "tasks": [{"name": "real"}],
        "metadata": [{"name": "phantom"}, {"nested": [{"name": "other"}]}],
    },
    "empty": {"tasks": [], "metadata": {"name": "phantom"}},
    "string": {
        "tasks": [{"name": "real"}],
        "metadata": 'quoted "tasks": [{"name":"phantom"}] and } \\"',
    },
}


@pytest.mark.parametrize("payload", PAYLOADS.values(), ids=PAYLOADS.keys())
def test_wrapper_metadata_at_every_split(payload: dict[str, Any]) -> None:
    model = cast(Any, IterableModel(Item))
    text = json.dumps(payload)
    expected = model.model_validate(payload).tasks
    assert model.model_json_schema().get("additionalProperties") is not False
    for split in range(len(text) + 1):
        assert list(model.tasks_from_chunks([text[:split], text[split:]])) == expected
    assert list(model.tasks_from_chunks(iter(text))) == expected


def test_wrapper_tail_does_not_hide_later_roots_or_tool_calls() -> None:
    model = cast(Any, IterableModel(Item))
    first = json.dumps(PAYLOADS["object"])
    bare = '{"name":"bare"}'
    wrapped = '{"tasks":[{"name":"next"}],"metadata":{"name":"wrong"}}'
    expected = [Item(name="real"), Item(name="bare"), Item(name="next")]
    assert list(model.tasks_from_chunks(iter(first + bare + wrapped))) == expected
    assert (
        list(
            model.tasks_from_chunks(
                [first, _ToolCallStart(bare), _ToolCallStart(wrapped)]
            )
        )
        == expected
    )


def test_task_objects_keep_their_own_tasks_fields() -> None:
    class Group(BaseModel):
        name: str
        tasks: list[Item]

    model = cast(Any, IterableModel(Group))
    item = {"name": "group", "tasks": [{"name": "child"}]}
    expected = [Group.model_validate(item)]
    assert list(model.tasks_from_chunks(iter(json.dumps(item)))) == expected
    assert (
        list(
            model.tasks_from_chunks(
                iter(json.dumps({"tasks": [item], "metadata": {"name": "ignored"}}))
            )
        )
        == expected
    )


def test_wrapper_yields_before_the_tail_arrives() -> None:
    model = cast(Any, IterableModel(Item))
    may_read_tail = False

    def chunks() -> Iterator[str]:
        yield '{"tasks":[{"name":"real"}'
        assert may_read_tail, "complete tasks must not wait for wrapper metadata"
        yield '],"metadata":{"name":"phantom"}}'

    tasks = model.tasks_from_chunks(chunks())
    assert next(tasks) == Item(name="real")
    may_read_tail = True
    assert list(tasks) == []


@pytest.mark.asyncio
async def test_async_wrapper_yields_before_the_tail_arrives() -> None:
    model = cast(Any, IterableModel(Item))
    may_read_tail = False

    async def chunks() -> AsyncIterator[str]:
        yield '{"tasks":[{"name":"real"}'
        assert may_read_tail, "complete tasks must not wait for wrapper metadata"
        yield '],"metadata":{"name":"phantom"}}'

    tasks = model.tasks_from_chunks_async(chunks())
    assert await anext(tasks) == Item(name="real")
    may_read_tail = True
    assert [item async for item in tasks] == []


@pytest.mark.parametrize(
    "tail",
    [',"metadata":{"name":"unfinished"', ',"metadata":[{"nested":{"value":1}'],
)
def test_unfinished_metadata_objects_keep_truncation_errors(tail: str) -> None:
    from instructor.v2.core.errors import IncompleteOutputException

    model = cast(Any, IterableModel(Item))
    tasks = model.tasks_from_chunks(['{"tasks":[{"name":"real"}]' + tail])
    assert next(tasks) == Item(name="real")
    with pytest.raises(IncompleteOutputException):
        list(tasks)


@pytest.mark.parametrize("tail", ["", ',"metadata":"unfinished'])
def test_missing_wrapper_close_does_not_gain_stricter_validation(tail: str) -> None:
    model = cast(Any, IterableModel(Item))
    assert list(model.tasks_from_chunks(['{"tasks":[{"name":"real"}]' + tail])) == [
        Item(name="real")
    ]


@pytest.fixture
def native_endpoint() -> Iterator[tuple[str, list[dict[str, Any]]]]:
    calls: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(request)
            case, chunking = request["messages"][-1]["content"].split(":")
            text = json.dumps(PAYLOADS[case])
            streaming = request.get("stream", False)
            if streaming:
                fragments = list(text) if chunking == "character" else [text]
                events = [
                    {
                        "id": "local",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "local",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": fragment},
                                "finish_reason": None,
                            }
                        ],
                    }
                    for fragment in fragments
                ]
                body = (
                    "".join("data: " + json.dumps(event) + "\n\n" for event in events)
                    + "data: [DONE]\n\n"
                ).encode()
            else:
                body = json.dumps(
                    {
                        "id": "local",
                        "object": "chat.completion",
                        "created": 1,
                        "model": "local",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": text},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                ).encode()
            self.send_response(200)
            self.send_header(
                "Content-Type", "text/event-stream" if streaming else "application/json"
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("case", PAYLOADS)
@pytest.mark.parametrize("chunking", ["whole", "character"])
async def test_public_json_wrapper_metadata(
    native_endpoint: tuple[str, list[dict[str, Any]]],
    asynchronous: bool,
    case: str,
    chunking: str,
) -> None:
    pytest.importorskip("openai")
    url, calls = native_endpoint
    client: Any = instructor.from_provider(
        "openai/local",
        mode=instructor.Mode.JSON,
        api_key="local-only",
        base_url=url,
        async_client=asynchronous,
        max_retries=0,
        timeout=5,
    )
    messages = [{"role": "user", "content": f"{case}:{chunking}"}]
    try:
        expected = client.create(
            response_model=list[Item], messages=messages, max_retries=0
        )
        if asynchronous:
            expected = await expected
        tasks = client.create_iterable(
            response_model=Item, messages=messages, max_retries=0
        )
        if asynchronous:
            actual = [item async for item in tasks]
        else:
            actual = list(tasks)
        assert actual == expected
        assert len(calls) == 2
        assert all(call["response_format"] == {"type": "json_object"} for call in calls)
    finally:
        if asynchronous:
            await client.client.close()
        else:
            client.client.close()
