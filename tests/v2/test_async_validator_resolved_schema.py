"""Reject marked validators after Pydantic resolves container annotations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import TYPE_CHECKING, Any, ClassVar, Generic, NamedTuple, TypeVar

import pytest
from pydantic import BaseModel, ConfigDict
from typing_extensions import TypeAliasType, TypedDict

import instructor
from instructor.v2.validation import async_field_validator
from instructor.v2.core.client import Instructor, AsyncInstructor

if TYPE_CHECKING:
    NotAvailableInModule = Any


class Plain(BaseModel):
    value: str


class Marked(BaseModel):
    value: str

    @async_field_validator("value")
    async def check(cls, value: str) -> str:
        raise ValueError(f"Unsupported validator must not be skipped for {value}")


MarkedAlias = TypeAliasType("MarkedAlias", "list[Marked]")
PlainAlias = TypeAliasType("PlainAlias", "list[Plain]")
T = TypeVar("T")


@dataclass
class GenericData(Generic[T]):
    item: T


class GenericTuple(NamedTuple, Generic[T]):
    item: T


class GenericDict(TypedDict, Generic[T]):
    item: T


def response_model(kind: str, marked: bool) -> type[BaseModel]:
    if TYPE_CHECKING:
        LocalLeaf = BaseModel
    else:
        LocalLeaf = Marked if marked else Plain
    if kind == "dataclass":

        @dataclass
        class LocalContainer:
            item: LocalLeaf
    elif kind == "typeddict":

        class LocalContainer(TypedDict):
            item: LocalLeaf
    elif kind == "namedtuple":

        class LocalContainer(NamedTuple):
            item: LocalLeaf
    elif kind == "unresolved_classvar":

        @dataclass
        class LocalContainer:
            item: LocalLeaf
            registry: ClassVar[NotAvailableInModule]
    elif kind == "extra_values":

        class LocalContainer(BaseModel):
            model_config = ConfigDict(extra="allow")
            if not TYPE_CHECKING:
                __pydantic_extra__: dict[str, LocalLeaf]
    elif kind == "string_alias":
        LocalContainer = MarkedAlias if marked else PlainAlias
    else:
        container = {
            "generic_dataclass": GenericData,
            "generic_namedtuple": GenericTuple,
            "generic_typeddict": GenericDict,
        }[kind]
        LocalContainer = container[LocalLeaf]

    class Envelope(BaseModel):
        if TYPE_CHECKING:
            data: Any
        else:
            data: LocalContainer

    Envelope.model_rebuild(force=True, _types_namespace=locals())
    return Envelope


@pytest.mark.asyncio
@pytest.mark.parametrize("async_client", [False, True])
@pytest.mark.parametrize("marked", [False, True])
@pytest.mark.parametrize(
    "kind",
    [
        "dataclass",
        "typeddict",
        "namedtuple",
        "unresolved_classvar",
        "string_alias",
        "extra_values",
        "generic_dataclass",
        "generic_namedtuple",
        "generic_typeddict",
    ],
)
async def test_public_resolved_container_guard(
    kind: str, marked: bool, async_client: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = response_model(kind, marked)
    data: Any = (
        [{"value": "Ada"}]
        if kind in {"namedtuple", "generic_namedtuple", "string_alias"}
        else {"item": {"value": "Ada"}}
    )
    payload = {"data": data}
    # Establish a legal, fully resolved Pydantic response independently of the guard.
    expected = model.model_validate(payload).model_dump()
    schema = model.model_json_schema()
    assert model.__pydantic_complete__
    assert "registry" not in json.dumps(schema)
    requests: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(req)
            function = req["tools"][0]["function"]
            body = json.dumps(
                {
                    "id": "local",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "local",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call",
                                        "type": "function",
                                        "function": {
                                            "name": function["name"],
                                            "arguments": json.dumps(payload),
                                        },
                                    }
                                ],
                            },
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
    thread = Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(key, raising=False)
    client: Any = None
    try:
        client = instructor.from_provider(
            "openai/local",
            api_key="local-not-secret",
            mode=instructor.Mode.TOOLS,
            async_client=async_client,
        )

        async def create() -> Any:
            if isinstance(client, AsyncInstructor):
                return await client.create(
                    response_model=model,
                    messages=[{"role": "user", "content": "Extract a value"}],
                    max_retries=0,
                )
            assert isinstance(client, Instructor)
            return client.create(
                response_model=model,
                messages=[{"role": "user", "content": "Extract a value"}],
                max_retries=0,
            )

        if marked:
            with pytest.raises(
                ValueError, match="Instructor async validators are not supported"
            ):
                await create()
            assert requests == []
        else:
            assert (await create()).model_dump() == expected
            assert len(requests) == 1
    finally:
        if isinstance(client, (Instructor, AsyncInstructor)):
            assert client.client is not None
            if async_client:
                await client.client.close()
            else:
                client.client.close()
        server.shutdown()
        server.server_close()
        thread.join()
