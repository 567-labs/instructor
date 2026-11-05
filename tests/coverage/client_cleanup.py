from __future__ import annotations

import asyncio
import inspect
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import pytest
from pydantic.warnings import PydanticDeprecatedSince20


def clear_proxy_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "ALL_PROXY",
        "all_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
    ):
        monkeypatch.delenv(name, raising=False)


@contextmanager
def ignore_fireworks_pydantic_warning() -> Generator[None, None, None]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Pydantic V1 style `@validator` validators are deprecated\..*",
            category=PydanticDeprecatedSince20,
            module=r"fireworks\.client\.image_api",
        )
        yield


def close_idle_event_loop() -> None:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                message=r"There is no current event loop",
                category=DeprecationWarning,
            )
            loop = asyncio.get_event_loop()
    except (RuntimeError, DeprecationWarning):
        return

    if loop.is_running():
        return
    if not loop.is_closed():
        loop.close()
    asyncio.set_event_loop(None)


def close_provider_client(client: Any, *, async_client: bool = False) -> None:
    if client is None:
        return

    if type(client).__module__.startswith("fireworks.client"):
        if async_client:
            asyncio.run(close_async_provider_client(client))
        else:
            client._client_v1.close()
            client._image_client_v1.close()
        return

    methods = (
        (("aclose", ()), ("close", ()), ("__aexit__", (None, None, None)))
        if async_client
        else (("close", ()), ("__exit__", (None, None, None)))
    )
    for name, args in methods:
        method = getattr(client, name, None)
        if not callable(method):
            continue
        result = method(*args)
        if inspect.isawaitable(result):

            async def wait_for_close(awaitable: Any) -> None:
                await awaitable

            asyncio.run(wait_for_close(result))
        return


async def close_async_provider_client(client: Any) -> None:
    if client is None:
        return

    if type(client).__module__.startswith("fireworks.client"):
        client._client_v1.close()
        client._image_client_v1.close()
        await client.aclose()
        return

    for name, args in (
        ("aclose", ()),
        ("close", ()),
        ("__aexit__", (None, None, None)),
    ):
        method = getattr(client, name, None)
        if not callable(method):
            continue
        result = method(*args)
        if inspect.isawaitable(result):
            await result
        return
