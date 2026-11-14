from __future__ import annotations

from typing import Any, cast

import pytest

from instructor.v2.auto_client import from_provider


@pytest.mark.parametrize("async_client", [False, True])
def test_from_provider_ollama_forwards_api_key(async_client: bool) -> None:
    client = cast(
        Any, from_provider("ollama/llama3.2", async_client=async_client, api_key="sec")
    )

    assert client.client.api_key == "sec"


@pytest.mark.parametrize("async_client", [False, True])
def test_from_provider_ollama_defaults_api_key(async_client: bool) -> None:
    client = cast(Any, from_provider("ollama/llama3.2", async_client=async_client))

    assert client.client.api_key == "ollama"
