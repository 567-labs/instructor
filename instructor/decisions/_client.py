"""HTTP clients for decision endpoints."""

from __future__ import annotations

import os
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from ._schema import build_questions, parse_answers

T = TypeVar("T", bound=BaseModel)
_ENDPOINTS = {
    "typesafe": ("https://api.typesafe.ai/v1/systemone", "TYPESAFE_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/alpha/decisions", "OPENROUTER_API_KEY"),
}


def _request(model: str, response_model: type[T], context: dict[str, Any]):
    fields = build_questions(response_model, context)
    body = {
        "model": model,
        "state": context,
        "questions": {field.name: field.question for field in fields},
    }
    return fields, body


class DecisionsClient:
    """A synchronous client for typed decisions."""

    def __init__(
        self,
        model: str,
        endpoint: str,
        api_key: str,
        http_client: httpx.Client | None = None,
        timeout: float = 60.0,
    ):
        self.model = model
        self.endpoint = endpoint
        self._api_key = api_key
        self._owned = http_client is None
        self._client = (
            http_client if http_client is not None else httpx.Client(timeout=timeout)
        )

    def create(
        self, *, response_model: type[T], context: dict[str, Any], strict: bool = True
    ) -> T:
        result, _ = self.create_with_completion(
            response_model=response_model, context=context, strict=strict
        )
        return result

    def create_with_completion(
        self, *, response_model: type[T], context: dict[str, Any], strict: bool = True
    ) -> tuple[T, dict[str, Any]]:
        fields, body = _request(self.model, response_model, context)
        response = self._client.post(
            self.endpoint,
            json=body,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        response.raise_for_status()
        raw = response.json()
        result = parse_answers(response_model, fields, raw, context, strict)
        return result, raw

    def close(self) -> None:
        if self._owned:
            self._client.close()

    def __enter__(self) -> DecisionsClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class AsyncDecisionsClient:
    """An asynchronous client for typed decisions."""

    def __init__(
        self,
        model: str,
        endpoint: str,
        api_key: str,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 60.0,
    ):
        self.model = model
        self.endpoint = endpoint
        self._api_key = api_key
        self._owned = http_client is None
        self._client = (
            http_client
            if http_client is not None
            else httpx.AsyncClient(timeout=timeout)
        )

    async def create(
        self, *, response_model: type[T], context: dict[str, Any], strict: bool = True
    ) -> T:
        result, _ = await self.create_with_completion(
            response_model=response_model, context=context, strict=strict
        )
        return result

    async def create_with_completion(
        self, *, response_model: type[T], context: dict[str, Any], strict: bool = True
    ) -> tuple[T, dict[str, Any]]:
        fields, body = _request(self.model, response_model, context)
        response = await self._client.post(
            self.endpoint,
            json=body,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        response.raise_for_status()
        raw = response.json()
        result = parse_answers(response_model, fields, raw, context, strict)
        return result, raw

    async def close(self) -> None:
        if self._owned:
            await self._client.aclose()

    async def __aenter__(self) -> AsyncDecisionsClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


def from_decisions_provider(
    provider: str, model: str, *, async_client: bool, api_key: str | None, **kwargs: Any
) -> DecisionsClient | AsyncDecisionsClient:
    if provider not in _ENDPOINTS:
        raise ValueError(f"Decisions mode does not support provider {provider!r}")
    endpoint, key_name = _ENDPOINTS[provider]
    api_key = api_key or os.environ.get(key_name)
    if not api_key:
        raise ValueError(f"Set {key_name} or pass api_key to use decisions mode")
    endpoint = kwargs.pop("endpoint", endpoint)
    http_client = kwargs.pop("http_client", None)
    timeout = kwargs.pop("timeout", 60.0)
    if kwargs:
        raise TypeError(
            f"Unsupported decisions client options: {', '.join(sorted(kwargs))}"
        )
    if async_client:
        if http_client is not None and not isinstance(http_client, httpx.AsyncClient):
            raise TypeError("async decisions require an httpx.AsyncClient")
        return AsyncDecisionsClient(model, endpoint, api_key, http_client, timeout)
    if http_client is not None and not isinstance(http_client, httpx.Client):
        raise TypeError("decisions require an httpx.Client")
    return DecisionsClient(model, endpoint, api_key, http_client, timeout)
