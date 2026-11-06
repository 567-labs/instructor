"""HTTP clients for decision endpoints."""

from __future__ import annotations

import os
from typing import Any, TypeVar

import httpx
import openai
from pydantic import BaseModel

from .openai import DecisionInput, answer_envelope, request_body
from .schema import _Field, build_questions, parse_answers

T = TypeVar("T", bound=BaseModel)
_ENDPOINTS = {
    "openai": ("https://api.openai.com/v1/decisions", "OPENAI_API_KEY"),
    "typesafe": ("https://api.typesafe.ai/v1/systemone", "TYPESAFE_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/alpha/decisions", "OPENROUTER_API_KEY"),
}


class _DefaultTimeout:
    """Distinguish an omitted timeout from explicit None."""


_DEFAULT_TIMEOUT = _DefaultTimeout()


def _request(
    model: str,
    response_model: type[T],
    context: dict[str, Any] | None,
    provider: str,
    native_input: DecisionInput | None,
) -> tuple[list[_Field], dict[str, Any], dict[str, Any]]:
    if native_input is not None and provider != "openai":
        raise TypeError(
            "input is supported only for OpenAI Decisions; use context for Jev"
        )
    if context is None:
        if native_input is None:
            raise TypeError("context must be a dictionary")
        context = {}
    fields = build_questions(response_model, context)
    if provider == "openai":
        return fields, request_body(model, fields, context, native_input), context
    body = {
        "model": model,
        "state": context,
        "questions": {field.name: field.question for field in fields},
    }
    return fields, body, context


def _post(
    client: httpx.Client | httpx.AsyncClient | openai.OpenAI | openai.AsyncOpenAI,
    body: dict[str, Any],
    endpoint: str,
    api_key: str | None,
    timeout: Any,
) -> Any:
    if isinstance(client, (openai.OpenAI, openai.AsyncOpenAI)):
        resource = getattr(client, "decisions", None)
        if resource is not None:
            return resource.with_raw_response.create(**body)
        # Older supported SDKs still provide authenticated requests to this endpoint.
        return client.post("/decisions", cast_to=object, body=body)
    return client.post(
        endpoint,
        json=body,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )


def _raw_json(response: Any) -> Any:
    if isinstance(response, httpx.Response):
        response.raise_for_status()
        return response.json()
    if hasattr(response, "http_response"):
        return response.http_response.json()
    return response


def _parse(
    response_model: type[T],
    fields: list[_Field],
    raw: Any,
    context: dict[str, Any],
    strict: bool,
    provider: str,
) -> T:
    if provider == "openai":
        raw = answer_envelope(fields, raw)
    return parse_answers(response_model, fields, raw, context, strict)


class DecisionsClient:
    """A synchronous client for typed decisions."""

    def __init__(
        self,
        model: str,
        endpoint: str,
        api_key: str | None,
        http_client: httpx.Client | None = None,
        timeout: float | None | _DefaultTimeout = _DEFAULT_TIMEOUT,
        *,
        provider: str = "typesafe",
        _sdk_client: openai.OpenAI | None = None,
    ):
        self._provider = provider
        self.model = model
        self.endpoint = endpoint
        self._api_key = api_key
        self._timeout = (
            httpx.USE_CLIENT_DEFAULT
            if isinstance(timeout, _DefaultTimeout)
            else timeout
        )
        borrowed = _sdk_client if _sdk_client is not None else http_client
        self._owned = borrowed is None
        self._client = (
            borrowed
            if borrowed is not None
            else httpx.Client(
                timeout=60.0 if isinstance(timeout, _DefaultTimeout) else timeout
            )
        )

    def create(
        self,
        *,
        response_model: type[T],
        context: dict[str, Any] | None = None,
        input: DecisionInput | None = None,
        strict: bool = True,
    ) -> T:
        result, _ = self.create_with_completion(
            response_model=response_model, context=context, input=input, strict=strict
        )
        return result

    def create_with_completion(
        self,
        *,
        response_model: type[T],
        context: dict[str, Any] | None = None,
        input: DecisionInput | None = None,
        strict: bool = True,
    ) -> tuple[T, dict[str, Any]]:
        fields, body, context = _request(
            self.model, response_model, context, self._provider, input
        )
        response = _post(
            self._client, body, self.endpoint, self._api_key, self._timeout
        )
        raw = _raw_json(response)
        result = _parse(response_model, fields, raw, context, strict, self._provider)
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
        api_key: str | None,
        http_client: httpx.AsyncClient | None = None,
        timeout: float | None | _DefaultTimeout = _DEFAULT_TIMEOUT,
        *,
        provider: str = "typesafe",
        _sdk_client: openai.AsyncOpenAI | None = None,
    ):
        self._provider = provider
        self.model = model
        self.endpoint = endpoint
        self._api_key = api_key
        self._timeout = (
            httpx.USE_CLIENT_DEFAULT
            if isinstance(timeout, _DefaultTimeout)
            else timeout
        )
        borrowed = _sdk_client if _sdk_client is not None else http_client
        self._owned = borrowed is None
        self._client = (
            borrowed
            if borrowed is not None
            else httpx.AsyncClient(
                timeout=60.0 if isinstance(timeout, _DefaultTimeout) else timeout
            )
        )

    async def create(
        self,
        *,
        response_model: type[T],
        context: dict[str, Any] | None = None,
        input: DecisionInput | None = None,
        strict: bool = True,
    ) -> T:
        result, _ = await self.create_with_completion(
            response_model=response_model, context=context, input=input, strict=strict
        )
        return result

    async def create_with_completion(
        self,
        *,
        response_model: type[T],
        context: dict[str, Any] | None = None,
        input: DecisionInput | None = None,
        strict: bool = True,
    ) -> tuple[T, dict[str, Any]]:
        fields, body, context = _request(
            self.model, response_model, context, self._provider, input
        )
        response = await _post(
            self._client, body, self.endpoint, self._api_key, self._timeout
        )
        raw = _raw_json(response)
        result = _parse(response_model, fields, raw, context, strict, self._provider)
        return result, raw

    async def close(self) -> None:
        if self._owned and isinstance(self._client, httpx.AsyncClient):
            await self._client.aclose()

    async def __aenter__(self) -> AsyncDecisionsClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


def from_openai_decisions(
    client: openai.OpenAI | openai.AsyncOpenAI,
    model: str | None,
    *,
    timeout: float | None | _DefaultTimeout = _DEFAULT_TIMEOUT,
) -> DecisionsClient | AsyncDecisionsClient:
    options: dict[str, Any] = {"max_retries": 0}
    if not isinstance(timeout, _DefaultTimeout):
        options["timeout"] = timeout
    model = model or "gpt-6-luna"
    endpoint = str(client.base_url.join("decisions"))
    if isinstance(client, openai.AsyncOpenAI):
        return AsyncDecisionsClient(
            model,
            endpoint,
            None,
            provider="openai",
            _sdk_client=client.with_options(**options),
        )
    return DecisionsClient(
        model,
        endpoint,
        None,
        provider="openai",
        _sdk_client=client.with_options(**options),
    )


def from_decisions_provider(
    provider: str,
    model: str,
    *,
    async_client: bool,
    api_key: str | None,
    endpoint: str | None = None,
    http_client: httpx.Client | httpx.AsyncClient | None = None,
    timeout: float | None | _DefaultTimeout = _DEFAULT_TIMEOUT,
) -> DecisionsClient | AsyncDecisionsClient:
    if provider not in _ENDPOINTS:
        raise ValueError(f"Decisions mode does not support provider {provider!r}")
    default_endpoint, key_name = _ENDPOINTS[provider]
    api_key = api_key or os.environ.get(key_name)
    if not api_key:
        raise ValueError(f"Set {key_name} or pass api_key to use decisions mode")
    endpoint = default_endpoint if endpoint is None else endpoint
    if async_client:
        if http_client is not None and not isinstance(http_client, httpx.AsyncClient):
            raise TypeError("async decisions require an httpx.AsyncClient")
        return AsyncDecisionsClient(
            model, endpoint, api_key, http_client, timeout, provider=provider
        )
    if http_client is not None and not isinstance(http_client, httpx.Client):
        raise TypeError("decisions require an httpx.Client")
    return DecisionsClient(
        model, endpoint, api_key, http_client, timeout, provider=provider
    )
