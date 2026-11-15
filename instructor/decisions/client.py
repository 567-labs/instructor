"""HTTP clients for decision endpoints."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, TypeVar

import httpx
from pydantic import BaseModel

from .openai import DecisionInput, answer_envelope, request_body
from .schema import _Field, build_questions, parse_answers

if TYPE_CHECKING:
    import openai

T = TypeVar("T", bound=BaseModel)
_ENDPOINTS = {
    "openai": ("https://api.openai.com/v1/decisions", "OPENAI_API_KEY"),
    "typesafe": ("https://api.typesafe.ai/v1/systemone", "TYPESAFE_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/alpha/decisions", "OPENROUTER_API_KEY"),
}


# Distinguishes an omitted timeout from an explicit None.
_DEFAULT_TIMEOUT: Any = object()


def _request(
    model: str,
    response_model: type[T],
    context: dict[str, Any] | None,
    provider: str,
    native_input: DecisionInput | None,
) -> tuple[list[_Field], dict[str, Any], dict[str, Any]]:
    if native_input is not None and provider != "openai":
        raise TypeError(
            "input is supported only for OpenAI Decisions. Use context for Jev."
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
    if isinstance(client, (httpx.Client, httpx.AsyncClient)):
        return client.post(
            endpoint,
            json=body,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
    # SDK clients send authenticated requests and return the decoded JSON body.
    return client.post("/decisions", cast_to=object, body=body)


def _raw_json(response: Any) -> Any:
    if isinstance(response, httpx.Response):
        response.raise_for_status()
        return response.json()
    return response


class _DecisionsBase:
    def __init__(
        self,
        model: str,
        endpoint: str,
        api_key: str | None,
        http_client: Any = None,
        timeout: float | None = _DEFAULT_TIMEOUT,
        *,
        provider: str = "typesafe",
        _sdk_client: Any = None,
    ):
        self._provider = provider
        self.model = model
        self.endpoint = endpoint
        self._api_key = api_key
        default = timeout is _DEFAULT_TIMEOUT
        self._timeout = httpx.USE_CLIENT_DEFAULT if default else timeout
        borrowed = _sdk_client if _sdk_client is not None else http_client
        self._owned = borrowed is None
        self._client: Any = (
            borrowed
            if borrowed is not None
            else self._http_type(timeout=60.0 if default else timeout)
        )

    _http_type: type[httpx.Client] | type[httpx.AsyncClient]

    def _prepare(
        self,
        response_model: type[T],
        context: dict[str, Any] | None,
        input: DecisionInput | None,
    ) -> tuple[list[_Field], dict[str, Any], Any]:
        fields, body, context = _request(
            self.model, response_model, context, self._provider, input
        )
        response = _post(
            self._client, body, self.endpoint, self._api_key, self._timeout
        )
        return fields, context, response

    def _finish(
        self,
        response_model: type[T],
        fields: list[_Field],
        context: dict[str, Any],
        response: Any,
        strict: bool,
    ) -> tuple[T, dict[str, Any]]:
        raw = _raw_json(response)
        if self._provider == "openai":
            raw_answers = answer_envelope(fields, raw)
        else:
            raw_answers = raw
        result = parse_answers(response_model, fields, raw_answers, context, strict)
        return result, raw


class DecisionsClient(_DecisionsBase):
    """A synchronous client for typed decisions."""

    _http_type = httpx.Client

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
        fields, context, response = self._prepare(response_model, context, input)
        return self._finish(response_model, fields, context, response, strict)

    def close(self) -> None:
        if self._owned:
            self._client.close()

    def __enter__(self) -> DecisionsClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class AsyncDecisionsClient(_DecisionsBase):
    """An asynchronous client for typed decisions."""

    _http_type = httpx.AsyncClient

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
        fields, context, response = self._prepare(response_model, context, input)
        return self._finish(response_model, fields, context, await response, strict)

    async def close(self) -> None:
        if self._owned:
            await self._client.aclose()

    async def __aenter__(self) -> AsyncDecisionsClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


def from_openai_decisions(
    client: openai.OpenAI | openai.AsyncOpenAI,
    model: str | None,
    *,
    timeout: float | None = _DEFAULT_TIMEOUT,
) -> DecisionsClient | AsyncDecisionsClient:
    import openai

    options: dict[str, Any] = {"max_retries": 0}
    if timeout is not _DEFAULT_TIMEOUT:
        options["timeout"] = timeout
    cls = (
        AsyncDecisionsClient
        if isinstance(client, openai.AsyncOpenAI)
        else DecisionsClient
    )
    return cls(
        model or "gpt-6-luna",
        str(client.base_url.join("decisions")),
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
    timeout: float | None = _DEFAULT_TIMEOUT,
) -> DecisionsClient | AsyncDecisionsClient:
    if provider not in _ENDPOINTS:
        raise ValueError(f"Decisions mode does not support provider {provider!r}")
    default_endpoint, key_name = _ENDPOINTS[provider]
    api_key = api_key or os.environ.get(key_name)
    if not api_key:
        raise ValueError(f"Set {key_name} or pass api_key to use decisions mode")
    cls = AsyncDecisionsClient if async_client else DecisionsClient
    if http_client is not None and not isinstance(http_client, cls._http_type):
        raise TypeError(
            "async decisions require an httpx.AsyncClient"
            if async_client
            else "decisions require an httpx.Client"
        )
    return cls(
        model,
        default_endpoint if endpoint is None else endpoint,
        api_key,
        http_client,
        timeout,
        provider=provider,
    )
