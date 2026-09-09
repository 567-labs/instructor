"""Caching utilities for Instructor.

This module provides a very small abstraction layer so that users can
plug different cache back-ends (in-process LRU, `diskcache`, `redis`, …)
into the Instructor client via the ``cache=...`` keyword::

    from instructor import from_provider
    from instructor.cache import AutoCache

    cache = AutoCache(maxsize=10_000)
    client = from_provider("openai/gpt-4o", cache=cache)

The cache object must implement :class:`BaseCache`.  A minimal
requirement is to expose synchronous ``get`` / ``set`` methods (async
wrappers currently call them directly).  The default implementation
``AutoCache`` is an in-process LRU cache with a configurable size.

This first iteration purposefully keeps the API narrow: no eviction
hooks, no invalidation, no TTL for the LRU variant.  The objective is to
provide a safe foundation which we will extend in follow-up work.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import date, datetime
from enum import Enum
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any
from pydantic import BaseModel

__all__ = [
    "BaseCache",
    "AutoCache",
    "DiskCache",
    "make_cache_key",
]


class BaseCache(ABC):
    """Abstract cache contract.

    Concrete subclasses *must* be thread-safe.
    """

    @abstractmethod
    def get(self, key: str) -> Any | None:  # noqa: ANN401 – value type arbitrary
        """Return *None* to indicate a cache miss."""

    @abstractmethod
    def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,  # noqa: ARG002
    ) -> None:  # noqa: ANN401
        """Store *value* under *key*.

        ``ttl`` is time-to-live in **seconds**.  Implementations *may*
        ignore it (e.g. :class:`AutoCache`).
        """


class AutoCache(BaseCache):
    """Thread-safe in-process LRU cache using :class:`collections.OrderedDict`."""

    def __init__(self, maxsize: int = 128):
        if maxsize <= 0:
            raise ValueError("maxsize must be > 0")
        self._maxsize = maxsize
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    # ---------------------------------------------------------------------
    # BaseCache implementation
    # ---------------------------------------------------------------------
    def get(self, key: str) -> Any | None:  # noqa: ANN401
        with self._lock:
            try:
                value = self._cache.pop(key)
            except KeyError:
                return None
            # Move to the end (most recently used)
            self._cache[key] = value
            return value

    def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,  # noqa: ARG002
    ) -> None:  # noqa: ANN401
        # *ttl* is ignored for the in-process cache.
        with self._lock:
            if key in self._cache:
                self._cache.pop(key, None)
            self._cache[key] = value
            if len(self._cache) > self._maxsize:
                # popitem(last=False) pops the *least* recently used entry
                self._cache.popitem(last=False)


# -------------------------------------------------------------------------
# Optional back-ends – imported lazily so users do not need extra deps
# -------------------------------------------------------------------------


def _import_diskcache():  # pragma: no cover – only executed when requested
    import importlib.util

    if importlib.util.find_spec("diskcache") is None:
        raise ImportError(
            'diskcache is not installed. Install it with `pip install "instructor[diskcache]"`.'
        )
    import diskcache

    return diskcache


class DiskCache(BaseCache):
    """Wrapper around `diskcache.Cache`."""

    def __init__(self, directory: str = ".instructor_cache", **kwargs: Any):
        diskcache = _import_diskcache()
        self._cache = diskcache.Cache(directory, **kwargs)

    def get(self, key: str) -> Any | None:  # noqa: ANN401
        return self._cache.get(key)

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:  # noqa: ANN401
        if ttl is None:
            self._cache.set(key, value)
        else:
            self._cache.set(key, value, expire=ttl)


# -------------------------------------------------------------------------
# Cache-key helper
# -------------------------------------------------------------------------


def _canonical_cache_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _canonical_cache_value(value.model_dump(exclude_none=True))
    if isinstance(value, type) and issubclass(value, BaseModel):
        return _canonical_cache_value(value.model_json_schema())
    if isinstance(value, Enum):
        return _canonical_cache_value(value.value)
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, int):
        return ["int", value]
    if isinstance(value, float):
        return ["float", value]
    if isinstance(value, dict):
        items = [
            [_canonical_cache_value(key), _canonical_cache_value(item)]
            for key, item in value.items()
        ]
        items.sort(key=lambda pair: json.dumps(pair[0], sort_keys=True))
        return ["map", items]
    if isinstance(value, (list, tuple)):
        return ["sequence", [_canonical_cache_value(item) for item in value]]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if isinstance(value, datetime):
        return ["datetime", value.isoformat()]
    if isinstance(value, date):
        return ["date", value.isoformat()]
    raise TypeError(
        f"Cannot build a cache key for {type(value).__name__}; "
        "use serializable request settings or disable caching for this call"
    )


def make_cache_key(
    *,
    messages: Any,
    model: str | None,
    response_model: type[BaseModel] | None,
    mode: str | None = None,
    system: Any = None,
    provider: str | None = None,
    namespace: str | None = None,
    request_kwargs: dict[str, Any] | None = None,
) -> str:  # noqa: ANN401
    """Compute a *deterministic* cache key.

    The key space uses SHA-256("json payload") to keep the final length
    fixed regardless of input size.

    Components that influence the key:
        • provider/model name
        • serialized *messages* (user + system prompt, etc.)
        • *system* – providers such as Anthropic and Bedrock hoist system
          messages out of ``messages`` into a separate top-level parameter,
          so it has to be hashed separately or two calls that only differ in
          their system prompt would collide.
        • *mode* (Tools, JSON, …) – helps when users change Instructor mode
        • *response_model* schema – so edits to field definitions or
          descriptions invalidate prior cache entries (critical!).
    """

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "mode": mode,
    }
    if provider is not None:
        payload["provider"] = provider
    if namespace is not None:
        payload["namespace"] = namespace
    if request_kwargs is not None:
        generation_fields = (
            "temperature",
            "top_p",
            "top_k",
            "seed",
            "max_tokens",
            "max_completion_tokens",
            "max_output_tokens",
            "frequency_penalty",
            "presence_penalty",
            "n",
            "stop",
            "stop_sequences",
            "logit_bias",
            "reasoning_effort",
            "reasoning",
            "thinking",
            "config",
            "generation_config",
            "inferenceConfig",
            "additionalModelRequestFields",
        )
        generation = {}
        for field in generation_fields:
            value = request_kwargs.get(field)
            if value is not None:
                if isinstance(value, BaseModel):
                    value = value.model_dump(
                        exclude_none=True, exclude={"http_options"}
                    )
                elif field == "config" and isinstance(value, dict):
                    value = {
                        key: item
                        for key, item in value.items()
                        if key != "http_options"
                    }
                generation[field] = value
        payload["generation"] = generation

    # Only added when present so keys for providers that keep the system
    # prompt inside ``messages`` (OpenAI & friends) stay unchanged.
    if system is not None:
        payload["system"] = system

    if response_model is not None:
        # Include the entire JSON schema – guarantees busting when either
        # a field or its meta (title, description, constraints) changes.
        payload["schema"] = response_model.model_json_schema()

    data = json.dumps(_canonical_cache_value(payload), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def client_cache_identity(func: Any) -> dict[str, Any]:
    """Snapshot mutable SDK endpoint and authentication settings on each call.

    Provider adapters may wrap the SDK in a closure instead of passing a bound
    method. Inspect those captured clients too. Values are only used in the
    hashed identity, never persisted as plaintext or logged.
    """
    from inspect import isfunction

    settings = (
        "base_url",
        "_base_url",
        "api_key",
        "_api_key",
        "auth_token",
        "_token",
        "organization",
        "project",
        "location",
        "vertexai",
        "_http_options",
        "_custom_headers",
        "_custom_query",
        "_headers",
        "_auth",
        "_azure_ad_token",
        "_azure_ad_token_provider",
        "_credentials",
    )
    children = (
        "_client",
        "_api_client",
        "_client_wrapper",
        "_raw_client",
        "httpx_client",
        "_httpx_client",
        "_async_httpx_client",
    )
    seen: set[int] = set()

    def snapshot(obj: Any) -> dict[str, Any]:
        if obj is None or id(obj) in seen:
            return {}
        seen.add(id(obj))
        result = {}
        for name in settings:
            if hasattr(obj, name):
                value = getattr(obj, name)
                if name in {"base_url", "_base_url"} and value is not None:
                    value = str(value)
                elif name == "_headers" and hasattr(value, "multi_items"):
                    value = value.multi_items()
                result[name] = value
        for name in children:
            child = getattr(obj, name, None)
            if child is not None:
                result[name] = snapshot(child)
        return result

    identity = {"bound": snapshot(getattr(func, "__self__", None))}
    if isfunction(func) and func.__closure__:
        identity["captured"] = {
            str(index): snapshot(cell.cell_contents)
            for index, cell in enumerate(func.__closure__)
        }
    return identity


def make_request_cache_key(
    *,
    request: dict[str, Any],
    args: tuple[Any, ...],
    response_model: type[BaseModel],
    provider: str,
    mode: str,
    namespace: str,
    context: dict[str, Any] | None,
    strict: bool | None,
    client_identity: dict[str, Any] | None = None,
) -> str | None:
    """Hash the complete prepared request and validation policy.

    Unsupported values disable caching rather than risk ambiguous string keys.
    Namespaces default to a unique client scope; an explicit cache_namespace
    opts into sharing between clients and must identify the endpoint and tenant.
    """

    def encode(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return encode(value.model_dump(mode="json"))
        if isinstance(value, type) and issubclass(value, BaseModel):
            return encode(value.model_json_schema())
        if isinstance(value, dict):
            # JSON coerces mapping keys to strings, aliasing distinct policies
            # such as {1: "allowed"} and {"1": "allowed"} in validation context.
            if any(not isinstance(key, str) for key in value):
                raise TypeError("Cache identity mappings require string keys")
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(item) for item in value]
        return value

    try:
        payload = {
            "version": 2,
            "client": client_identity,
            "request": request,
            "args": args,
            "schema": response_model.model_json_schema(),
            "provider": provider,
            "mode": mode,
            "namespace": namespace,
            "context": context,
            "strict": strict,
        }
        data = json.dumps(encode(payload), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError, AttributeError, RecursionError):
        return None
    return hashlib.sha256(data.encode()).hexdigest()


def load_cached_response(
    cache: BaseCache,
    key: str,
    response_model: type[BaseModel],
    *,
    context: dict[str, Any] | None = None,
    strict: bool | None = None,
):  # noqa: ANN201
    """Return parsed model if *key* exists in *cache* else None."""
    from instructor.v2.core.cache_response import load_cached_response as load

    return load(cache, key, response_model, context=context, strict=strict)


def store_cached_response(
    cache: BaseCache, key: str, model: BaseModel, ttl: int | None = None
) -> None:  # noqa: D401
    """Serialize *model* and optional raw response to JSON and cache it."""
    from instructor.v2.core.cache_response import store_cached_response as store

    store(cache, key, model, ttl=ttl)
