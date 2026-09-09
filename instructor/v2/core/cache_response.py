"""Serialize and restore cached runtime responses."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from instructor.v2.validation.async_validators import reject_async_validators

if TYPE_CHECKING:
    from instructor.cache import BaseCache

logger = logging.getLogger("instructor.cache")


def load_cached_response(
    cache: BaseCache,
    key: str,
    response_model: type[BaseModel],
    *,
    context: dict[str, Any] | None = None,
    strict: bool | None = None,
):
    """Return parsed model if *key* exists in *cache* else None."""
    reject_async_validators(response_model)
    cached = cache.get(key)
    if cached is None:
        return None

    try:
        data = json.loads(cached)
        model_json = data["model"]
        raw_json = data.get("raw")
    except Exception:
        model_json = cached
        raw_json = None

    obj = response_model.model_validate_json(model_json, context=context, strict=strict)
    if raw_json is not None:
        # `_raw_response` is an internal attribute used by Instructor; it may not
        # be declared on the Pydantic model type.
        try:
            raw_data = json.loads(raw_json)

            if isinstance(raw_data, dict) and any(
                key in raw_data for key in ["id", "object", "model", "choices"]
            ):
                # Completion-like mappings regain attribute access; other JSON
                # shapes remain plain data structures.
                object.__setattr__(
                    obj,
                    "_raw_response",
                    json.loads(raw_json, object_hook=lambda d: SimpleNamespace(**d)),
                )
                logger.debug("Restored raw response as SimpleNamespace object")
            else:
                object.__setattr__(obj, "_raw_response", raw_data)
                logger.debug("Restored raw response as plain data structure")
        except (json.JSONDecodeError, TypeError):
            # Preserve the historical string fallback.
            object.__setattr__(obj, "_raw_response", raw_json)
            logger.debug(
                "Restored raw response as string (original could not be fully serialized)"
            )
    logger.debug("cache hit: %s", key)
    return obj


def store_cached_response(
    cache: BaseCache, key: str, model: BaseModel, ttl: int | None = None
) -> None:
    """Serialize *model* and optional raw response to JSON and cache it."""
    raw_resp = getattr(model, "_raw_response", None)
    if raw_resp is not None:
        try:
            # Try Pydantic model serialization first (OpenAI, Anthropic, etc.)
            raw_resp_dump = getattr(raw_resp, "model_dump_json", None)
            if callable(raw_resp_dump):
                raw_json = raw_resp_dump()
            else:
                raise AttributeError("raw_resp has no model_dump_json")
            logger.debug("Cached raw response as Pydantic JSON")
        except (AttributeError, TypeError):
            # Fallback for non-Pydantic responses (custom providers, plain dicts, etc.)
            try:
                raw_json = json.dumps(raw_resp, default=str)
                logger.debug(
                    "Cached raw response as plain JSON (provider may not support full reconstruction)"
                )
            except (TypeError, ValueError):
                # Final fallback - string representation
                raw_json = str(raw_resp)
                logger.warning(
                    "Raw response could not be serialized as JSON, using string fallback. "
                    "create_with_completion may not fully restore original object structure."
                )
    else:
        raw_json = None

    payload = {
        "model": model.model_dump_json(),
        "raw": raw_json,
    }
    cache.set(key, json.dumps(payload), ttl=ttl)
    logger.debug("cache store: %s", key)
