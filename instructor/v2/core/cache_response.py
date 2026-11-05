"""Serialize and restore cached runtime responses."""

from __future__ import annotations

import inspect
import json
import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError
from pydantic_core import SchemaValidator

from instructor.v2.validation.async_validators import reject_async_validators

if TYPE_CHECKING:
    from instructor.cache import BaseCache

logger = logging.getLogger("instructor.cache")


_SUPPORTS_BY_NAME = (
    "by_name" in inspect.signature(BaseModel.model_validate_json).parameters
)


def _field_name_schema(node: Any, *, by_alias: bool) -> Any:
    """Enable field names on nested schemas without changing the caller's model."""
    if isinstance(node, list):
        return [_field_name_schema(item, by_alias=by_alias) for item in node]
    if not isinstance(node, dict):
        return node
    result = {
        key: _field_name_schema(value, by_alias=by_alias)
        for key, value in node.items()
        if by_alias or key != "validation_alias"
    }
    if node.get("type") in {"model", "typed-dict", "dataclass"}:
        result["config"] = {
            **node.get("config", {}),
            "populate_by_name": True,
            "validate_by_name": True,
            "validate_by_alias": by_alias,
        }
    return result


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
    except (ValueError, TypeError):
        data = None

    def validate_document(value: Any, *, by_alias: bool) -> BaseModel:
        if _SUPPORTS_BY_NAME:
            return response_model.model_validate_json(
                value, by_alias=by_alias, by_name=True, context=context, strict=strict
            )
        # Older Pydantic versions require a field-name validator built from the schema.
        validator = SchemaValidator(
            _field_name_schema(
                response_model.__pydantic_core_schema__, by_alias=by_alias
            )
        )
        return validator.validate_json(value, context=context, strict=strict)

    def is_shape_error(exc: ValidationError) -> bool:
        return all(
            error["type"] in {"json_invalid", "model_type", "missing"}
            for error in exc.errors(include_input=False)
        )

    def validate_old_envelope(inner_json: str) -> BaseModel:
        try:
            inner = json.loads(inner_json)
        except ValueError:
            inner = None
        if (
            isinstance(inner, dict)
            and inner.keys() == response_model.model_fields.keys()
        ):
            try:
                return validate_document(inner_json, by_alias=False)
            except ValidationError as exc:
                if not is_shape_error(exc):
                    raise
                # Earlier nested TypedDict wrappers retained their wire keys.
        return validate_document(inner_json, by_alias=True)

    raw_json = None
    tagged = (
        isinstance(data, dict) and data.get("$instructor_cache") == "field-names-v2"
    )
    envelope_keys = (
        {"model", "raw", "$instructor_cache"} if tagged else {"model", "raw"}
    )
    if (
        isinstance(data, dict)
        and data.keys() <= envelope_keys
        and isinstance(data.get("model"), str)
        and (data.get("raw") is None or isinstance(data["raw"], str))
    ):
        try:
            obj = (
                validate_document(data["model"], by_alias=False)
                if tagged
                else validate_old_envelope(data["model"])
            )
        except ValidationError as exc:
            if tagged or not is_shape_error(exc):
                raise
            # Legacy business models can themselves have fields named model and raw.
            try:
                obj = validate_document(cached, by_alias=True)
            except ValidationError:
                raise exc from None
        else:
            raw_json = data.get("raw")
    else:
        obj = validate_document(cached, by_alias=True)
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
    from instructor.v2.core.response_model import _CACHE_FIELD_NAMES_CONTEXT

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
        "$instructor_cache": "field-names-v2",
        "model": model.model_dump_json(
            round_trip=True, by_alias=False, context=_CACHE_FIELD_NAMES_CONTEXT
        ),
        "raw": raw_json,
    }
    cache.set(key, json.dumps(payload), ttl=ttl)
    logger.debug("cache store: %s", key)
