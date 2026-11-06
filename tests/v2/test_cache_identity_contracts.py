"""Cache identity must preserve typed mapping keys in validation context."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

from instructor.cache import make_request_cache_key


class Answer(BaseModel):
    value: int


def key(context: dict[str, Any]) -> str | None:
    return make_request_cache_key(
        request={"model": "local"},
        args=(),
        response_model=Answer,
        provider="openai",
        mode="json",
        namespace="contract",
        context=context,
        strict=True,
    )


@pytest.mark.parametrize("mapping_key", [1, False, None, 1.5])
def test_context_mapping_keys_do_not_alias_json_strings(mapping_key: Any) -> None:
    typed = {"policy": {mapping_key: "allowed"}}
    stringified = json.loads(json.dumps(typed))
    typed_key = key(typed)
    string_key = key(stringified)
    assert string_key is not None
    # JSON stringifies these keys; bypass caching rather than alias a policy.
    assert typed_key is None


def test_context_mapping_order_does_not_change_identity() -> None:
    first = key({"policy": {"a": 1, "b": 2}})
    assert first is not None
    assert first == key({"policy": {"b": 2, "a": 1}})


@pytest.mark.parametrize("container", [list, tuple])
def test_context_mapping_keys_inside_sequences_bypass_cache(container: Any) -> None:
    assert key({"policy": container([{1: "allowed"}])}) is None


@pytest.mark.parametrize("value", [object(), float("nan"), float("inf")])
def test_unsupported_context_keeps_bypassing_cache(value: Any) -> None:
    assert key({"policy": value}) is None


def test_cyclic_context_bypasses_cache() -> None:
    context: dict[str, Any] = {}
    context["self"] = context
    assert key(context) is None
