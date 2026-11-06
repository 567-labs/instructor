"""Tests for response_schema / prepare_response_model caching (Issue #2603).

Verifies that response_schema() memoizes the wrapped class per input model,
eliminating the per-request class creation that caused unbounded memory growth.
"""

from __future__ import annotations


from typing import Any, cast

import pytest
from pydantic import BaseModel, create_model

from instructor.v2.core.function_calls import (
    ResponseSchema,
    _response_schema_cache,
    response_schema,
)
from instructor.v2.core.response_model import prepare_response_model
from instructor.v2.providers.openai.schema import generate_openai_schema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class User(BaseModel):
    """A user with name and age."""

    name: str
    age: int


class Product(BaseModel):
    """A product."""

    title: str
    price: float
    in_stock: bool


# ---------------------------------------------------------------------------
# Core: response_schema caching
# ---------------------------------------------------------------------------


class TestResponseSchemaCaching:
    """response_schema() must return the same class for the same input."""

    def test_same_model_returns_same_object(self) -> None:
        """Calling response_schema(User) twice returns the exact same class."""
        a = response_schema(User)
        b = response_schema(User)
        assert a is b, "response_schema must return cached class on second call"

    def test_different_models_return_different_objects(self) -> None:
        """Different input models produce different wrapper classes."""
        user_schema = response_schema(User)
        product_schema = response_schema(Product)
        assert user_schema is not product_schema

    def test_cached_class_is_response_schema_subclass(self) -> None:
        """The cached result is still a proper ResponseSchema subclass."""
        result = response_schema(User)
        assert issubclass(result, ResponseSchema)
        assert issubclass(result, User)
        assert issubclass(result, BaseModel)

    def test_cached_class_preserves_schema(self) -> None:
        assert generate_openai_schema(response_schema(User)) == generate_openai_schema(
            User
        )

    def test_type_error_for_non_basemodel(self) -> None:
        """response_schema still raises TypeError for non-BaseModel inputs."""
        with pytest.raises(TypeError, match="pydantic.BaseModel"):
            response_schema(cast(Any, int))

        with pytest.raises(TypeError, match="pydantic.BaseModel"):
            response_schema(cast(Any, str))


# ---------------------------------------------------------------------------
# Integration: prepare_response_model caching
# ---------------------------------------------------------------------------


class TestPrepareResponseModelCaching:
    """prepare_response_model() must reuse cached response_schema results."""

    def test_plain_basemodel_cached(self) -> None:
        """Repeated prepare_response_model(User) returns the same class."""
        _response_schema_cache.clear()
        a = prepare_response_model(User)
        b = prepare_response_model(User)
        assert a is b, "prepare_response_model must return cached class"

    def test_none_returns_none(self) -> None:
        """None input still returns None."""
        assert prepare_response_model(None) is None

    def test_list_basemodel_still_works(self) -> None:
        """list[User] still produces an IterableBase subclass."""
        from instructor.v2.dsl.iterable import IterableBase

        result = prepare_response_model(list[User])
        assert result is not None
        assert issubclass(result, IterableBase)
        assert issubclass(result, ResponseSchema)

    def test_simple_type_still_works(self) -> None:
        """int/str still produce ModelAdapter subclasses."""
        from instructor.v2.dsl.simple_type import AdapterBase

        result_int = prepare_response_model(int)
        result_str = prepare_response_model(str)
        assert result_int is not None
        assert result_str is not None
        assert issubclass(result_int, AdapterBase)
        assert issubclass(result_str, AdapterBase)


# ---------------------------------------------------------------------------
# generate_openai_schema cache effectiveness
# ---------------------------------------------------------------------------


class TestOpenAISchemaCacheEffectiveness:
    """With response_schema caching, generate_openai_schema's lru_cache should hit."""

    def test_lru_cache_hits_after_fix(self) -> None:
        """generate_openai_schema should get cache hits for repeated models."""
        _response_schema_cache.clear()
        generate_openai_schema.cache_clear()

        for _ in range(10):
            prepared = prepare_response_model(User)
            generate_openai_schema(prepared)

        info = generate_openai_schema.cache_info()
        assert info.hits >= 9, (
            f"Expected >= 9 cache hits, got {info.hits}. Cache info: {info}"
        )
        assert info.misses <= 1, (
            f"Expected <= 1 cache miss, got {info.misses}. Cache info: {info}"
        )

        generate_openai_schema.cache_clear()


# ---------------------------------------------------------------------------
# Edge case: dynamic models and WeakKeyDictionary cleanup
# ---------------------------------------------------------------------------


def test_distinct_dynamic_models_reuse_only_their_own_live_wrappers() -> None:
    first = create_model("Dynamic", value=(str, ...))
    second = create_model("Dynamic", value=(int, ...))
    first_wrapper = response_schema(first)
    second_wrapper = response_schema(second)
    assert response_schema(first) is first_wrapper
    assert response_schema(second) is second_wrapper
    assert first_wrapper is not second_wrapper


# ---------------------------------------------------------------------------
# Edge case: subclass of ResponseSchema passed in
# ---------------------------------------------------------------------------


class TestAlreadyWrappedModel:
    """Models that already subclass ResponseSchema should not be double-wrapped."""

    def test_prepare_skips_already_wrapped(self) -> None:
        """prepare_response_model skips response_schema for ResponseSchema subclasses."""
        wrapped = response_schema(User)
        # Passing the already-wrapped class should return it as-is
        result = prepare_response_model(wrapped)
        assert result is wrapped


def test_wrapper_and_dynamic_source_are_collectible():
    import gc
    import weakref
    from pydantic import create_model

    def build():
        model = create_model("EphemeralInput", value=(int, ...))
        wrapped = response_schema(model)
        return weakref.ref(model), weakref.ref(wrapped)

    model, wrapped = build()
    gc.collect()
    assert wrapped() is None
    assert model() is None


def test_rebuilding_the_input_invalidates_its_wrapper():
    class Editable(BaseModel):
        value: int

    first = response_schema(Editable)
    Editable.model_fields["value"].annotation = str
    Editable.model_rebuild(force=True)
    second = response_schema(Editable)
    assert first is not second
    assert second.model_json_schema()["properties"]["value"]["type"] == "string"


def test_concurrent_requests_share_the_same_live_wrapper():
    from concurrent.futures import ThreadPoolExecutor

    class Shared(BaseModel):
        name: str

    with ThreadPoolExecutor(max_workers=4) as pool:
        models = list(pool.map(response_schema, [Shared] * 12))
    assert all(model is models[0] for model in models)
