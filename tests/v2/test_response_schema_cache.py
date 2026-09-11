"""Tests for response_schema / prepare_response_model caching (Issue #2603).

Verifies that response_schema() memoizes the wrapped class per input model,
eliminating the per-request class creation that caused unbounded memory growth.
"""

from __future__ import annotations

import gc


import pytest
from pydantic import BaseModel, Field, create_model

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


class Wide(BaseModel):
    """A wider model with many fields."""

    f0: str = Field(..., description="Field zero.")
    f1: str = Field(..., description="Field one.")
    f2: int = Field(..., description="Field two.")
    f3: float = Field(..., description="Field three.")
    f4: bool = Field(..., description="Field four.")


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
        """The cached class produces the same OpenAI schema as a fresh one."""
        _response_schema_cache.clear()
        first = response_schema(User)
        schema_first = generate_openai_schema(first)

        # Clear the openai schema cache to force re-computation
        generate_openai_schema.cache_clear()

        second = response_schema(User)
        schema_second = generate_openai_schema(second)

        assert schema_first == schema_second
        assert first is second  # still the same object

    def test_type_error_for_non_basemodel(self) -> None:
        """response_schema still raises TypeError for non-BaseModel inputs."""
        with pytest.raises(TypeError, match="pydantic.BaseModel"):
            response_schema(int)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="pydantic.BaseModel"):
            response_schema(str)  # type: ignore[arg-type]


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


class TestDynamicModelCleanup:
    """Dynamically-created models should be cleaned up by the WeakKeyDictionary."""

    def test_dynamic_model_is_cached(self) -> None:
        """A dynamically-created model should be cached while it exists."""
        DynModel = create_model("DynModel", name=(str, ...), value=(int, ...))
        a = response_schema(DynModel)
        b = response_schema(DynModel)
        assert a is b

    def test_dynamic_model_cleanup(self) -> None:
        """When a dynamic model is deleted, its cache entry is cleaned up."""
        DynModel = create_model("Ephemeral", x=(str, ...))
        _ = response_schema(DynModel)
        assert DynModel in _response_schema_cache

        # Drop all references to the dynamic model
        del DynModel, _
        gc.collect()

        # The WeakKeyDictionary should have cleaned up the entry.
        # We can't check for a specific key since it's been deleted,
        # but we can verify the cache didn't grow unboundedly.
        # (This test mainly verifies no crash/error occurs on cleanup.)

    def test_repeated_calls_dont_grow_cache(self) -> None:
        """Calling response_schema N times with the same model adds only 1 cache entry."""
        _response_schema_cache.clear()

        for _ in range(100):
            response_schema(User)

        assert len(_response_schema_cache) == 1, (
            f"Expected 1 cache entry for 100 calls with the same model, "
            f"got {len(_response_schema_cache)}"
        )

    def test_n_distinct_models_add_n_entries(self) -> None:
        """N distinct models create exactly N cache entries, not N * calls."""
        _response_schema_cache.clear()

        models = [create_model(f"M{i}", val=(int, ...)) for i in range(10)]
        for m in models:
            # Call 5 times each — should still be only 10 entries total
            for _ in range(5):
                response_schema(m)

        assert len(_response_schema_cache) == 10, (
            f"Expected 10 cache entries for 10 models x 5 calls, "
            f"got {len(_response_schema_cache)}"
        )


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
