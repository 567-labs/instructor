"""Retention and mutation contracts relevant to issue #2603.

Use weak references, not a list of generated classes: the latter creates the
retention being measured. Byte counts belong in the offline allocation probe.
"""

import gc
import weakref
from collections.abc import Iterable
from typing import Any

import pytest
from pydantic import BaseModel, create_model
from typing_extensions import TypedDict

import instructor
from instructor.function_calls import openai_schema as public_schema
from instructor.processing.function_calls import openai_schema as processing_schema
from instructor.utils.core import prepare_response_model as legacy_prepare
from instructor.v2.core.function_calls import openai_schema, response_schema
from instructor.v2.core.response_model import prepare_response_model
from instructor.v2.providers.openai.schema import generate_openai_schema
from instructor.v2.validation.async_validators import async_field_validator


class User(BaseModel):
    name: str


class Nested(BaseModel):
    users: list[User]


class Record(TypedDict):
    name: str


@pytest.fixture(autouse=True)
def clear_schema_cache():
    generate_openai_schema.cache_clear()
    yield
    generate_openai_schema.cache_clear()


INPUT_MODELS: list[Any] = [
    int,
    User,
    Nested,
    Record,
    list[User],
    Iterable[User],
    list[Record],
]


def test_generated_classes_are_collectible():
    capacity = generate_openai_schema.cache_info().maxsize
    assert capacity is not None
    # Weak refs in first-use order. Repeated requests may reuse one wrapper,
    # and every model's block runs to completion before the next starts, so
    # the LRU schema cache must hold exactly the newest ``capacity`` wrappers.
    prepared_only: dict[weakref.ReferenceType, None] = {}
    cached: dict[weakref.ReferenceType, None] = {}
    for input_model in INPUT_MODELS:
        for _ in range(10):
            prepared = prepare_response_model(input_model)
            assert prepared is not None
            prepared_only[weakref.ref(prepared)] = None
        for _ in range(capacity + 10):
            prepared = prepare_response_model(input_model)
            assert prepared is not None
            cached[weakref.ref(prepared)] = None
            generate_openai_schema(prepared)
    del prepared
    gc.collect()
    assert all(ref() is None for ref in prepared_only)
    live = [ref() is not None for ref in cached]
    assert len(cached) > capacity
    assert live == [False] * (len(cached) - capacity) + [True] * capacity
    generate_openai_schema.cache_clear()
    gc.collect()
    assert all(ref() is None for ref in cached)


def test_dynamic_input_and_wrapped_classes_are_released_after_eviction():
    capacity = generate_openai_schema.cache_info().maxsize
    assert capacity is not None
    inputs = []
    outputs = []
    for index in range(capacity + 10):
        source = create_model(f"Ephemeral{index}", value=(int, ...))
        prepared = prepare_response_model(source)
        assert prepared is not None
        inputs.append(weakref.ref(source))
        outputs.append(weakref.ref(prepared))
        generate_openai_schema(prepared)
    del source, prepared
    gc.collect()
    assert all(ref() is None for ref in inputs[:10] + outputs[:10])
    generate_openai_schema.cache_clear()
    gc.collect()
    assert all(ref() is None for ref in inputs + outputs)


def test_public_aliases_use_the_same_implementation():
    assert legacy_prepare is prepare_response_model
    assert instructor.openai_schema is public_schema
    assert public_schema is processing_schema is openai_schema is response_schema


def test_preparation_observes_source_rebuild_and_isolates_wrapper_mutation():
    source = create_model("Mutable", value=(int, 1))
    first = prepare_response_model(source)
    assert first is not None
    first.model_fields["value"].description = "wrapper-local"
    first.model_rebuild(force=True)
    source.model_fields["value"].default = 2
    source.model_fields["value"].description = "updated source"
    source.model_rebuild(force=True)
    second = prepare_response_model(source)
    assert second is not None
    assert second().model_dump()["value"] == 2
    assert first().model_dump()["value"] == 1
    assert second.model_fields["value"].description == "updated source"
    assert first.model_fields["value"].description == "wrapper-local"
    assert prepare_response_model(second) is second


def test_explicit_schema_cache_clear_refreshes_rebuilt_prepared_model():
    prepared = prepare_response_model(create_model("Mutable", value=(int, 1)))
    assert prepared is not None
    assert (
        generate_openai_schema(prepared)["parameters"]["properties"]["value"]["default"]
        == 1
    )
    prepared.model_fields["value"].default = 2
    prepared.model_rebuild(force=True)
    generate_openai_schema.cache_clear()
    assert (
        generate_openai_schema(prepared)["parameters"]["properties"]["value"]["default"]
        == 2
    )


@pytest.mark.parametrize(
    "container", [lambda model: model, lambda model: list[model]], ids=["model", "list"]
)
def test_repeated_preparation_rechecks_new_nested_async_validator(container):
    child: Any = create_model("Child", value=(str, ...))
    parent = create_model("Parent", child=(child, ...))
    response_model = container(parent)
    prepare_response_model(response_model)

    @async_field_validator("value")
    async def reject(_cls, value):
        return value

    child.reject = reject
    with pytest.raises(ValueError, match="async validators are not supported"):
        prepare_response_model(response_model)
