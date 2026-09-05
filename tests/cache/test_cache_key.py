from typing import Any
from datetime import date, datetime
from enum import Enum

from instructor.cache import make_cache_key
import pytest
from pydantic import BaseModel, Field  # type: ignore[import-not-found]


messages = [
    {"role": "user", "content": "hello"},
]
model_name = "gpt-4.1-mini"


def test_cache_normalizes_models_schemas_and_enum_values():
    class Settings(BaseModel):
        seed: int

    class Choice(Enum):
        FIRST = "first"

    def key(value):
        return make_cache_key(messages=value, model=model_name, response_model=None)

    assert key(Settings(seed=4)) == key({"seed": 4})
    assert key(Settings) == key(Settings.model_json_schema())
    assert key(Choice.FIRST) == key("first")
    assert key(date(2026, 9, 4)) != key("2026-09-04")
    assert key(datetime(2026, 9, 4)) != key("2026-09-04T00:00:00")
    assert key(date(2026, 9, 4)) != key(datetime(2026, 9, 4))


@pytest.mark.parametrize("field", ["temperature", "config", "inferenceConfig"])
def test_cache_key_isolates_generation_configuration(field):
    def key(value):
        return make_cache_key(
            messages=messages,
            model=model_name,
            response_model=None,
            provider="genai",
            request_kwargs={field: value},
        )

    assert key({"temperature": 0.2}) != key({"temperature": 0.8})
    assert key({"temperature": 0.2, "seed": 4}) == key({"seed": 4, "temperature": 0.2})


def test_cache_key_isolates_provider():
    kwargs: dict[str, Any] = dict(
        messages=messages, model=model_name, response_model=None
    )
    assert make_cache_key(**kwargs, provider="openai") != make_cache_key(
        **kwargs, provider="litellm"
    )


def test_cache_namespace_separates_accounts():
    kwargs: dict[str, Any] = dict(
        messages=messages, model=model_name, response_model=None
    )
    assert make_cache_key(**kwargs, namespace="account-a") != make_cache_key(
        **kwargs, namespace="account-b"
    )


def test_cache_key_rejects_opaque_request_values():
    class Opaque:
        def __str__(self):
            return "same string for different values"

    with pytest.raises(TypeError, match="disable caching"):
        make_cache_key(
            messages=messages,
            model=model_name,
            response_model=None,
            request_kwargs={"config": {"callback": Opaque()}},
        )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (b"a", {"__instructor_bytes__": "61"}),
        (b"a", ["bytes", "61"]),
        ({1: "value"}, {"1": "value"}),
        (True, 1),
    ],
)
def test_cache_encoding_distinguishes_value_types(left, right):
    def key(value):
        return make_cache_key(messages=value, model=model_name, response_model=None)

    assert key(left) != key(right)


def test_genai_config_key_is_stable_across_mapping_and_sdk_object():
    types = pytest.importorskip("google.genai.types")
    kwargs: dict[str, Any] = dict(
        messages=messages, model=model_name, response_model=None
    )
    mapping = {"temperature": 0.2, "seed": 4}
    sdk_config = types.GenerateContentConfig(**mapping)
    assert make_cache_key(
        **kwargs, request_kwargs={"config": mapping}
    ) == make_cache_key(**kwargs, request_kwargs={"config": sdk_config})


def test_genai_transport_options_are_excluded_from_generation_identity():
    kwargs: dict[str, Any] = dict(
        messages=messages, model=model_name, response_model=None
    )
    plain = {"temperature": 0.2}
    transport = {**plain, "http_options": {"headers": {"Authorization": "test-only"}}}
    assert make_cache_key(**kwargs, request_kwargs={"config": plain}) == make_cache_key(
        **kwargs, request_kwargs={"config": transport}
    )


class UserV1(BaseModel):
    name: str = Field(..., description="User name")


class UserV1DiffDesc(BaseModel):
    name: str = Field(..., description="User full name")


class UserV1DiffField(BaseModel):
    name: str
    age: int


class UserDoc1(BaseModel):
    """First docstring"""

    name: str


class UserDoc2(BaseModel):
    """Second different docstring"""

    name: str


def test_cache_key_changes_on_description_change():
    k1 = make_cache_key(messages=messages, model=model_name, response_model=UserV1)
    k2 = make_cache_key(
        messages=messages, model=model_name, response_model=UserV1DiffDesc
    )
    assert k1 != k2, "Changing field description should bust the cache key"


def test_cache_key_changes_on_field_change():
    k1 = make_cache_key(messages=messages, model=model_name, response_model=UserV1)
    k2 = make_cache_key(
        messages=messages, model=model_name, response_model=UserV1DiffField
    )
    assert k1 != k2, "Adding or removing fields should bust the cache key"


def test_cache_key_same_for_identical_schema():
    k1 = make_cache_key(messages=messages, model=model_name, response_model=UserV1)
    k2 = make_cache_key(messages=messages, model=model_name, response_model=UserV1)
    assert k1 == k2, "Identical schemas should produce identical cache keys"


def test_cache_key_changes_on_docstring_change():
    k1 = make_cache_key(messages=messages, model=model_name, response_model=UserDoc1)
    k2 = make_cache_key(messages=messages, model=model_name, response_model=UserDoc2)
    assert k1 != k2, "Changing class docstring should bust the cache key"


def test_cache_key_changes_on_system_prompt_change():
    k1 = make_cache_key(
        messages=messages,
        model=model_name,
        response_model=UserV1,
        system="Always answer in French.",
    )
    k2 = make_cache_key(
        messages=messages,
        model=model_name,
        response_model=UserV1,
        system="Always answer in German.",
    )
    assert k1 != k2, "Changing the hoisted system prompt should bust the cache key"


def test_cache_key_unchanged_when_no_system_prompt():
    without = make_cache_key(messages=messages, model=model_name, response_model=UserV1)
    explicit_none = make_cache_key(
        messages=messages, model=model_name, response_model=UserV1, system=None
    )
    assert without == explicit_none, (
        "Keys must stay stable for providers without a hoisted system prompt"
    )
