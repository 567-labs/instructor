import pytest
from pydantic import BaseModel, ValidationError, ValidationInfo, field_validator

from instructor.cache import AutoCache, load_cached_response, store_cached_response


class AllowedValue(BaseModel):
    value: int

    @field_validator("value")
    @classmethod
    def check_limit(cls, value: int, info: ValidationInfo) -> int:
        if info.context and value > info.context["limit"]:
            raise ValueError("value exceeds current limit")
        return value


def test_cache_hit_uses_current_validation_context() -> None:
    cache = AutoCache()
    store_cached_response(cache, "policy", AllowedValue(value=10))
    assert (
        load_cached_response(cache, "policy", AllowedValue, context={"limit": 20}).value
        == 10
    )
    with pytest.raises(ValidationError, match="current limit"):
        load_cached_response(cache, "policy", AllowedValue, context={"limit": 5})


def test_cache_hit_respects_strict_validation() -> None:
    cache = AutoCache()
    cache.set("legacy", '{"value": "10"}')
    assert load_cached_response(cache, "legacy", AllowedValue, strict=False).value == 10
    with pytest.raises(ValidationError):
        load_cached_response(cache, "legacy", AllowedValue, strict=True)
