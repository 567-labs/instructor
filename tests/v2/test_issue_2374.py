from __future__ import annotations

from unittest.mock import MagicMock

from pydantic import BaseModel

import instructor
from instructor import Mode, Provider
from instructor.cache import AutoCache
from instructor.v2.core.function_calls import ResponseSchema
from instructor.v2.core.response_model import prepare_response_model
from instructor.v2.dsl.iterable import IterableBase


class User(BaseModel):
    name: str


def test_prepare_response_model_handles_list_of_basemodel() -> None:
    prepared = prepare_response_model(list[User])

    assert prepared is not None
    assert issubclass(prepared, IterableBase)


def test_prepare_response_model_handles_primitive_str() -> None:
    prepared = prepare_response_model(str)

    assert prepared is not None
    assert issubclass(prepared, ResponseSchema)


def test_patch_prepares_list_response_model_before_cache() -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create = MagicMock(
        return_value=MagicMock(text='[{"name": "test"}]')
    )
    patched = instructor.patch(mock_client, mode=Mode.MD_JSON, provider=Provider.GEMINI)

    try:
        patched.chat.completions.create(  # ty: ignore[no-matching-overload]
            response_model=list[User],
            messages=[{"role": "user", "content": "hi"}],
            cache=AutoCache(),
        )
    except AttributeError as exc:
        if "model_json_schema" in str(exc):
            raise AssertionError(f"Regression #2374: {exc}") from exc
    except Exception:
        pass


def test_patch_prepares_str_response_model_before_cache() -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create = MagicMock(return_value=MagicMock(text='"hi"'))
    patched = instructor.patch(mock_client, mode=Mode.MD_JSON, provider=Provider.GEMINI)

    try:
        patched.chat.completions.create(  # ty: ignore[no-matching-overload]
            response_model=str,
            messages=[{"role": "user", "content": "hi"}],
            cache=AutoCache(),
        )
    except AttributeError as exc:
        if "model_json_schema" in str(exc):
            raise AssertionError(f"Regression #2374: {exc}") from exc
    except Exception:
        pass
