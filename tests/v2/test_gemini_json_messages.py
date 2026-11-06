from __future__ import annotations

from pydantic import BaseModel

from instructor.v2.providers.gemini.utils import handle_gemini_json


class User(BaseModel):
    name: str


def test_handle_gemini_json_accepts_empty_messages() -> None:
    response_model, kwargs = handle_gemini_json(User, {"messages": []})

    assert response_model is User
    assert "contents" in kwargs
    assert kwargs["contents"][0]["role"] == "user"
