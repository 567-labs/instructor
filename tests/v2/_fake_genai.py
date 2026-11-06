"""Small Google GenAI test doubles shared by deterministic provider tests."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from types import ModuleType
from typing import Any

import pytest


class FakePart:
    def __init__(
        self,
        *,
        text: str | None = None,
        data: bytes | None = None,
        mime_type: str | None = None,
        file_uri: str | None = None,
        function_call: Any = None,
        function_response: Any = None,
    ) -> None:
        self.text = text
        self.data = data
        self.mime_type = mime_type
        self.file_uri = file_uri
        self.function_call = function_call
        self.function_response = function_response

    @classmethod
    def from_text(cls, text: str) -> FakePart:
        return cls(text=text)

    @classmethod
    def from_bytes(cls, data: bytes, mime_type: str) -> FakePart:
        return cls(data=data, mime_type=mime_type)

    @classmethod
    def from_uri(cls, file_uri: str, mime_type: str) -> FakePart:
        return cls(file_uri=file_uri, mime_type=mime_type)

    @classmethod
    def from_function_response(cls, name: str, response: dict[str, Any]) -> FakePart:
        return cls(function_response={"name": name, "response": response})


class FakeContent:
    def __init__(self, *, role: str, parts: list[Any]) -> None:
        self.role = role
        self.parts = parts


class FakeFile:
    pass


def install_fake_genai(
    monkeypatch: pytest.MonkeyPatch,
    *,
    extra_types: Mapping[str, object] | None = None,
    client_factory: type[Any] | None = None,
) -> None:
    types_module = ModuleType("google.genai.types")
    types_module.__dict__.update(
        {"Part": FakePart, "Content": FakeContent, "File": FakeFile}
    )
    if extra_types is not None:
        types_module.__dict__.update(extra_types)

    genai_module = ModuleType("google.genai")
    genai_module.__dict__["types"] = types_module
    if client_factory is not None:
        genai_module.__dict__["Client"] = client_factory

    google_module = ModuleType("google")
    google_module.__dict__["genai"] = genai_module

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_module)
