"""Offline installed-wheel contracts; run outside the repository checkout."""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import os
from pathlib import Path
import sys

import instructor
import openai
from pydantic import BaseModel


def main() -> None:
    package = Path(instructor.__file__).resolve().parent
    assert package.is_relative_to(Path(sys.prefix).resolve()), package
    assert (package / "py.typed").is_file()
    assert instructor.__version__ == importlib.metadata.version("instructor")

    # Resolve the public core exports in an environment without dev/provider extras.
    for name in instructor.__all__:
        getattr(instructor, name)
    for module, name in (
        ("instructor.core.client", "Instructor"),
        ("instructor.core.client", "AsyncInstructor"),
        ("instructor.function_calls", "OpenAISchema"),
        ("instructor.function_calls", "openai_schema"),
        ("instructor.dsl.partial", "Partial"),
    ):
        assert getattr(importlib.import_module(module), name) is getattr(
            instructor, name
        )

    class Result(BaseModel):
        value: int

    schema = instructor.generate_openai_schema(Result)
    assert schema["parameters"]["required"] == ["value"]
    assert instructor.response_schema(Result).model_validate({"value": 7}).value == 7

    # Real SDK construction only: no network requests and no real credentials.
    with openai.OpenAI(api_key="package-contract") as client:
        assert isinstance(instructor.from_openai(client), instructor.Instructor)

    async def check_async() -> None:
        async with openai.AsyncOpenAI(api_key="package-contract") as client:
            assert isinstance(
                instructor.from_openai(client), instructor.AsyncInstructor
            )

    asyncio.run(check_async())
    if os.environ.get("PACKAGE_EXTRAS") == "google-genai":
        assert callable(instructor.from_genai)
    else:
        assert "from_genai" not in instructor.__all__
        assert "from_anthropic" not in instructor.__all__
    for name in ("instructor", "openai", "pydantic", "pydantic-core"):
        print(f"{name}=={importlib.metadata.version(name)}")
    print(f"Installed-wheel contracts passed on Python {sys.version.split()[0]}")


if __name__ == "__main__":
    main()
