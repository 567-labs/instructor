"""Offline installed-wheel contracts; run outside the repository checkout."""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from typing import Any, Literal

import instructor
import openai
from pydantic import BaseModel, Field


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

    class Route(BaseModel):
        team: Literal["billing", "support"] = Field(
            description="Which team handles this request?"
        )

    schema = instructor.generate_openai_schema(Result)
    assert schema["parameters"]["required"] == ["value"]
    assert instructor.response_schema(Result).model_validate({"value": 7}).value == 7

    sdk_major = int(importlib.metadata.version("openai").split(".", 1)[0])
    http = importlib.import_module("httpx2" if sdk_major >= 3 else "httpx")
    endpoint = "https://package-contract.invalid/v1"

    def completion(request: Any) -> Any:
        body = json.loads(request.content)
        if request.url.path == "/v1/decisions":
            assert (
                body["model"] == "gpt-6-luna"
                and body["input"] == "Refund the duplicate charge."
            )
            assert body["questions"][0]["name"] == "team"
            return http.Response(
                200,
                json={
                    "model": "gpt-6-luna",
                    "answers": [
                        {"name": "team", "type": "choice", "choice": "billing"}
                    ],
                    "usage": {"input_tokens": 8, "output_tokens": 0, "total_tokens": 8},
                },
            )
        assert request.url.path == "/v1/chat/completions"
        assert body["tools"][0]["function"]["name"] == "Result"
        return http.Response(
            200,
            json={
                "id": "offline-completion",
                "object": "chat.completion",
                "created": 1,
                "model": "offline-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "offline-tool",
                                    "type": "function",
                                    "function": {
                                        "name": "Result",
                                        "arguments": '{"value":7}',
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
        )

    with openai.OpenAI(
        api_key="package-contract",
        base_url=endpoint,
        http_client=http.Client(
            transport=http.MockTransport(completion), trust_env=False
        ),
    ) as client:
        wrapped = instructor.from_openai(client)
        assert isinstance(wrapped, instructor.Instructor)
        result = wrapped.chat.completions.create(
            model="offline-model",
            response_model=Result,
            messages=[{"role": "user", "content": "Return seven."}],
            max_retries=0,
        )
        assert isinstance(result, Result) and result.value == 7
        with instructor.from_openai(
            client, mode=instructor.Mode.DECISIONS
        ) as decisions:
            assert (
                decisions.create(
                    response_model=Route, input="Refund the duplicate charge."
                ).team
                == "billing"
            )

    async def check_async() -> None:
        async with openai.AsyncOpenAI(
            api_key="package-contract",
            base_url=endpoint,
            http_client=http.AsyncClient(
                transport=http.MockTransport(completion), trust_env=False
            ),
        ) as client:
            wrapped = instructor.from_openai(client)
            assert isinstance(wrapped, instructor.AsyncInstructor)
            result = await wrapped.chat.completions.create(
                model="offline-model",
                response_model=Result,
                messages=[{"role": "user", "content": "Return seven."}],
                max_retries=0,
            )
            assert isinstance(result, Result) and result.value == 7
            async with instructor.from_openai(
                client, mode=instructor.Mode.DECISIONS
            ) as decisions:
                decision = await decisions.create(
                    response_model=Route, input="Refund the duplicate charge."
                )
                assert decision.team == "billing"

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
