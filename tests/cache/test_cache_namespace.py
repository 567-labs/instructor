from typing import Any

import openai
import pytest
from pydantic import BaseModel

import instructor
from instructor.cache import AutoCache


class Answer(BaseModel):
    value: int


@pytest.mark.parametrize("namespace", [None, "", 123])
def test_sync_rejects_invalid_namespace(namespace: Any) -> None:
    with openai.OpenAI(api_key="test-only", base_url="http://127.0.0.1:1") as sdk:
        client = instructor.from_openai(sdk)
        with pytest.raises(ValueError, match="cache_namespace"):
            client.create(
                model="test",
                response_model=Answer,
                messages=[{"role": "user", "content": "42"}],
                cache=AutoCache(),
                cache_namespace=namespace,
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("namespace", [None, "", 123])
async def test_async_rejects_invalid_namespace(namespace: Any) -> None:
    async with openai.AsyncOpenAI(
        api_key="test-only", base_url="http://127.0.0.1:1"
    ) as sdk:
        client = instructor.from_openai(sdk)
        with pytest.raises(ValueError, match="cache_namespace"):
            await client.create(
                model="test",
                response_model=Answer,
                messages=[{"role": "user", "content": "42"}],
                cache=AutoCache(),
                cache_namespace=namespace,
            )
