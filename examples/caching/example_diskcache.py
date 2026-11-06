"""Share Instructor's persistent response cache across sync and async clients."""

import asyncio

from pydantic import BaseModel

from instructor import from_provider
from instructor.cache import DiskCache


class UserDetail(BaseModel):
    name: str
    age: int


async def main() -> None:
    cache = DiskCache("./my_cache_directory")
    client = from_provider("openai/gpt-4.1-mini", cache=cache)
    async_client = from_provider("openai/gpt-4.1-mini", async_client=True, cache=cache)
    messages = [{"role": "user", "content": "Extract Jason is 25 years old."}]

    namespace = "example-account:api.openai.com"
    first = client.create(
        response_model=UserDetail, messages=messages, cache_namespace=namespace
    )
    second = await async_client.create(
        response_model=UserDetail, messages=messages, cache_namespace=namespace
    )
    assert first == second
    print(first)


if __name__ == "__main__":
    asyncio.run(main())
