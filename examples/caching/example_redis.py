"""Store Instructor's serialized responses in Redis with per-request TTLs."""

from typing import Any

from pydantic import BaseModel
from redis import Redis

from instructor import from_provider
from instructor.cache import BaseCache


class RedisCache(BaseCache):
    def __init__(self, store: Redis) -> None:
        self.store = store

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        self.store.set(key, value, ex=ttl)


class UserDetail(BaseModel):
    name: str
    age: int


if __name__ == "__main__":
    with Redis(host="localhost", decode_responses=True) as store:
        client = from_provider("openai/gpt-6-luna", cache=RedisCache(store))
        messages = [{"role": "user", "content": "Extract Jason is 25 years old."}]
        first = client.create(
            response_model=UserDetail, messages=messages, cache_ttl=3600
        )
        assert client.create(response_model=UserDetail, messages=messages) == first
        print(first)
