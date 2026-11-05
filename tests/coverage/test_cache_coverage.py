from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import instructor
import pytest
from instructor.cache import (
    AutoCache,
    BaseCache,
    DiskCache,
    load_cached_response,
    make_cache_key,
    store_cached_response,
)
from pydantic import BaseModel


class User(BaseModel):
    name: str


def test_cache_key_without_response_model_is_stable_and_schema_independent() -> None:
    messages = [{"role": "user", "content": "hello"}]

    without_schema = make_cache_key(
        messages=messages, model="local", response_model=None, mode="json"
    )
    repeated = make_cache_key(
        messages=messages, model="local", response_model=None, mode="json"
    )
    with_schema = make_cache_key(
        messages=messages, model="local", response_model=User, mode="json"
    )

    assert without_schema == repeated
    assert without_schema != with_schema


class MemoryCache(BaseCache):
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}
        self.writes: list[tuple[str, Any, int | None]] = []

    def get(self, key: str) -> Any | None:
        return self.values.get(key)

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        self.values[key] = value
        self.writes.append((key, value, ttl))


def completion(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=User(name=name).model_dump_json()),
                finish_reason="stop",
            )
        ],
        usage={},
    )


def test_auto_cache_rejects_invalid_size_and_evicts_least_recently_used() -> None:
    with pytest.raises(ValueError, match="maxsize must be > 0"):
        AutoCache(maxsize=0)

    cache = AutoCache(maxsize=2)
    assert cache.get("missing") is None

    cache.set("first", "old")
    cache.set("second", "value")
    cache.set("first", "updated")
    cache.set("third", "value")

    assert cache.get("first") == "updated"
    assert cache.get("second") is None
    assert cache.get("third") == "value"


def test_disk_cache_round_trip_supports_plain_and_ttl_writes(tmp_path: Any) -> None:
    cache = DiskCache(directory=str(tmp_path / "responses"))
    try:
        assert cache.get("missing") is None

        cache.set("plain", {"name": "plain"})
        cache.set("timed", {"name": "timed"}, ttl=60)

        assert cache.get("plain") == {"name": "plain"}
        assert cache.get("timed") == {"name": "timed"}
    finally:
        cache._cache.close()


def test_sync_client_stores_on_miss_and_uses_cache_on_hit() -> None:
    calls: list[str] = []

    def complete(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        calls.append("called")
        return completion("sync")

    cache = MemoryCache()
    client = instructor.from_litellm(complete, mode=instructor.Mode.JSON)
    first = client.create(
        response_model=User,
        cache=cache,
        cache_ttl=30,
        messages=[{"role": "user", "content": "hello"}],
        model="local",
    )
    second = client.create(
        response_model=User,
        cache=cache,
        cache_ttl=30,
        messages=[{"role": "user", "content": "hello"}],
        model="local",
    )

    assert isinstance(first, User)
    assert isinstance(second, User)
    assert first.name == second.name == "sync"
    assert calls == ["called"]
    assert len(cache.writes) == 1
    assert cache.writes[0][2] == 30


@pytest.mark.asyncio
async def test_async_client_stores_on_miss_and_uses_cache_on_hit() -> None:
    calls: list[str] = []

    async def complete(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        calls.append("called")
        return completion("async")

    cache = MemoryCache()
    client = instructor.from_litellm(
        complete, mode=instructor.Mode.JSON, async_client=True
    )
    first = await client.create(
        response_model=User,
        cache=cache,
        cache_ttl=45,
        messages=[{"role": "user", "content": "hello"}],
        model="local",
    )
    second = await client.create(
        response_model=User,
        cache=cache,
        cache_ttl=45,
        messages=[{"role": "user", "content": "hello"}],
        model="local",
    )

    assert isinstance(first, User)
    assert isinstance(second, User)
    assert first.name == second.name == "async"
    assert calls == ["called"]
    assert len(cache.writes) == 1
    assert cache.writes[0][2] == 45


def test_load_cached_response_accepts_legacy_model_json() -> None:
    cache = AutoCache()
    cache.set("legacy", User(name="legacy").model_dump_json())

    restored = load_cached_response(cache, "legacy", User)

    assert restored == User(name="legacy")
    assert not hasattr(restored, "_raw_response")


def test_store_cached_response_without_raw_response_round_trips() -> None:
    cache = MemoryCache()

    store_cached_response(cache, "model-only", User(name="model-only"), ttl=15)
    payload = json.loads(cache.values["model-only"])
    restored = load_cached_response(cache, "model-only", User)

    assert payload["raw"] is None
    assert restored == User(name="model-only")
    assert cache.writes[0][2] == 15


def test_pydantic_raw_response_is_restored_as_completion_like_object() -> None:
    class RawCompletion(BaseModel):
        id: str
        choices: list[dict[str, Any]]

    cache = AutoCache()
    user = User(name="pydantic")
    object.__setattr__(
        user,
        "_raw_response",
        RawCompletion(id="chatcmpl-1", choices=[{"message": {"content": "ok"}}]),
    )

    store_cached_response(cache, "pydantic", user, ttl=10)
    restored = load_cached_response(cache, "pydantic", User)

    assert isinstance(restored, User)
    assert restored.name == "pydantic"
    assert restored._raw_response.id == "chatcmpl-1"
    assert restored._raw_response.choices[0].message.content == "ok"


def test_unserializable_raw_response_uses_string_fallback(caplog: Any) -> None:
    cache = MemoryCache()
    raw_response: list[Any] = []
    raw_response.append(raw_response)
    user = User(name="fallback")
    object.__setattr__(user, "_raw_response", raw_response)

    store_cached_response(cache, "fallback", user)
    payload = json.loads(cache.values["fallback"])
    restored = load_cached_response(cache, "fallback", User)

    assert payload["raw"] == "[[...]]"
    assert isinstance(restored, User)
    assert restored.name == "fallback"
    assert restored._raw_response == "[[...]]"
    assert "Raw response could not be serialized as JSON" in caplog.text


def test_disk_cache_rejects_unsafe_legacy_records_and_preserves_external_files(
    tmp_path, monkeypatch
):
    import sqlite3
    import diskcache
    from diskcache.core import MODE_TEXT

    directory = tmp_path / "responses"
    outside = tmp_path / "outside.txt"
    outside.write_text("sentinel")
    with diskcache.Cache(str(directory)) as legacy:
        legacy.set("old-object", {"name": "pickled"})
        legacy.set("old-text", "inline text")
        legacy.set("file", "placeholder")
    with sqlite3.connect(directory / "cache.db") as db:
        db.execute(
            "UPDATE Cache SET mode=?,filename=?,value=NULL WHERE key=?",
            (MODE_TEXT, "../outside.txt", "file"),
        )

    def unexpected_unpickle(*_args, **_kwargs):
        raise AssertionError("cache reads must not invoke pickle")

    monkeypatch.setattr(diskcache.core.pickle, "load", unexpected_unpickle)
    cache = DiskCache(directory=str(directory))
    try:
        assert cache.get("old-object") is None
        assert cache.get("old-text") == "inline text"
        assert cache.get("file") is None
        cache.set("file", b"replacement")
        assert cache.get("file") == b"replacement"
        assert outside.read_text() == "sentinel"
    finally:
        cache._cache.close()


def test_disk_cache_keeps_json_types_and_rejects_lossy_or_malformed_values(tmp_path):
    import io
    import sqlite3

    cache = DiskCache(directory=str(tmp_path))
    try:
        cache.set("json", {"items": [1, True, None]})
        assert cache.get("json") == {"items": [1, True, None]}
        with pytest.raises(TypeError):
            cache.set("tuple", (1, 2))
        with pytest.raises(TypeError):
            cache._cache.set(1, "invalid key")
        cache._cache.set("stream", io.BytesIO(b"safe"), read=True)
        assert cache.get("stream") == b"safe"
        assert "json" in list(cache._cache.iterkeys())
        with sqlite3.connect(tmp_path / "cache.db") as db:
            db.execute("UPDATE Cache SET value=? WHERE key=?", ("invalid", "json"))
            db.execute("UPDATE Cache SET raw=0 WHERE key=?", ("stream",))
        assert cache.get("json") is None
        with pytest.raises(ValueError, match="encoded cache key"):
            list(cache._cache.iterkeys())
    finally:
        cache._cache.close()


def test_disk_cache_refuses_codec_overrides_and_database_symlinks(tmp_path):
    import diskcache

    with pytest.raises(TypeError, match="inline data codec"):
        DiskCache(str(tmp_path), disk=diskcache.Disk)
    (tmp_path / "cache.db").symlink_to(tmp_path / "outside.db")
    with pytest.raises(ValueError, match="local regular files"):
        DiskCache(str(tmp_path))
