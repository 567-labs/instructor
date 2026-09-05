"""
Tests for dakera_memory.py — no live server or LLM required.

Run with:
    pytest test_dakera_memory.py -v
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, call

import httpx
import pytest

from dakera_memory import (
    DakeraMemory,
    DakeraMemoryHook,
    build_context_messages,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mock_response(payload: dict, status_code: int = 200) -> httpx.Response:
    """Return a minimal httpx.Response wrapping *payload*.

    httpx.Response.raise_for_status() requires ``._request`` to be set;
    we attach a minimal dummy request so the check doesn't blow up.
    """
    resp = httpx.Response(
        status_code=status_code,
        content=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )
    # Attach a dummy request so raise_for_status() can inspect it.
    resp.request = httpx.Request("POST", "http://localhost:3300/v1/memory/store")
    return resp


@pytest.fixture()
def mem(monkeypatch):
    """DakeraMemory instance with a mocked httpx.Client."""
    with patch("dakera_memory.httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        instance = DakeraMemory(
            base_url="http://localhost:3300",
            api_key="test-key",
            agent_id="test-agent",
        )
        instance._client = mock_client  # expose for assertions
        yield instance


# ---------------------------------------------------------------------------
# DakeraMemory unit tests
# ---------------------------------------------------------------------------


class TestDakeraMemoryStore:
    def test_store_calls_correct_endpoint(self, mem):
        mem._client.post.return_value = _mock_response(
            {"memory": {"id": "abc123", "content": "hello"}}
        )
        result = mem.store("hello")

        mem._client.post.assert_called_once_with(
            "/v1/memory/store",
            json={
                "content": "hello",
                "agent_id": "test-agent",
            },
        )
        assert result["id"] == "abc123"

    def test_store_with_all_optional_fields(self, mem):
        mem._client.post.return_value = _mock_response(
            {"memory": {"id": "xyz"}}
        )
        mem.store(
            "some content",
            importance=0.9,
            tags=["a", "b"],
        )
        _, kwargs = mem._client.post.call_args
        payload = kwargs["json"]
        assert payload["importance"] == 0.9
        assert payload["tags"] == ["a", "b"]

    def test_store_with_session_id(self, mem):
        mem.session_id = "session-42"
        mem._client.post.return_value = _mock_response({"memory": {}})
        mem.store("content")
        _, kwargs = mem._client.post.call_args
        assert kwargs["json"]["session_id"] == "session-42"

    def test_store_raises_on_http_error(self, mem):
        mem._client.post.return_value = _mock_response({}, status_code=500)
        with pytest.raises(httpx.HTTPStatusError):
            mem.store("bad")


class TestDakeraMemorySearch:
    def test_search_calls_correct_endpoint(self, mem):
        mem._client.post.return_value = _mock_response(
            {"memories": [{"memory": {"content": "Python tip"}, "score": 0.91}]}
        )
        results = mem.search("Python", top_k=3)

        mem._client.post.assert_called_once_with(
            "/v1/memory/search",
            json={"agent_id": "test-agent", "query": "Python", "top_k": 3},
        )
        assert len(results) == 1
        assert results[0]["score"] == 0.91

    def test_search_returns_empty_list_when_no_memories(self, mem):
        mem._client.post.return_value = _mock_response({"memories": []})
        assert mem.search("anything") == []

    def test_search_default_top_k_is_five(self, mem):
        mem._client.post.return_value = _mock_response({"memories": []})
        mem.search("query")
        _, kwargs = mem._client.post.call_args
        assert kwargs["json"]["top_k"] == 5


class TestDakeraMemoryForget:
    def test_forget_all(self, mem):
        mem._client.post.return_value = _mock_response({})
        mem.forget()
        mem._client.post.assert_called_once_with(
            "/v1/memory/forget",
            json={"agent_id": "test-agent"},
        )

    def test_forget_specific_ids(self, mem):
        mem._client.post.return_value = _mock_response({})
        mem.forget(memory_ids=["id1", "id2"])
        _, kwargs = mem._client.post.call_args
        assert kwargs["json"]["memory_ids"] == ["id1", "id2"]


class TestDakeraMemoryContextManager:
    def test_context_manager_calls_close(self):
        with patch("dakera_memory.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            with DakeraMemory(api_key="x", agent_id="y") as m:
                pass
            mock_client.close.assert_called_once()


# ---------------------------------------------------------------------------
# build_context_messages tests
# ---------------------------------------------------------------------------


class TestBuildContextMessages:
    def _make_hits(self, *contents: str) -> list[dict]:
        return [
            {"memory": {"content": c}, "score": round(0.99 - i * 0.1, 2)}
            for i, c in enumerate(contents)
        ]

    def test_prepends_system_message(self):
        messages = [{"role": "user", "content": "hello"}]
        hits = self._make_hits("Fact A")
        result = build_context_messages(messages, hits)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "Fact A" in result[0]["content"]
        assert result[1] == messages[0]

    def test_returns_original_when_no_memories(self):
        messages = [{"role": "user", "content": "hi"}]
        result = build_context_messages(messages, [])
        assert result is messages  # same object, not a copy

    def test_respects_max_memories(self):
        messages = [{"role": "user", "content": "x"}]
        hits = self._make_hits("A", "B", "C", "D", "E")
        result = build_context_messages(messages, hits, max_memories=2)
        system_text = result[0]["content"]
        assert "A" in system_text
        assert "B" in system_text
        assert "C" not in system_text

    def test_custom_header(self):
        messages = [{"role": "user", "content": "x"}]
        hits = self._make_hits("Foo")
        result = build_context_messages(messages, hits, header="My custom header:")
        assert "My custom header:" in result[0]["content"]

    def test_does_not_mutate_original_messages(self):
        original = [{"role": "user", "content": "q"}]
        build_context_messages(original, self._make_hits("X"))
        assert len(original) == 1  # unchanged

    def test_score_appears_in_output(self):
        messages = [{"role": "user", "content": "q"}]
        hits = [{"memory": {"content": "tip"}, "score": 0.87}]
        result = build_context_messages(messages, hits)
        assert "0.87" in result[0]["content"]


# ---------------------------------------------------------------------------
# DakeraMemoryHook tests
# ---------------------------------------------------------------------------


class TestDakeraMemoryHookRecall:
    def _make_mem_mock(self, hits=None):
        m = MagicMock(spec=DakeraMemory)
        m.search.return_value = hits or []
        m.store.return_value = {"id": "new-mem"}
        return m

    def test_attach_registers_handlers(self):
        client = MagicMock()
        hook = DakeraMemoryHook(self._make_mem_mock())
        hook.attach(client)
        assert client.on.call_count == 2
        calls = {c.args[0] for c in client.on.call_args_list}
        assert "completion:kwargs" in calls
        assert "completion:response" in calls

    def test_detach_removes_handlers(self):
        client = MagicMock()
        hook = DakeraMemoryHook(self._make_mem_mock())
        hook.attach(client)
        hook.detach(client)
        assert client.off.call_count == 2

    def test_on_completion_kwargs_injects_memory(self):
        hits = [{"memory": {"content": "User likes Python"}, "score": 0.95}]
        mem_mock = self._make_mem_mock(hits=hits)
        hook = DakeraMemoryHook(mem_mock, top_k=3)

        messages = [{"role": "user", "content": "Which language should I use?"}]
        kwargs = {"messages": messages}
        hook._on_completion_kwargs(**kwargs)

        mem_mock.search.assert_called_once_with(
            "Which language should I use?", top_k=3
        )
        # Messages list is mutated in-place
        assert messages[0]["role"] == "system"  # memory block prepended
        assert any(m["role"] == "user" for m in messages)

    def test_on_completion_kwargs_skips_when_no_user_message(self):
        mem_mock = self._make_mem_mock()
        hook = DakeraMemoryHook(mem_mock)
        hook._on_completion_kwargs(messages=[{"role": "system", "content": "x"}])
        mem_mock.search.assert_not_called()

    def test_on_completion_kwargs_skips_when_no_hits(self):
        mem_mock = self._make_mem_mock(hits=[])
        hook = DakeraMemoryHook(mem_mock)
        messages = [{"role": "user", "content": "hello"}]
        hook._on_completion_kwargs(messages=messages)
        assert messages[0]["role"] == "user"  # untouched

    def test_on_completion_kwargs_handles_memory_error_gracefully(self):
        mem_mock = self._make_mem_mock()
        mem_mock.search.side_effect = RuntimeError("timeout")
        hook = DakeraMemoryHook(mem_mock)
        messages = [{"role": "user", "content": "q"}]
        # Should not raise
        hook._on_completion_kwargs(messages=messages)
        assert messages[0]["role"] == "user"

    def test_on_completion_kwargs_disabled_when_auto_recall_false(self):
        mem_mock = self._make_mem_mock(hits=[{"memory": {"content": "x"}, "score": 1.0}])
        hook = DakeraMemoryHook(mem_mock, auto_recall=False)
        messages = [{"role": "user", "content": "q"}]
        hook._on_completion_kwargs(messages=messages)
        mem_mock.search.assert_not_called()


class TestDakeraMemoryHookStore:
    def _make_response(self, content: str) -> MagicMock:
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = content
        return resp

    def _make_mem_mock(self):
        m = MagicMock(spec=DakeraMemory)
        m.store.return_value = {"id": "stored"}
        return m

    def test_on_completion_response_stores_reply(self):
        mem_mock = self._make_mem_mock()
        hook = DakeraMemoryHook(mem_mock, store_importance=0.7, store_tags=["test"])
        hook._on_completion_response(self._make_response("Great answer!"))
        mem_mock.store.assert_called_once_with(
            "Great answer!", importance=0.7, tags=["test"]
        )

    def test_on_completion_response_skips_empty_content(self):
        mem_mock = self._make_mem_mock()
        hook = DakeraMemoryHook(mem_mock)
        hook._on_completion_response(self._make_response(""))
        mem_mock.store.assert_not_called()

    def test_on_completion_response_handles_error_gracefully(self):
        mem_mock = self._make_mem_mock()
        mem_mock.store.side_effect = RuntimeError("network error")
        hook = DakeraMemoryHook(mem_mock)
        # Should not raise
        hook._on_completion_response(self._make_response("content"))

    def test_on_completion_response_disabled_when_auto_store_false(self):
        mem_mock = self._make_mem_mock()
        hook = DakeraMemoryHook(mem_mock, auto_store=False)
        hook._on_completion_response(self._make_response("content"))
        mem_mock.store.assert_not_called()


class TestDakeraMemoryHookManual:
    def test_recall_into_messages_uses_last_user_message(self):
        mem_mock = MagicMock(spec=DakeraMemory)
        mem_mock.search.return_value = []
        hook = DakeraMemoryHook(mem_mock)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "resp"},
            {"role": "user", "content": "second"},
        ]
        hook.recall_into_messages(messages)
        mem_mock.search.assert_called_once_with("second", top_k=5)

    def test_recall_into_messages_accepts_explicit_query(self):
        mem_mock = MagicMock(spec=DakeraMemory)
        mem_mock.search.return_value = []
        hook = DakeraMemoryHook(mem_mock)
        messages = [{"role": "user", "content": "ignored"}]
        hook.recall_into_messages(messages, query="explicit query")
        mem_mock.search.assert_called_once_with("explicit query", top_k=5)

    def test_manual_store_delegates_to_memory(self):
        mem_mock = MagicMock(spec=DakeraMemory)
        mem_mock.store.return_value = {"id": "m1"}
        hook = DakeraMemoryHook(mem_mock, store_importance=0.5, store_tags=["default"])
        result = hook.store("a fact")
        mem_mock.store.assert_called_once_with("a fact", importance=0.5, tags=["default"])
        assert result == {"id": "m1"}

    def test_manual_store_allows_overriding_importance_and_tags(self):
        mem_mock = MagicMock(spec=DakeraMemory)
        mem_mock.store.return_value = {}
        hook = DakeraMemoryHook(mem_mock, store_importance=0.5, store_tags=["default"])
        hook.store("fact", importance=0.99, tags=["custom"])
        mem_mock.store.assert_called_once_with("fact", importance=0.99, tags=["custom"])
