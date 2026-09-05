"""
dakera_memory.py — Dakera persistent memory integration for Instructor.

This module provides:
- DakeraMemory: a lightweight client wrapping the Dakera REST API
- DakeraMemoryHook: a hook-based mixin that auto-stores + auto-retrieves
  memories around every instructor.create() / instructor.chat.completions.create() call
- build_context_messages(): a helper to inject recalled memories as a
  system message into a message list before calling the LLM

Usage
-----
```python
import instructor
from dakera_memory import DakeraMemory, DakeraMemoryHook

client = instructor.from_provider("openai/gpt-4.1-mini")
mem = DakeraMemory(base_url="http://localhost:3300", api_key="demo", agent_id="my-agent")
hook = DakeraMemoryHook(mem)
hook.attach(client)

from pydantic import BaseModel

class Reply(BaseModel):
    text: str

messages = [{"role": "user", "content": "What did I have for breakfast?"}]
messages = hook.recall_into_messages(messages)

reply = client.create(messages=messages, response_model=Reply)
hook.store("What did I have for breakfast?", reply.text)
```
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:3300"
_DEFAULT_TOP_K = 5


# ---------------------------------------------------------------------------
# Core REST client
# ---------------------------------------------------------------------------


@dataclass
class DakeraMemory:
    """Thin wrapper around the Dakera REST API.

    Args:
        base_url:  Base URL of the Dakera server (default ``http://localhost:3300``).
        api_key:   Bearer token.  Falls back to the ``DAKERA_API_KEY`` env var.
        agent_id:  Logical agent / user identifier used to namespace memories.
        session_id: Optional session scope for the current conversation.
        timeout:   HTTP timeout in seconds (default 10).
    """

    base_url: str = field(default_factory=lambda: os.environ.get("DAKERA_BASE_URL", _DEFAULT_BASE_URL))
    api_key: str = field(default_factory=lambda: os.environ.get("DAKERA_API_KEY", ""))
    agent_id: str = "default"
    session_id: str | None = None
    timeout: float = 10.0

    def __post_init__(self) -> None:
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=self._auth_headers(),
            timeout=self.timeout,
        )

    def _auth_headers(self) -> dict[str, str]:
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(
        self,
        content: str,
        *,
        importance: float | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Store a memory and return the created record.

        Args:
            content:    The text to remember.
            importance: Optional 0-1 weight hint for the decay engine.
            tags:       Optional list of tag strings.

        Returns:
            The full memory record dict as returned by the server.
        """
        payload: dict[str, Any] = {
            "content": content,
            "agent_id": self.agent_id,
        }
        if self.session_id is not None:
            payload["session_id"] = self.session_id
        if importance is not None:
            payload["importance"] = importance
        if tags is not None:
            payload["tags"] = tags

        resp = self._client.post("/v1/memory/store", json=payload)
        resp.raise_for_status()
        return resp.json().get("memory", {})

    def search(self, query: str, *, top_k: int = _DEFAULT_TOP_K) -> list[dict[str, Any]]:
        """Search memories by semantic similarity.

        Args:
            query:  Natural-language query string.
            top_k:  Maximum number of results to return (default 5).

        Returns:
            List of ``{"memory": {...}, "score": float}`` dicts, most relevant first.
        """
        payload: dict[str, Any] = {
            "agent_id": self.agent_id,
            "query": query,
            "top_k": top_k,
        }
        resp = self._client.post("/v1/memory/search", json=payload)
        resp.raise_for_status()
        return resp.json().get("memories", [])

    def forget(self, memory_ids: list[str] | None = None) -> None:
        """Delete specific memories (or all memories for the agent if omitted).

        Args:
            memory_ids: List of memory IDs to delete.  Pass ``None`` to forget everything.
        """
        payload: dict[str, Any] = {"agent_id": self.agent_id}
        if memory_ids is not None:
            payload["memory_ids"] = memory_ids
        resp = self._client.post("/v1/memory/forget", json=payload)
        resp.raise_for_status()

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def __enter__(self) -> "DakeraMemory":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Async variant
# ---------------------------------------------------------------------------


@dataclass
class AsyncDakeraMemory:
    """Async counterpart of :class:`DakeraMemory` — use inside ``async`` code."""

    base_url: str = field(default_factory=lambda: os.environ.get("DAKERA_BASE_URL", _DEFAULT_BASE_URL))
    api_key: str = field(default_factory=lambda: os.environ.get("DAKERA_API_KEY", ""))
    agent_id: str = "default"
    session_id: str | None = None
    timeout: float = 10.0

    def __post_init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
            timeout=self.timeout,
        )

    async def store(
        self,
        content: str,
        *,
        importance: float | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"content": content, "agent_id": self.agent_id}
        if self.session_id is not None:
            payload["session_id"] = self.session_id
        if importance is not None:
            payload["importance"] = importance
        if tags is not None:
            payload["tags"] = tags
        resp = await self._client.post("/v1/memory/store", json=payload)
        resp.raise_for_status()
        return resp.json().get("memory", {})

    async def search(self, query: str, *, top_k: int = _DEFAULT_TOP_K) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {"agent_id": self.agent_id, "query": query, "top_k": top_k}
        resp = await self._client.post("/v1/memory/search", json=payload)
        resp.raise_for_status()
        return resp.json().get("memories", [])

    async def forget(self, memory_ids: list[str] | None = None) -> None:
        payload: dict[str, Any] = {"agent_id": self.agent_id}
        if memory_ids is not None:
            payload["memory_ids"] = memory_ids
        resp = await self._client.post("/v1/memory/forget", json=payload)
        resp.raise_for_status()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncDakeraMemory":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_context_messages(
    messages: list[dict[str, Any]],
    memories: list[dict[str, Any]],
    *,
    header: str = "Relevant memories from previous conversations:",
    max_memories: int = 5,
) -> list[dict[str, Any]]:
    """Prepend a system message containing recalled memories to *messages*.

    The injected system message is inserted **before** any existing system
    messages so the LLM sees the memory context first.

    Args:
        messages:     The original message list to be passed to the LLM.
        memories:     Result of :meth:`DakeraMemory.search` — list of
                      ``{"memory": {"content": ...}, "score": float}`` dicts.
        header:       Heading line that introduces the memory block.
        max_memories: Cap on how many memories to inject (default 5).

    Returns:
        A new message list with a system memory block prepended.
    """
    if not memories:
        return messages

    items = memories[:max_memories]
    lines = [header, ""]
    for i, hit in enumerate(items, 1):
        content = hit.get("memory", {}).get("content", "")
        score = hit.get("score", 0.0)
        lines.append(f"{i}. [{score:.2f}] {content}")

    memory_message: dict[str, Any] = {
        "role": "system",
        "content": "\n".join(lines),
    }

    # Keep existing system messages; prepend the memory block.
    return [memory_message] + list(messages)


# ---------------------------------------------------------------------------
# Hook-based integration
# ---------------------------------------------------------------------------


class DakeraMemoryHook:
    """Instructor hook that automatically stores and retrieves memories.

    Attach to any instructor-patched client once; thereafter every
    ``create()`` call will have past memories woven into the system context.

    Typical usage::

        client = instructor.from_provider("openai/gpt-4.1-mini")
        mem = DakeraMemory(api_key="demo", agent_id="alice")
        hook = DakeraMemoryHook(mem, top_k=3)
        hook.attach(client)

    When a ``completion:kwargs`` event fires:

    1. The hook extracts the last user message as the query.
    2. It searches Dakera for relevant memories.
    3. It injects those memories as a leading system message.

    When a ``completion:response`` event fires:

    1. The hook extracts the assistant's reply text.
    2. It stores that reply as a new memory (with ``importance=0.6``).

    You can disable auto-recall or auto-store independently::

        hook = DakeraMemoryHook(mem, auto_recall=True, auto_store=False)

    If you prefer full manual control, skip :meth:`attach` and call
    :meth:`recall_into_messages` / :meth:`store` yourself.
    """

    def __init__(
        self,
        memory: DakeraMemory,
        *,
        top_k: int = _DEFAULT_TOP_K,
        auto_recall: bool = True,
        auto_store: bool = True,
        store_importance: float = 0.6,
        store_tags: list[str] | None = None,
    ) -> None:
        self.memory = memory
        self.top_k = top_k
        self.auto_recall = auto_recall
        self.auto_store = auto_store
        self.store_importance = store_importance
        self.store_tags = store_tags or ["instructor"]

    # ------------------------------------------------------------------
    # Instructor hook handlers
    # ------------------------------------------------------------------

    def _on_completion_kwargs(self, *args: Any, **kwargs: Any) -> None:
        """Fires before each LLM call — inject memory context."""
        if not self.auto_recall:
            return
        messages: list[dict[str, Any]] = kwargs.get("messages", [])
        if not messages:
            return

        # Use the last user message as the semantic query.
        user_text = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
            None,
        )
        if not user_text:
            return

        try:
            hits = self.memory.search(user_text, top_k=self.top_k)
        except Exception as exc:  # never crash the LLM call on a memory error
            logger.warning("Dakera memory search failed: %s", exc)
            return

        if not hits:
            return

        enriched = build_context_messages(messages, hits, max_memories=self.top_k)
        # Mutate the messages list in-place so instructor sees the update.
        messages.clear()
        messages.extend(enriched)

    def _on_completion_response(self, response: Any) -> None:
        """Fires after a successful LLM call — persist the reply."""
        if not self.auto_store:
            return
        try:
            # Works with openai ChatCompletion objects and most other providers.
            choice = response.choices[0]
            text: str = ""
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                text = choice.message.content or ""
            elif hasattr(choice, "text"):
                text = choice.text or ""
            if not text:
                return
            self.memory.store(
                text,
                importance=self.store_importance,
                tags=self.store_tags,
            )
        except Exception as exc:
            logger.warning("Dakera memory store failed: %s", exc)

    def attach(self, client: Any) -> None:
        """Register hooks on an instructor-patched *client*.

        Args:
            client: Any object returned by ``instructor.from_provider()``,
                    ``instructor.from_openai()``, etc.
        """
        client.on("completion:kwargs", self._on_completion_kwargs)
        client.on("completion:response", self._on_completion_response)

    def detach(self, client: Any) -> None:
        """Remove previously registered hooks from *client*."""
        client.off("completion:kwargs", self._on_completion_kwargs)
        client.off("completion:response", self._on_completion_response)

    # ------------------------------------------------------------------
    # Manual helpers (bypass hooks)
    # ------------------------------------------------------------------

    def recall_into_messages(
        self,
        messages: list[dict[str, Any]],
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return *messages* with a prepended memory-context system message.

        Useful when you want manual control instead of automatic hooks.

        Args:
            messages: The original message list.
            query:    Query to search memories with.  Defaults to the last
                      user message in *messages*.

        Returns:
            A new message list (original is not mutated).
        """
        if query is None:
            query = next(
                (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
                None,
            )
        if not query:
            return messages

        hits = self.memory.search(query, top_k=self.top_k)
        return build_context_messages(messages, hits, max_memories=self.top_k)

    def store(
        self,
        content: str,
        *,
        importance: float | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Manually store *content* as a memory.

        Args:
            content:    Text to persist.
            importance: Override the default importance weight.
            tags:       Override the default tags.

        Returns:
            The created memory record.
        """
        return self.memory.store(
            content,
            importance=importance if importance is not None else self.store_importance,
            tags=tags if tags is not None else self.store_tags,
        )
