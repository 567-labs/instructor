---
title: Persistent Memory with Dakera
description: Add long-term, decay-weighted memory to Instructor applications using the Dakera open-source memory server.
---

# Persistent Memory with Dakera

By default, every `instructor.create()` call is stateless — the LLM has no
memory of previous conversations. This cookbook shows how to give your
Instructor-powered application **persistent, semantic memory** using
[Dakera](https://dakera.ai), a self-hosted vector memory server with
decay-weighted importance scoring.

## Why Dakera?

| Feature | Dakera |
|---------|--------|
| Self-hosted | Yes — Docker image, no external SaaS required |
| Decay weighting | Access-weighted importance; stale facts fade naturally |
| Semantic search | HNSW vector index for similarity recall |
| Auth | Optional Bearer token |
| Languages | REST API — any HTTP client works |

## Prerequisites

Start Dakera locally (requires Docker):

```bash
docker run -p 3300:3300 -e DAKERA_API_KEY=demo \
    ghcr.io/dakera-ai/dakera:latest
```

Install dependencies:

```bash
pip install instructor openai httpx
```

## The Dakera REST API

Three endpoints are all you need:

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/memory/store` | Store a memory |
| `POST /v1/memory/search` | Semantic search over stored memories |
| `POST /v1/memory/forget` | Delete memories |

## Integration Helper

Save the following as `dakera_memory.py` alongside your project
([full source on GitHub](https://github.com/instructor-ai/instructor/blob/main/examples/persistent_memory_dakera/dakera_memory.py)):

```python
import os
from dataclasses import dataclass, field
from typing import Any
import httpx

@dataclass
class DakeraMemory:
    base_url: str = field(
        default_factory=lambda: os.environ.get("DAKERA_BASE_URL", "http://localhost:3300")
    )
    api_key: str = field(
        default_factory=lambda: os.environ.get("DAKERA_API_KEY", "")
    )
    agent_id: str = "default"
    session_id: str | None = None
    timeout: float = 10.0

    def __post_init__(self) -> None:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self._client = httpx.Client(base_url=self.base_url, headers=headers, timeout=self.timeout)

    def store(self, content: str, *, importance: float | None = None, tags: list[str] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"content": content, "agent_id": self.agent_id}
        if self.session_id:
            payload["session_id"] = self.session_id
        if importance is not None:
            payload["importance"] = importance
        if tags is not None:
            payload["tags"] = tags
        resp = self._client.post("/v1/memory/store", json=payload)
        resp.raise_for_status()
        return resp.json().get("memory", {})

    def search(self, query: str, *, top_k: int = 5) -> list[dict[str, Any]]:
        resp = self._client.post("/v1/memory/search", json={"agent_id": self.agent_id, "query": query, "top_k": top_k})
        resp.raise_for_status()
        return resp.json().get("memories", [])

    def forget(self, memory_ids: list[str] | None = None) -> None:
        payload: dict[str, Any] = {"agent_id": self.agent_id}
        if memory_ids is not None:
            payload["memory_ids"] = memory_ids
        self._client.post("/v1/memory/forget", json=payload).raise_for_status()

    def close(self) -> None:
        self._client.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
```

## Approach 1 — Manual Store and Recall

Full explicit control: store facts yourself, recall them before each LLM call,
decide what to remember afterwards.

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional
from dakera_memory import DakeraMemory

class ChatReply(BaseModel):
    text: str
    should_remember: bool = Field(description="True if this contains a long-term fact")
    memory_hint: Optional[str] = Field(None, description="Compact version of the fact to store")

client = instructor.from_openai(OpenAI())
mem = DakeraMemory(api_key="demo", agent_id="alice")

# Store some facts from a previous session
mem.store("The user prefers Python over JavaScript.", importance=0.9, tags=["preference"])
mem.store("The user is building an e-commerce chatbot.", importance=0.8, tags=["project"])

# New query — recall relevant memories first
query = "What Python framework should I use for my chatbot backend?"
hits = mem.search(query, top_k=3)

# Build context: inject memories as a leading system message
memory_block = "\n".join(
    f"- [{h['score']:.2f}] {h['memory']['content']}"
    for h in hits
)
messages = [
    {"role": "system", "content": f"Relevant memories:\n{memory_block}"},
    {"role": "user", "content": query},
]

reply = client.create(model="gpt-4.1-mini", response_model=ChatReply, messages=messages)
print(reply.text)
# → "Given that you prefer Python and are building an e-commerce chatbot,
#     FastAPI is an excellent choice — it's fast, type-safe, and pairs well
#     with Pydantic models that instructor already uses."

# Store anything worth keeping
if reply.should_remember and reply.memory_hint:
    mem.store(reply.memory_hint, importance=0.6, tags=["advice"])
```

## Approach 2 — Hook-Based (Zero Boilerplate)

Attach a hook once; every subsequent `create()` call automatically recalls
relevant memories and stores the reply.

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel
from dakera_memory import DakeraMemory, DakeraMemoryHook

class Reply(BaseModel):
    text: str

client = instructor.from_openai(OpenAI())
mem = DakeraMemory(api_key="demo", agent_id="bob")
hook = DakeraMemoryHook(mem, top_k=3, auto_recall=True, auto_store=True)
hook.attach(client)

# Memory is injected automatically — no extra code needed
reply = client.create(
    model="gpt-4.1-mini",
    response_model=Reply,
    messages=[{"role": "user", "content": "What did I tell you about my project?"}],
)
print(reply.text)

# Clean up when done
hook.detach(client)
mem.close()
```

### How the Hook Works

The `DakeraMemoryHook` registers two instructor event handlers:

| Hook event | Action |
|-----------|--------|
| `completion:kwargs` | Extracts the last user message, searches Dakera, prepends a system memory block |
| `completion:response` | Extracts the assistant reply text, stores it with configurable importance |

```python
# Hook constructor options
hook = DakeraMemoryHook(
    mem,
    top_k=5,               # how many memories to inject
    auto_recall=True,       # inject memories before each call
    auto_store=True,        # store replies after each call
    store_importance=0.6,   # decay weight for auto-stored replies
    store_tags=["instructor"],
)
```

## Approach 3 — Structured Extraction Pipeline

Use Instructor's structured output to extract a `UserProfile` Pydantic model,
then persist each field as a separate searchable memory for fine-grained recall.

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional
from dakera_memory import DakeraMemory

class UserProfile(BaseModel):
    name: str
    interests: list[str]
    location: Optional[str] = None
    profession: Optional[str] = None

client = instructor.from_openai(OpenAI())
mem = DakeraMemory(api_key="demo", agent_id="carol")

intro = "I'm Sam, a data engineer in Amsterdam. I love hiking, cooking, and Rust."
profile = client.create(
    model="gpt-4.1-mini",
    response_model=UserProfile,
    messages=[{"role": "user", "content": f"Extract a profile from: {intro}"}],
)

# Store each fact separately for fine-grained retrieval
for interest in profile.interests:
    mem.store(f"{profile.name} is interested in {interest}.", importance=0.7, tags=["interest"])
if profile.profession:
    mem.store(f"{profile.name} works as {profile.profession}.", importance=0.9, tags=["profession"])
if profile.location:
    mem.store(f"{profile.name} lives in {profile.location}.", importance=0.6, tags=["location"])
```

## Approach 4 — Persistent Multi-Turn Chat

A complete CLI chatbot that remembers facts across restarts.
Run it twice: what you say in session 1 is recalled in session 2.

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional
from dakera_memory import DakeraMemory, build_context_messages

class ChatReply(BaseModel):
    text: str
    should_remember: bool
    memory_hint: Optional[str] = None

client = instructor.from_openai(OpenAI())
mem = DakeraMemory(api_key="demo", agent_id="persistent-chat")
history = []
system = {"role": "system", "content": "You are a helpful assistant with long-term memory."}

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("quit", "exit"):
        break

    history.append({"role": "user", "content": user_input})
    hits = mem.search(user_input, top_k=3)
    messages = build_context_messages([system] + history, hits)

    reply = client.create(model="gpt-4.1-mini", response_model=ChatReply, messages=messages)
    print(f"Assistant: {reply.text}")
    history.append({"role": "assistant", "content": reply.text})

    if reply.should_remember and reply.memory_hint:
        mem.store(reply.memory_hint, importance=0.75, tags=["chat"])
        print(f"  [Remembered: {reply.memory_hint}]")

mem.close()
```

## Complete Working Example

The full source code for all four approaches, plus a test suite, is available at:

- `examples/persistent_memory_dakera/dakera_memory.py` — core helpers
- `examples/persistent_memory_dakera/run.py` — cookbook examples 1–4
- `examples/persistent_memory_dakera/test_dakera_memory.py` — pytest suite

## Key Patterns

### Memory decay

Dakera weights memories by access frequency and configurable importance (0–1).
Rarely-accessed memories naturally become less influential without manual cleanup.
Use higher importance values for critical user facts:

```python
mem.store("User's name is Alice", importance=0.95)       # durable
mem.store("User asked about the weather", importance=0.2) # ephemeral
```

### Session scoping

Pass `session_id` to scope memories to the current conversation, while
`agent_id` gives cross-session identity:

```python
mem = DakeraMemory(api_key="demo", agent_id="alice", session_id="session-2026-07-01")
```

### Error resilience

Never let a memory failure crash your LLM call. The hook implementation
catches all exceptions from Dakera and logs a warning:

```python
try:
    hits = self.memory.search(user_text, top_k=self.top_k)
except Exception as exc:
    logger.warning("Dakera memory search failed: %s", exc)
    return  # continue without memory context
```
