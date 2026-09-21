"""
Persistent Memory with Dakera — Instructor Cookbook
=====================================================

This example shows how to give an Instructor-powered chatbot **long-term
memory across sessions** using Dakera, a self-hosted decay-weighted vector
memory server.

Prerequisites
-------------
1. Start Dakera locally (Docker):

       docker run -p 3300:3300 -e DAKERA_API_KEY=demo \
           ghcr.io/dakera-ai/dakera:latest

2. Install dependencies:

       pip install instructor openai httpx

3. Export your LLM key:

       export OPENAI_API_KEY=sk-...

What this example demonstrates
-------------------------------
- Storing facts as memories after each LLM exchange
- Retrieving relevant memories and injecting them as context
- Structured extraction (Pydantic models) combined with persistent memory
- A multi-turn CLI chat loop that remembers across restarts
- The "hook" approach (automatic) vs the "manual" approach
"""

from __future__ import annotations

import os
import textwrap
from typing import Optional

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

from dakera_memory import DakeraMemory, DakeraMemoryHook, build_context_messages

# ---------------------------------------------------------------------------
# Configuration — override via environment variables or edit here.
# ---------------------------------------------------------------------------
DAKERA_URL = os.environ.get("DAKERA_BASE_URL", "http://localhost:3300")
DAKERA_KEY = os.environ.get("DAKERA_API_KEY", "demo")
AGENT_ID = os.environ.get("DAKERA_AGENT_ID", "instructor-demo")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-mini")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class UserProfile(BaseModel):
    """Structured profile extracted from a user's self-introduction."""

    name: str = Field(description="The user's name")
    interests: list[str] = Field(description="List of the user's interests or hobbies")
    location: Optional[str] = Field(None, description="Where the user lives, if mentioned")
    profession: Optional[str] = Field(None, description="The user's job or profession, if mentioned")


class ChatReply(BaseModel):
    """A single assistant reply in a conversation."""

    text: str = Field(description="The assistant's reply to the user")
    should_remember: bool = Field(
        description="True if this exchange contains information worth remembering long-term"
    )
    memory_hint: Optional[str] = Field(
        None,
        description="A compact rephrasing of the key fact(s) to store if should_remember is True",
    )


# ---------------------------------------------------------------------------
# Example 1 — Manual store/recall (explicit control)
# ---------------------------------------------------------------------------


def example_manual_store_and_recall() -> None:
    """Demonstrate manually storing and recalling memories."""
    print("=" * 60)
    print("Example 1: Manual store + recall")
    print("=" * 60)

    client = instructor.from_openai(OpenAI())
    mem = DakeraMemory(base_url=DAKERA_URL, api_key=DAKERA_KEY, agent_id=AGENT_ID)

    # --- Store some facts -------------------------------------------------
    facts = [
        "The user prefers Python over JavaScript.",
        "The user is building a chatbot for their e-commerce startup.",
        "The user mentioned they have trouble with async code in Python.",
        "The user's name is Alex and they live in Berlin.",
    ]
    for fact in facts:
        record = mem.store(fact, importance=0.8, tags=["profile", "preferences"])
        print(f"Stored: {record.get('id', '?')} — {fact[:60]}")

    print()

    # --- Recall on a new question -----------------------------------------
    question = "How should I structure my Python chatbot project?"
    hits = mem.search(question, top_k=3)
    messages = build_context_messages(
        [{"role": "user", "content": question}],
        hits,
    )

    print("Messages sent to LLM:")
    for m in messages:
        role = m["role"].upper()
        content = textwrap.shorten(m["content"], width=120, placeholder="...")
        print(f"  [{role}] {content}")
    print()

    reply = client.create(
        model=LLM_MODEL,
        response_model=ChatReply,
        messages=messages,
    )
    print(f"Assistant: {reply.text}")
    print(f"Should remember: {reply.should_remember}")
    if reply.memory_hint:
        stored = mem.store(reply.memory_hint, importance=0.5, tags=["assistant-advice"])
        print(f"Stored advice memory: {stored.get('id', '?')}")

    mem.close()
    print()


# ---------------------------------------------------------------------------
# Example 2 — Hook-based (automatic, zero boilerplate)
# ---------------------------------------------------------------------------


def example_hook_based_memory() -> None:
    """Demonstrate the automatic hook approach."""
    print("=" * 60)
    print("Example 2: Hook-based automatic memory")
    print("=" * 60)

    raw_client = OpenAI()
    client = instructor.from_openai(raw_client)

    mem = DakeraMemory(base_url=DAKERA_URL, api_key=DAKERA_KEY, agent_id=f"{AGENT_ID}-hook")
    hook = DakeraMemoryHook(mem, top_k=3, auto_recall=True, auto_store=True)
    hook.attach(client)

    # The hook automatically injects recalled memories before this call
    # and stores the assistant's reply afterwards.
    reply = client.create(
        model=LLM_MODEL,
        response_model=ChatReply,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful programming assistant. Be concise.",
            },
            {
                "role": "user",
                "content": "What's the best way to handle rate limits in an async Python HTTP client?",
            },
        ],
    )
    print(f"Assistant (hook): {reply.text[:200]}...")
    print()

    # Detach when done so the hook doesn't fire on unrelated calls.
    hook.detach(client)
    mem.close()


# ---------------------------------------------------------------------------
# Example 3 — Structured extraction → memory pipeline
# ---------------------------------------------------------------------------


def example_structured_extraction_to_memory() -> None:
    """Extract a structured UserProfile and persist it as a memory."""
    print("=" * 60)
    print("Example 3: Structured extraction → memory")
    print("=" * 60)

    client = instructor.from_openai(OpenAI())
    mem = DakeraMemory(base_url=DAKERA_URL, api_key=DAKERA_KEY, agent_id=f"{AGENT_ID}-profile")

    introduction = (
        "Hi! I'm Sam, a data engineer at a fintech company in Amsterdam. "
        "I love hiking, cooking Italian food, and I've been learning Rust lately. "
        "Most of my work involves building data pipelines with Apache Spark."
    )

    profile = client.create(
        model=LLM_MODEL,
        response_model=UserProfile,
        messages=[
            {"role": "user", "content": f"Extract a user profile from: {introduction}"},
        ],
    )
    print(f"Extracted profile: {profile.model_dump_json(indent=2)}")

    # Store each interest separately for fine-grained retrieval.
    for interest in profile.interests:
        mem.store(
            f"{profile.name} is interested in {interest}.",
            importance=0.7,
            tags=["profile", "interest"],
        )
    if profile.profession:
        mem.store(
            f"{profile.name} works as {profile.profession}.",
            importance=0.9,
            tags=["profile", "profession"],
        )
    if profile.location:
        mem.store(
            f"{profile.name} lives in {profile.location}.",
            importance=0.6,
            tags=["profile", "location"],
        )

    print(f"\nStored {len(profile.interests) + 2} memories for {profile.name}")

    # Later: retrieve context for a relevant query
    query = "What programming topics might this user want help with?"
    hits = mem.search(query, top_k=4)
    messages = build_context_messages(
        [{"role": "user", "content": query}],
        hits,
    )
    recommendation = client.create(
        model=LLM_MODEL,
        response_model=ChatReply,
        messages=messages,
    )
    print(f"\nPersonalized recommendation: {recommendation.text}")

    mem.close()
    print()


# ---------------------------------------------------------------------------
# Example 4 — Multi-turn CLI chat loop with persistent memory
# ---------------------------------------------------------------------------


def chat_loop() -> None:
    """Interactive chat loop that remembers facts across restarts.

    Run this script twice: facts you share in session 1 will be recalled
    in session 2, because memories persist in Dakera between runs.
    """
    print("=" * 60)
    print("Example 4: Multi-turn chat with persistent memory")
    print("Type 'quit' to exit, 'forget' to wipe all memories.")
    print("=" * 60)

    client = instructor.from_openai(OpenAI())
    mem = DakeraMemory(
        base_url=DAKERA_URL,
        api_key=DAKERA_KEY,
        agent_id=f"{AGENT_ID}-chat",
    )

    history: list[dict[str, str]] = []
    system_prompt = {
        "role": "system",
        "content": (
            "You are a helpful assistant with long-term memory. "
            "When you see 'Relevant memories' at the top, use them to personalise your replies. "
            "Be concise and friendly."
        ),
    }

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "forget":
            mem.forget()
            history.clear()
            print("[All memories cleared]")
            continue

        history.append({"role": "user", "content": user_input})

        # Recall relevant past memories and inject them.
        hits = mem.search(user_input, top_k=3)
        messages_with_context = build_context_messages(
            [system_prompt] + history,
            hits,
        )

        reply = client.create(
            model=LLM_MODEL,
            response_model=ChatReply,
            messages=messages_with_context,
        )

        print(f"Assistant: {reply.text}")
        history.append({"role": "assistant", "content": reply.text})

        # Decide what to persist.
        if reply.should_remember and reply.memory_hint:
            stored = mem.store(reply.memory_hint, importance=0.75, tags=["chat"])
            print(f"  [Remembered: {reply.memory_hint[:80]}]")

    mem.close()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode in ("all", "1"):
        example_manual_store_and_recall()
    if mode in ("all", "2"):
        example_hook_based_memory()
    if mode in ("all", "3"):
        example_structured_extraction_to_memory()
    if mode in ("all", "chat", "4"):
        chat_loop()
