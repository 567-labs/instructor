# Persistent memory with Dakera

This example retrieves earlier statements from one Dakera namespace, sends them as reference data to an Instructor model, and returns a validated Pydantic response. The caller can select an exact quote from the current message to store after validation.

## Connect the services

Run a Dakera server and install the official Python SDK. From a checkout of Instructor, set `DAKERA_API_KEY`, `DAKERA_AGENT_ID`, and `OPENAI_API_KEY`. The example uses the local Dakera server at `http://localhost:3300` unless you set `DAKERA_BASE_URL`. Use a Dakera key scoped to the namespaces your application is allowed to access.

```shell
uv run --with 'dakera>=0.14,<0.15' python -m examples.persistent_memory_dakera.run
```

## Recall and store a confirmed quote

Choose `agent_id` from the authenticated application context. Each request supplies its own namespace, current message and optional quote. Retrieved text is JSON-encoded inside a user message. The model provider receives those statements along with the current question.

```python
from examples.persistent_memory_dakera.dakera_memory import create_with_memory
from examples.persistent_memory_dakera.run import Reply

reply = create_with_memory(
    model, memory, Reply,
    agent_id=authenticated_agent_id,
    message="I prefer dark mode. Which display setting should I use?",
    remember_quote="I prefer dark mode.",
)
```

Omit `remember_quote` for a read-only turn. For a profile or extraction task, pass your own Pydantic response model. A failed model call or validation leaves storage untouched. A confirmed quote is stored as supplied, so a reviewer can trace it back to the current input.

## Use asynchronous clients

Use `AsyncDakeraClient` and an async Instructor client. The helper awaits recall, the model response and any requested write in order.

```python
from examples.persistent_memory_dakera.dakera_memory import create_with_memory_async

reply = await create_with_memory_async(
    async_model, async_memory, Reply,
    agent_id=authenticated_agent_id,
    message="What display preference did I mention?",
)
```

## Forget a stored statement

Use the ID of the record you want to remove and the same application-selected namespace. The SDK sends the scoped deletion to Dakera.

```python
memory.forget(agent_id=authenticated_agent_id, memory_id=selected_memory_id)
```

The runnable example uses the public methods in Dakera Python 0.14. [Example source](https://github.com/567-labs/instructor/tree/main/examples/persistent_memory_dakera) · [Dakera Python SDK](https://github.com/dakera-ai/dakera-py) · [Dakera REST API](https://dakera.ai/docs/api)
