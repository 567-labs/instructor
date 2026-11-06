---
title: Debugging Instructor Applications
description: Learn how to debug Instructor applications with hooks, logging, and exception handling. Practical techniques for inspecting inputs, outputs, and retries.
---

# Debugging

This guide shows how to quickly inspect inputs/outputs, capture retries, and reproduce failures when working with Instructor. It focuses on practical techniques using hooks, logging, and exception data.

## Enable Logs

### Quick Debug Mode (Recommended)

Use Python's `logging` module at the start of your script, before creating the Instructor client:

```python
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("instructor").setLevel(logging.DEBUG)
```

This enables debug messages from `instructor` and its child loggers while keeping other loggers at INFO. Instructor does not read an `INSTRUCTOR_DEBUG` environment variable.

### Manual Debug Configuration

To include debug messages from other libraries too, set the root logger to DEBUG:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("instructor").setLevel(logging.DEBUG)
```

You will see messages for:
- Raw responses (provider-specific objects)
- Handler/mode selection
- Retry attempts and parse errors
- Reask adjustments to `messages`

Tip: Set a handler/formatter to include timestamps and module names.

## Observe the Flow with Hooks

Hooks let you tap into key moments without modifying core code:

```python
from instructor.core.hooks import HookName

# Attach one or more handlers
client.on(HookName.COMPLETION_KWARGS, lambda **kw: print("KWARGS:", kw))
client.on(HookName.COMPLETION_RESPONSE, lambda resp: print("RESPONSE:", type(resp)))
client.on(HookName.PARSE_ERROR, lambda e: print("PARSE ERROR:", e))
client.on(HookName.COMPLETION_LAST_ATTEMPT, lambda e: print("LAST ATTEMPT:", e))
client.on(HookName.COMPLETION_ERROR, lambda e: print("COMPLETION ERROR:", e))
```

Common uses:
- Capture the final `kwargs` passed to the provider (including mode/tools/response_format).
- Record raw responses (e.g., to logs or a file) for offline analysis.
- Inspect parse errors and how reask modifies the next attempt.

Note: Handlers that accept `**kwargs` (or a parameter named `_instructor_meta`) receive a metadata dict with:
- `attempt_number`, `correlation_id`, `mode`, `response_model_name`.
Add `**kwargs` to your handler signature to access it:

```python
client.on(HookName.COMPLETION_KWARGS, lambda **kw: print(kw.get("_instructor_meta")))
```

## Inspect Raw Responses

Most parsed models returned by Instructor carry the original provider response for debugging:

```python
model = client.create(...)
raw = getattr(model, "_raw_response", None)
print(raw)
```

This is useful for checking provider metadata like token usage, model version, and provider-specific fields.

## Handling Failures & Retries

When all retries are exhausted, an `InstructorRetryException` is raised. It includes detailed context:

```python
from instructor.core.exceptions import InstructorRetryException

try:
    client.create(...)
except InstructorRetryException as e:
    print("Attempts:", e.n_attempts)
    print("Last completion:", e.last_completion)
    print("Create kwargs:", e.create_kwargs)  # reproducible input
    print("Failed attempts:", e.failed_attempts)  # list of (attempt, exception, completion)
    # If available, a compact trace packet to help debugging
    if hasattr(e, "trace_packet") and e.trace_packet:
        print("Trace packet:", e.trace_packet)
```

Use `e.create_kwargs` and `e.failed_attempts` to craft a minimal reproduction.

## Minimal Reproduction Template

```python
import openai
import instructor
from pydantic import BaseModel

class MyModel(BaseModel):
    # fields...
    pass

client = instructor.from_provider("openai/gpt-5-nano")

create_kwargs = {
    # paste from InstructorRetryException.create_kwargs
}

try:
    client.create(response_model=MyModel, **create_kwargs)
except Exception as err:
    # Inspect and iterate
    raise
```

This pattern captures the exact inputs that triggered a failure.

## Strict vs Non-Strict Parsing

- `strict=True` enforces exact schema matches and can surface schema drift early.
- If providers sometimes return extra fields or slightly different types (e.g., floats for ints), try `strict=False` to validate non‑strictly.

```python
client.create(..., response_model=MyModel, strict=True)
```

## Customizing Retries

You can pass an integer (attempt count) or a `tenacity` retrying object to control behavior:

```python
from tenacity import Retrying, stop_after_attempt, stop_after_delay

max_retries = Retrying(stop=stop_after_attempt(3) | stop_after_delay(10))
client.create(..., max_retries=max_retries)
```

This is helpful when balancing latency and robustness.

## Multimodal & Message Conversion

If you send images/audio/PDFs or text that may include media paths/URIs, Instructor can convert messages for provider formats.

- For supported modes, `processing.multimodal.convert_messages` runs automatically.
- If debugging content issues, log `messages` before and after conversion using the hooks above, and ensure media types/URIs are valid.

## Caching Considerations

If you’re using a cache (`cache=...`), remember:
- Successful parsed responses are stored; retrieving from cache skips the provider call.
- If debugging live provider behavior, temporarily disable cache or change the cache key (e.g., tweak a message).

```python
model = client.create(..., cache=None)
```

## Common Troubleshooting Tips

- Validate the `response_model.model_json_schema()` matches what you expect the provider to return.
- Confirm `mode` is valid for your provider; mismatches can cause parsing failures.
- Check provider‑side limits (max tokens/response length); incomplete outputs raise specific exceptions.
- If using markdown JSON (`MD_JSON`), ensure the provider is actually returning a ```json code block.

If you need deeper visibility, add a custom handler to write kwargs/responses/errors to disk with a timestamp and correlation id.

## Example: Local Debug Run

You can run a minimal, no‑network example that exercises hooks, logging, and parsing flow using a fake provider function:

- File: `examples/debugging/run.py`
- Run:

```bash
python examples/debugging/run.py
```

This script:
- Enables DEBUG logging for `instructor.*`
- Patches a fake provider `create` with `instructor.patch(mode=Mode.JSON)`
- Attaches hook handlers to print kwargs, response types, and parse errors
- Parses a simple JSON payload into a Pydantic model and prints the result
