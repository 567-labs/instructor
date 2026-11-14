---
title: Logging and Monitoring with Instructor - Debug Guide
description: Implement comprehensive logging for Instructor LLM calls. Track API usage, debug issues, and monitor performance with DEBUG level logging.
---

Enable DEBUG on the Instructor logger to inspect request preparation and response processing. The request helper logs the mode and response model. Provider SDKs and HTTP clients have their own loggers. Enabling DEBUG on the root logger can also capture their HTTP headers and payloads.

```python
import instructor
import logging

from pydantic import BaseModel

# Set logging to DEBUG
logging.basicConfig(level=logging.INFO)
logging.getLogger("instructor").setLevel(logging.DEBUG)

client = instructor.from_provider("openai/gpt-6-luna")


class UserDetail(BaseModel):
    name: str
    age: int


user = client.create(
    response_model=UserDetail,
    messages=[
        {"role": "user", "content": "Extract Jason is 25 years old"},
    ],
)  # type: ignore
```

## Inspecting raw responses

To include full provider responses from the sync and async response helpers, set
`INSTRUCTOR_LOG_RAW_RESPONSES=1` in the application environment and enable DEBUG
on the `instructor` logger. Use this for local debugging with data you can safely
write to logs. The flag controls Instructor's raw-response messages. Provider SDK
loggers have separate settings.

## Provider initialization logs

`from_provider()` now emits structured logs at the `INFO` level when a provider
is initialized. Enable logging to see which provider and model are being used.

```python
import logging
import instructor

logging.basicConfig(level=logging.INFO)

instructor.from_provider("openai/gpt-6-luna")
```

Example output:

```
INFO:instructor.auto_client:Initializing openai provider with model gpt-6-luna
INFO:instructor.auto_client:Client initialized
```
