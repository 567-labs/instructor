---
title: Trace Instructor with Future AGI
description: Send a span for each Instructor create call to a Future AGI project, with the input messages, the validated output and the call duration.
---

# Trace Instructor with Future AGI

The `traceAI-instructor` package wraps `Instructor.create` and `AsyncInstructor.create` and exports one OpenTelemetry span per call to a Future AGI project. It hooks the Instructor client and not a provider SDK, so the setup is the same for every provider. This example uses OpenAI.

## Install the packages

```shell
pip install instructor "traceAI-instructor>=0.2.1"
```

Use Python 3.10 to 3.13. `traceAI-instructor` doesn't support Python 3.14 yet.

## Set your keys

Set `FI_API_KEY` and `FI_SECRET_KEY` from your Future AGI account, and `OPENAI_API_KEY` for OpenAI. `register` reads the two Future AGI keys from the environment and sends spans to `https://api.futureagi.com/tracer/v1/traces`.

## Trace a structured response

Register the tracer provider and instrument Instructor before you create the client.

```python
import instructor
from fi_instrumentation import register
from fi_instrumentation.fi_types import ProjectType
from pydantic import BaseModel
from traceai_instructor import InstructorInstrumentor

trace_provider = register(
    project_type=ProjectType.OBSERVE,
    project_name="instructor-docs-page",
)
InstructorInstrumentor().instrument(tracer_provider=trace_provider)


class UserInfo(BaseModel):
    name: str
    age: int


client = instructor.from_provider("openai/gpt-4o-mini")
user_info = client.create(
    response_model=UserInfo,
    messages=[
        {
            "role": "system",
            "content": "Extract the name and age from the text and return them in a structured format.",
        },
        {"role": "user", "content": "John Doe is nine years old."},
    ],
)
print(user_info, type(user_info))
```

## What you see in Future AGI

The first export creates the `instructor-docs-page` project as an Observe project. Each `client.create` call becomes a trace with one span. The span records the input messages, the validated JSON output, the duration and the status.

To keep prompts and responses out of the span, set `FI_HIDE_INPUTS=true` and `FI_HIDE_OUTPUTS=true` before you run the script.

[Future AGI Instructor guide](https://docs.futureagi.com/docs/integrations/traceai/instructor) · [traceAI-instructor source](https://github.com/future-agi/traceAI/tree/main/python/frameworks/instructor)
