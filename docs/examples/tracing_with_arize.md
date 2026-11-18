---
title: Trace Instructor with Arize AX or Phoenix
description: Inspect model timing and token usage while keeping prompt and completion content out of model spans.
---

# Trace Instructor with Arize AX or Phoenix

OpenInference can trace the OpenAI SDK calls that Instructor makes. Choose a local Phoenix collector or your Arize AX project, then instrument the SDK with the same content settings. The OpenInference instrumentor used here requires OpenAI SDK 2.8 or later.

## Choose a collector

For **local Phoenix**, start your Phoenix server and install `arize-phoenix-otel` and `openinference-instrumentation-openai` alongside Instructor. This example sends spans to the local HTTP collector.

```python
from phoenix.otel import register

tracer_provider = register(
    project_name="instructor-example",
    endpoint="http://localhost:6006/v1/traces",
    protocol="http/protobuf",
    auto_instrument=False,
)
```

For **Arize AX**, install `arize-otel` and `openinference-instrumentation-openai` alongside Instructor. Supply the space ID and API key for the AX project where you want to export spans.

```python
import os

from arize.otel import register

tracer_provider = register(
    space_id=os.environ["ARIZE_SPACE_ID"],
    api_key=os.environ["ARIZE_API_KEY"],
    project_name="instructor-example",
    auto_instrument=False,
)
```

## Trace a structured response

Run one collector setup above, set `OPENAI_API_KEY`, and instrument the SDK before creating the client. This synthetic example keeps prompt and completion values, messages, tool definitions, and invocation parameters out of model spans.

```python
import instructor
from openai import OpenAI
from openinference.instrumentation import TraceConfig
from openinference.instrumentation.openai import OpenAIInstrumentor
from pydantic import BaseModel

config = TraceConfig(
    hide_inputs=True,
    hide_outputs=True,
    hide_input_messages=True,
    hide_output_messages=True,
    hide_llm_tools=True,
    hide_llm_invocation_parameters=True,
)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider, config=config)


class WeatherDetail(BaseModel):
    city: str
    temperature: int


client = instructor.from_openai(OpenAI())
weather = client.create(
    model="gpt-6-luna",
    response_model=WeatherDetail,
    messages=[
        {"role": "user", "content": "The weather in Paris is 18 degrees Celsius."}
    ],
)
print(weather.model_dump_json())
```

Open the `instructor-example` project in your selected collector. Each model call, including a retry, can produce a span with the model name, duration and token usage. To collect message content for an evaluation dataset, choose the specific fields you intend to export and change their settings before instrumenting the SDK. [OpenInference trace configuration](https://arize-ai.github.io/openinference/spec/configuration.html)

The Arize guides cover collector setup and evaluations for each product. [Arize AX Instructor tracing](https://arize.com/docs/ax/integrations/python-agent-frameworks/instructor/instructor-tracing) · [Phoenix Instructor tracing](https://arize.com/docs/phoenix/integrations/python/instructor/instructor-tracing) · [LLM evaluation guide](https://arize.com/resources/llm-evaluation/)
