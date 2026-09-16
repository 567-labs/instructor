---
title: Observability & Tracing with Arize
description: Trace Instructor structured-output calls with OpenInference and send them to Arize AX or Phoenix.
---

# Observability & Tracing with Arize

[Arize AI](https://arize.com/) provides tracing and evaluation for AI applications. Use **Arize AX** for managed cloud or enterprise self-hosted deployments, or **Arize Phoenix** for an open-source, self-hosted workflow. Both accept OpenInference traces, so the same Instructor instrumentation works with either destination.

This example traces Instructor calls made through the OpenAI client and exports them to Arize AX. For Phoenix, use the Phoenix endpoint and authentication settings described in the [Instructor tracing guide](https://arize.com/docs/phoenix/integrations/python/instructor/instructor-tracing).

## Setup

Install Instructor, the OpenAI client, and the Arize OpenTelemetry packages:

```shell
pip install --upgrade instructor openai pydantic arize-otel openinference-instrumentation-openai
```

Set your credentials:

```shell
export OPENAI_API_KEY="<your-openai-api-key>"
export ARIZE_SPACE_ID="<your-arize-space-id>"
export ARIZE_API_KEY="<your-arize-api-key>"
```

## Trace an Instructor call

Register the tracer and instrument OpenAI before creating the client. Instructor then adds structured-output validation while the OpenInference instrumentor records the model call.

```python
import os

import instructor
from arize.otel import register
from openai import OpenAI
from openinference.instrumentation.openai import OpenAIInstrumentor
from pydantic import BaseModel

tracer_provider = register(
    space_id=os.environ["ARIZE_SPACE_ID"],
    api_key=os.environ["ARIZE_API_KEY"],
    project_name="instructor-example",
)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

client = instructor.from_openai(OpenAI())


class WeatherDetail(BaseModel):
    city: str
    temperature: int


weather = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=WeatherDetail,
    messages=[
        {
            "role": "user",
            "content": "The weather in Paris is 18 degrees Celsius.",
        }
    ],
)

print(weather.model_dump_json(indent=2))
```

The trace includes the model request, response, token usage, latency, and Instructor retry calls. Open project `instructor-example` in AX to inspect it.

## Evaluate structured outputs

Tracing provides examples that can become evaluation datasets and regression cases. Use the [LLM evaluation guide](https://arize.com/resources/llm-evaluation/) for general model and application quality, or the [agent evaluation guide](https://arize.com/guides/ai-agent-handbook/agent-evaluation/) when Instructor is part of a tool-using or multi-step agent workflow.

## Resources

- [Arize AX Instructor tracing](https://arize.com/docs/ax/integrations/python-agent-frameworks/instructor/instructor-tracing)
- [Arize Phoenix Instructor tracing](https://arize.com/docs/phoenix/integrations/python/instructor/instructor-tracing)
- [Instructor OpenAI integration](../integrations/openai.md)
