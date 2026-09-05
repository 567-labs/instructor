---
title: Observability & Tracing with Langfuse
description: Learn how to trace and monitor Instructor API calls using Langfuse for comprehensive observability in your LLM applications.
---

# Observability & Tracing with Langfuse

**What is Langfuse?**

> **What is Langfuse?** [Langfuse](https://langfuse.com) ([GitHub](https://github.com/langfuse/langfuse)) is an open source LLM engineering platform that helps teams trace API calls, monitor performance, and debug issues in their AI applications.

![Instructor Trace in Langfuse showing structured output monitoring and observability](https://langfuse.com/images/docs/instructor-trace.png)

This cookbook shows how to use Langfuse to trace and monitor model calls made with the Instructor library.

## Setup

> **Note** : Before continuing with this section, make sure that you've signed up for an account with [Langfuse](https://langfuse.com). You'll need your private and public key to start tracing with Langfuse.

First, let's start by installing the necessary dependencies.

```shell
pip install --upgrade langfuse openai pydantic instructor
```

It is easy to use instructor with Langfuse. We use the [Langfuse OpenAI Integration](https://langfuse.com/docs/integrations/openai) and simply patch the client with instructor. This works with both synchronous and asynchronous clients.

### Langfuse-Instructor integration with synchronous OpenAI client

```python
import os

import instructor
from langfuse.openai import OpenAI
from pydantic import BaseModel

# Set your API keys Here
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-..."
os.environ["LANGFUSE_SECRET_KEY"] = "sk-..."
os.environ["LANGFUSE_BASE_URL"] = "https://us.cloud.langfuse.com"
os.environ["OPENAI_API_KEY"] = "sk-..."

# Patch Langfuse wrapper of synchronous OpenAI client with instructor
client = instructor.patch(OpenAI())


class WeatherDetail(BaseModel):
    city: str
    temperature: int


# Run synchronous OpenAI client
weather_info = client.chat.completions.create(
    model="gpt-4o",
    response_model=WeatherDetail,
    messages=[
        {"role": "user", "content": "The weather in Paris is 18 degrees Celsius."},
    ],
)

print(weather_info.model_dump_json(indent=2))
"""
{
  "city": "Paris",
  "temperature": 18
}
"""
```

Once we've run this request successfully, we'll see that we have a trace available in the Langfuse dashboard for you to look at.

### Langfuse-Instructor integration with asynchronous OpenAI client

```python
import asyncio
import os

import instructor
from langfuse.openai import AsyncOpenAI
from pydantic import BaseModel

# Set your API keys Here
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-"
os.environ["LANGFUSE_BASE_URL"] = "https://us.cloud.langfuse.com"
os.environ["OPENAI_API_KEY"] = "sk-..."


# Patch Langfuse wrapper of asynchronous OpenAI client with instructor
client = instructor.patch(AsyncOpenAI())


class WeatherDetail(BaseModel):
    city: str
    temperature: int


async def main():
    # Run asynchronous OpenAI client
    weather_info = await client.chat.completions.create(
        model="gpt-4o",
        response_model=WeatherDetail,
        messages=[
            {"role": "user", "content": "The weather in Paris is 18 degrees Celsius."},
        ],
    )

    print(weather_info.model_dump_json(indent=2))
    """
    {
    "city": "Paris",
    "temperature": 18
    }
    """


asyncio.run(main())
```

Here's a [public link](https://cloud.langfuse.com/project/cloramnkj0002jz088vzn1ja4/traces/0da3f599-b807-4e14-9888-cf68fa53d976?timestamp=2025-03-31T16:12:40.076Z&display=details) to the trace that we generated which you can view in Langfuse.

## Example

In this example, we first classify customer feedback into categories like `PRAISE`, `SUGGESTION`, `BUG` and `QUESTION`, and further scores the relevance of each feedback to the business on a scale of 0.0 to 1.0. In this case, we use the asynchronous OpenAI client `AsyncOpenAI` to classify and evaluate the feedback.

```python
import asyncio
from enum import Enum
import os

import instructor
from langfuse import get_client, observe
from langfuse.openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator

# Set your API keys Here
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-..."
os.environ["LANGFUSE_SECRET_KEY"] = "sk-..."
os.environ["LANGFUSE_BASE_URL"] = "https://us.cloud.langfuse.com"
os.environ["OPENAI_API_KEY"] = "sk-..."


# Patch the Langfuse wrapper with Instructor.
client = instructor.patch(AsyncOpenAI(), mode=instructor.Mode.TOOLS)

# Initialize Langfuse (needed for scoring)
langfuse = get_client()

# Rate limit the number of requests
sem = asyncio.Semaphore(5)


# Define feedback categories
class FeedbackType(Enum):
    PRAISE = "PRAISE"
    SUGGESTION = "SUGGESTION"
    BUG = "BUG"
    QUESTION = "QUESTION"


# Model for feedback classification
class FeedbackClassification(BaseModel):
    feedback_text: str = Field(...)
    classification: list[FeedbackType] = Field(
        description="Predicted categories for the feedback"
    )
    relevance_score: float = Field(
        default=0.0,
        description="Score of the query evaluating its relevance to the business between 0.0 and 1.0",
    )

    # Make sure feedback type is list
    @field_validator("classification", mode="before")
    def validate_classification(cls, v):
        if not isinstance(v, list):
            v = [v]
        return v


@observe()  # Langfuse decorator to automatically log spans to Langfuse
async def classify_feedback(feedback: str):
    """
    Classify customer feedback into categories and evaluate relevance.
    """
    async with sem:  # simple rate limiting
        response = await client.chat.completions.create(
            model="gpt-4o",
            response_model=FeedbackClassification,
            max_retries=2,
            messages=[
                {
                    "role": "user",
                    "content": f"Classify and score this feedback: {feedback}",
                },
            ],
        )

        # Retrieve observation_id of current span
        observation_id = langfuse.get_current_observation_id()

        return feedback, response, observation_id


def score_relevance(trace_id: str, observation_id: str, relevance_score: float):
    """
    Score the relevance of a feedback query in Langfuse given the observation_id.
    """
    langfuse.create_score(
        trace_id=trace_id,
        observation_id=observation_id,
        name="feedback-relevance",
        value=relevance_score,
    )


@observe()  # Langfuse decorator to automatically log trace to Langfuse
async def main(feedbacks: list[str]):
    tasks = [classify_feedback(feedback) for feedback in feedbacks]
    results = []

    for task in asyncio.as_completed(tasks):
        feedback, classification, observation_id = await task
        result = {
            "feedback": feedback,
            "classification": [c.value for c in classification.classification],
            "relevance_score": classification.relevance_score,
        }
        results.append(result)

        # Retrieve trace_id of current trace
        trace_id = langfuse.get_current_trace_id()

        # Score the relevance of the feedback in Langfuse
        score_relevance(trace_id, observation_id, classification.relevance_score)

    # Flush observations to Langfuse
    langfuse.flush()
    return results


feedback_messages = [
    "The chat bot on your website does not work.",
    "Your customer service is exceptional!",
    "Could you add more features to your app?",
    "I have a question about my recent order.",
]

feedback_classifications = asyncio.run(main(feedback_messages))

for classification in feedback_classifications:
    print(f"Feedback: {classification['feedback']}")
    print(f"Classification: {classification['classification']}")
    print(f"Relevance Score: {classification['relevance_score']}")


"""
Feedback: I have a question about my recent order.
Classification: ['QUESTION']
Relevance Score: 0.0
Feedback: Could you add more features to your app?
Classification: ['SUGGESTION']
Relevance Score: 0.0
Feedback: The chat bot on your website does not work.
Classification: ['BUG']
Relevance Score: 0.9
Feedback: Your customer service is exceptional!
Classification: ['PRAISE']
Relevance Score: 0.9
"""
```

We can see that with Langfuse, we were able to generate these different completions and view them with our own UI. Click here to see the [public trace](https://cloud.langfuse.com/project/cloramnkj0002jz088vzn1ja4/traces/ba27e7b1-e23e-4f50-87de-420cf038190f?timestamp=2025-03-31T16:12:57.041Z&display=details) for the 5 completions that we generated.
