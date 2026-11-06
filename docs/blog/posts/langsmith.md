---
authors:
- jxnl
categories:
- LLM Techniques
comments: true
date: 2024-02-18
description: Explore how LangSmith enhances OpenAI clients with seamless LLM observability
  and the `instructor` package for question classification.
draft: false
tags:
- LangSmith
- OpenAI
- LLM
- Python
- API Development
---

# Seamless Support with Langsmith

Its a common misconception that LangChain's [LangSmith](https://www.langchain.com/langsmith) is only compatible with LangChain's models. In reality, LangSmith is a unified DevOps platform for developing, collaborating, testing, deploying, and monitoring LLM applications. In this blog we will explore how LangSmith can be used to enhance the OpenAI client alongside `instructor`.

<!-- more -->

## LangSmith

In order to use langsmith, you first need to set your LangSmith API key.

```
export LANGCHAIN_API_KEY=<your-api-key>
```

Next, you will need to install the LangSmith SDK:

```
pip install -U langsmith
pip install -U instructor
```

You can find this example in our [examples directory](../../examples/bulk_classification.md):

```bash
# The example code is available in the examples directory
# See: https://python.useinstructor.com/examples/bulk_classification
```

In this example we'll use the `wrap_openai` function to wrap the OpenAI client with LangSmith. This will allow us to use LangSmith's observability and monitoring features with the OpenAI client. Then we'll use `instructor` to patch the client with the `TOOLS` mode. This will allow us to use `instructor` to add additional functionality to the client. We'll use [asyncio](./learn-async.md) to classify a list of questions.

```python
import instructor
import asyncio

from langsmith import traceable
from langsmith.wrappers import wrap_openai

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator
from typing import List
from enum import Enum

# Wrap the OpenAI client with LangSmith
wrapped_client = wrap_openai(AsyncOpenAI())

# Create instructor client with LangSmith-wrapped client
# Note: When using LangSmith, you may need to pass the wrapped client
# For most cases, use: client = instructor.from_provider("openai/gpt-4o", mode=instructor.Mode.TOOLS)
client = instructor.from_provider("openai/gpt-4o", mode=instructor.Mode.TOOLS)

# Rate limit the number of requests
sem = asyncio.Semaphore(5)


# Use an Enum to define the types of questions
class QuestionType(Enum):
    CONTACT = "CONTACT"
    TIMELINE_QUERY = "TIMELINE_QUERY"
    DOCUMENT_SEARCH = "DOCUMENT_SEARCH"
    COMPARE_CONTRAST = "COMPARE_CONTRAST"
    EMAIL = "EMAIL"
    PHOTOS = "PHOTOS"
    SUMMARY = "SUMMARY"


# You can add more instructions and examples in the description
# or you can put it in the prompt in `messages=[...]`
class QuestionClassification(BaseModel):
    """
    Predict the type of question that is being asked.
    Here are some tips on how to predict the question type:
    CONTACT: Searches for some contact information.
    TIMELINE_QUERY: "When did something happen?
    DOCUMENT_SEARCH: "Find me a document"
    COMPARE_CONTRAST: "Compare and contrast two things"
    EMAIL: "Find me an email, search for an email"
    PHOTOS: "Find me a photo, search for a photo"
    SUMMARY: "Summarize a large amount of data"
    """

    # If you want only one classification, just change it to
    #   `classification: QuestionType` rather than `classifications: List[QuestionType]``
    chain_of_thought: str = Field(
        ..., description="The chain of thought that led to the classification"
    )
    classification: List[QuestionType] = Field(
        description=f"An accuracy and correct prediction predicted class of question. Only allowed types: {[t.value for t in QuestionType]}, should be used",
    )

    @field_validator("classification", mode="before")
    def validate_classification(cls, v):
        # sometimes the API returns a single value, just make sure it's a list
        if not isinstance(v, list):
            v = [v]
        return v


@traceable(name="classify-question")
async def classify(data: str) -> QuestionClassification:
    """
    Perform multi-label classification on the input text.
    Change the prompt to fit your use case.

    Args:
        data (str): The input text to classify.
    """
    async with sem:  # some simple rate limiting
        return data, await client.create(
            model="gpt-5.4-mini",
            response_model=QuestionClassification,
            max_retries=2,
            messages=[
                {
                    "role": "user",
                    "content": f"Classify the following question: {data}",
                },
            ],
        )


async def main(questions: List[str]):
    tasks = [classify(question) for question in questions]

    for task in asyncio.as_completed(tasks):
        question, label = await task
        resp = {
            "question": question,
            "classification": [c.value for c in label.classification],
            "chain_of_thought": label.chain_of_thought,
        }
        resps.append(resp)
    return resps


if __name__ == "__main__":
    import asyncio

    questions = [
        "What was that ai app that i saw on the news the other day?",
        "Can you find the trainline booking email?",
        "what did I do on Monday?",
        "Tell me about todays meeting and how it relates to the email on Monday",
    ]

    resp = asyncio.run(main(questions))

    for r in resp:
        print("q:", r["question"])
        #> q: what did I do on Monday?
        print("c:", r["classification"])
        #> c: ['SUMMARY']
```

If you follow what we've done is wrapped the client and proceeded to quickly use asyncio to classify a list of questions. This is a simple example of how you can use LangSmith to enhance the OpenAI client. You can use LangSmith to monitor and observe the client, and use `instructor` to add additional functionality to the client.

To take a look at trace of this run check out this shareable [link](https://smith.langchain.com/public/eaae9f95-3779-4bbb-824d-97aa8a57a4e0/r).

![](./img/langsmith.png)
