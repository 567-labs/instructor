---
title: Search Query Segmentation with Instructor - Multi-Task Extraction
description: Segment complex search queries into actionable tasks using Instructor. Break down user queries into parallel executable tasks with structured outputs.
---

# Example: Segmenting Search Queries

In this example, we will demonstrate how to leverage the `MultiTask` and `enum.Enum` features of OpenAI Function Call to segment search queries. We will define the necessary structures using Pydantic and demonstrate how segment queries into multiple sub queries and execute them in parallel with `asyncio`.

!!! tips "Motivation"

    Extracting a list of tasks from text is a common use case for leveraging language models. This pattern can be applied to various applications, such as virtual assistants like Siri or Alexa, where understanding user intent and breaking down requests into actionable tasks is crucial. In this example, we will demonstrate how to use OpenAI Function Call to segment search queries and execute them in parallel.

## Structure of the Data

The `Search` class is a Pydantic model that defines the structure of the search query. It has three fields: `title`, `query`, and `type`. The `title` field is the title of the request, the `query` field is the query to search for relevant content, and the `type` field is the type of search. The `execute` method is used to execute the search query.

```python
import instructor
from typing import Iterable, Literal
from pydantic import BaseModel, Field

# Apply the patch to the OpenAI client
# enables response_model keyword
client = instructor.from_provider("openai/gpt-5-nano")


class Search(BaseModel):
    query: str = Field(..., description="Query to search for relevant content")
    type: Literal["web", "image", "video"] = Field(..., description="Type of search")

    async def execute(self):
        print(
            f"Searching for `{self.title}` with query `{self.query}` using `{self.type}`"
        )


def segment(data: str) -> Search:
    return client.create(
        model="gpt-4o-mini",
        response_model=Iterable[Search],
        messages=[
            {
                "role": "user",
                "content": f"Consider the data below: '\n{data}' and segment it into multiple search queries",
            },
        ],
        max_tokens=1000,
    )


for search in segment("Search for a picture of a cat and a video of a dog"):
    print(search.model_dump_json())
    #> {"query":"picture of a cat","type":"image"}
    #> {"query":"video of a dog","type":"video"}
```
