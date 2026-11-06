---
authors:
  - jxnl
categories:
  - LLM Techniques
comments: true
date: 2024-06-06
description:
  Explore enhancing RAG systems with time filters using Instructor and
  Pydantic for accurate, relevant data retrieval.
draft: false
tags:
  - RAG
  - Time Filters
  - Pydantic
  - Instructor
  - LLM Techniques
---

# Enhancing RAG with Time Filters Using Instructor

Retrieval-augmented generation (RAG) systems often need to handle queries with time-based constraints, like "What new features were released last quarter?" or "Show me support tickets from the past week." Effective time filtering is crucial for providing accurate, relevant responses.

Instructor is a Python library that simplifies integrating large language models (LLMs) with data sources and APIs. It allows defining structured output models using Pydantic, which can be used as prompts or to parse LLM outputs.

<!-- more -->

## Modeling Time Filters

To handle time filters, we can define a Pydantic model representing a time range:

```python
from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class TimeFilter(BaseModel):
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
```

The `TimeFilter` model can represent an absolute date range or a relative time range like "last week" or "previous month."

We can then combine this with a search query string:

```python
class SearchQuery(BaseModel):
    query: str
    time_filter: TimeFilter
```

## Prompting the LLM

Using Instructor, we can prompt the LLM to generate a `SearchQuery` object based on the user's query:

```python
import instructor

client = instructor.from_provider("openai/gpt-5-nano")

response = client.create(
    model="gpt-4o",
    response_model=SearchQuery,
    messages=[
        {
            "role": "system",
            "content": "You are a query generator for customer support tickets. The current date is 2024-02-17",
        },
        {
            "role": "user",
            "content": "Show me customer support tickets opened in the past week.",
        },
    ],
)

# Example response:
{
    "query": "Show me customer support tickets opened in the past week.",
    "time_filter": {
        "start_date": "2024-02-10T00:00:00",
        "end_date": "2024-02-17T00:00:00",
    },
}
```

## Nuances in dates and timezones

When working with time-based queries, it's important to consider the nuances of dates, timezones, and publication times. Depending on the data source, the user's location, and when the content was originally published, the definition of "past week" or "last month" may vary.

To handle this, you'll want to design your `TimeFilter` model to intelligently reason about these relative time periods. This could involve:

- Defaulting to the user's local timezone if available, or using a consistent default like UTC
- Defining clear rules for how to calculate the start and end of relative periods like "week" or "month"
  - e.g. does "past week" mean the last 7 days or the previous Sunday-Saturday range?
- Allowing for flexibility in how users specify dates (exact datetimes, just dates, natural language phrases)
- Validating and normalizing user input to fit the expected `TimeFilter` format
- Considering the original publication timestamp of the content, not just the current date
  - e.g. "articles published in the last month" should look at the publish date, not the query date

By building this logic into the `TimeFilter` model, you can abstract away the complexity and provide a consistent interface for the rest of your RAG system to work with standardized absolute datetime ranges

Of course, there may be edge cases or ambiguities that are hard to resolve programmatically. In these situations, you may need to prompt the user for clarification or make a best guess based on the available information. The key is to strive for a balance of flexibility and consistency in how you handle time-based queries, factoring in publication dates when relevant.

By modeling time filters with Pydantic and leveraging Instructor, RAG systems can effectively handle time-based queries. Clear prompts, careful model design, and appropriate parsing strategies enable accurate retrieval of information within specific time frames, enhancing the system's overall relevance and accuracy.
