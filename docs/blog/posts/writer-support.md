---
authors:
  - ivanleomk
  - yanomaly
categories:
  - Writer SDK
comments: true
date: 2024-11-19
description: Announcing Writer integration with Instructor for structured outputs and enterprise AI workflows
draft: false
slug: writer-support
tags:
  - Writer
  - Enterprise AI
  - Integrations
---

# Structured Outputs with Writer now supported

>

We're excited to announce that `instructor` now supports [Writer](https://writer.com)'s enterprise-grade LLMs, including their latest Palmyra X 004 model. This integration enables structured outputs and enterprise AI workflows with Writer's powerful language models.

## Getting Started

First, make sure that you've signed up for an account on [Writer](https://app.writer.com/aistudio/signup?utm_campaign=devrel) and obtained an API key using this [quickstart guide](https://dev.writer.com/api-guides/quickstart). Once you've done so, install `instructor` with Writer support by running `pip install instructor[writer]` in your terminal.

Make sure to set the `WRITER_API_KEY` environment variable with your Writer API key or pass it as an argument to the `Writer` constructor.

<!-- more -->

```python
import instructor
from pydantic import BaseModel

# Initialize Writer client
client = instructor.from_provider("writer/claude-3-5-sonnet-20241022")


class User(BaseModel):
    name: str
    age: int


# Extract structured data
user = client.create(
    model="palmyra-x-004",
    messages=[{"role": "user", "content": "Extract: John is 30 years old"}],
    response_model=User,
)

print(user)
#> name='John' age=30
```

!!! note

    If you'd like to use the Async version of the Writer client, you can do so by using `instructor.from_provider("writer/claude-3-5-sonnet-20241022")`.

We also support streaming with the Writer client using our `create_partial` method. This allows you to process responses incrementally as they arrive.

This is particularly valuable for maintaining responsive applications and delivering a smooth user experience, especially when dealing with larger responses so that users can see immediate results.

```python
import instructor
from pydantic import BaseModel

# Initialize Writer client
client = instructor.from_provider("writer/claude-3-5-sonnet-20241022")


text_block = """
In our recent online meeting, participants from various backgrounds joined to discuss the upcoming tech conference. The names and contact details of the participants were as follows:

- Name: John Doe, Email: johndoe@email.com, Twitter: @TechGuru44
- Name: Jane Smith, Email: janesmith@email.com, Twitter: @DigitalDiva88
- Name: Alex Johnson, Email: alexj@email.com, Twitter: @CodeMaster2023

During the meeting, we agreed on several key points. The conference will be held on March 15th, 2024, at the Grand Tech Arena located at 4521 Innovation Drive. Dr. Emily Johnson, a renowned AI researcher, will be our keynote speaker.

The budget for the event is set at $50,000, covering venue costs, speaker fees, and promotional activities. Each participant is expected to contribute an article to the conference blog by February 20th.

A follow-up meetingis scheduled for January 25th at 3 PM GMT to finalize the agenda and confirm the list of speakers.
"""


class User(BaseModel):
    name: str
    email: str
    twitter: str


class MeetingInfo(BaseModel):
    date: str
    location: str
    budget: int
    deadline: str


PartialMeetingInfo = instructor.Partial[MeetingInfo]


extraction_stream = client.create(
    model="palmyra-x-004",
    messages=[
        {
            "role": "user",
            "content": f"Get the information about the meeting and the users {text_block}",
        },
    ],
    response_model=PartialMeetingInfo,
    stream=True,
)  # type: ignore


for obj in extraction_stream:
    print(obj)
    #> date='March 15th, 2024' location='' budget=None deadline=None
    #> date='March 15th, 2024' location='Grand Tech Arena, 4521 Innovation' budget=None deadline=None
    #> date='March 15th, 2024' location='Grand Tech Arena, 4521 Innovation Drive' budget=50000 eadline='February 20th'
```

As with all our integrations, `instructor` ships with the ability to automatically retry requests that happen due to schema validation without you having to do anything.

```python
import instructor
from typing import Annotated
from pydantic import BaseModel, AfterValidator, Field

# Initialize Writer client
client = instructor.from_provider("writer/claude-3-5-sonnet-20241022")


# Example of model, that may require usage of retries
def uppercase_validator(v):
    if v.islower():
        raise ValueError("Name must be in uppercase")
    return v


class User(BaseModel):
    name: Annotated[str, AfterValidator(uppercase_validator)] = Field(
        ..., description="The name of the user"
    )
    age: int


user = client.create(
    model="palmyra-x-004",
    messages=[{"role": "user", "content": "Extract: jason is 12"}],
    response_model=User,
    max_retries=3,
)

print(user)
#> name='JASON' age=12
```

This was a sneak peek into the things that you can do with Writer and `instructor` - from classification of text to sentimen analysis and more.

We're excited to see what you build with `instructor` and Writer. If you have any other questions about writer, do check out the [Writer Documentation](https://dev.writer.com/home/introduction) for the API sdk.
