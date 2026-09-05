---
authors:
- ivanleomk
categories:
- OpenAI
comments: true
date: 2024-08-20
description: Explore the challenges of OpenAI's Structured Outputs and how 'instructor'
  offers solutions for LLM workflows.
draft: false
slug: should-i-be-using-structured-outputs
tags:
- OpenAI
- Structured Outputs
- Pydantic
- Data Validation
- LLM Techniques
---

# Should I Be Using Structured Outputs?

OpenAI recently announced Structured Outputs which ensures that generated responses match any arbitrary provided JSON Schema. In their [announcement article](https://openai.com/index/introducing-structured-outputs-in-the-api/), they acknowledged that it had been inspired by libraries such as `instructor`.

## Main Challenges

If you're building complex LLM workflows, you've likely considered OpenAI's Structured Outputs as a potential replacement for `instructor`.

But before you do so, three key challenges remain:

1. **Limited Validation And Retry Logic**: Structured Outputs ensure adherence to the schema but not useful content. You might get perfectly formatted yet unhelpful responses
2. **Streaming Challenges**: Parsing raw JSON objects from streamed responses with the sdk is error-prone and inefficient
3. **Unpredictable Latency Issues** : Structured Outputs suffers from random latency spikes that might result in an almost 20x increase in response time

Additionally, adopting Structured Outputs locks you into OpenAI's ecosystem, limiting your ability to experiment with diverse models or providers that might better suit specific use-cases.

This vendor lock-in increases vulnerability to provider outages, potentially causing application downtime and SLA violations, which can damage user trust and impact your business reputation.

In this article, we'll show how `instructor` addresses many of these challenges with features such as automatic reasking when validation fails, automatic support for validated streaming data and more.

<!-- more -->

### Limited Validation and Retry Logic

Validation is crucial for building reliable and effective applications. We want to catch errors in real time using `Pydantic` [validators](../../concepts/reask_validation.md) in order to allow our LLM to correct its responses on the fly.

Let's see an example of a simple validator below which ensures user names are always in uppercase.

```python
import openai
from pydantic import BaseModel, field_validator


class User(BaseModel):
    name: str
    age: int

    @field_validator("name")
    def ensure_uppercase(cls, v: str) -> str:
        if not v.isupper():
            raise ValueError("All letters must be uppercase. Got: " + v)
        return v


client = openai.OpenAI()
try:
    resp = client.beta.chat.completions.parse(
        response_format=User,
        messages=[
            {
                "role": "user",
                "content": "Extract the following user: Jason is 25 years old.",
            },
        ],
        model="gpt-4o-mini",
    )
except Exception as e:
    print(e)
    """
    1 validation error for User
    name
      Value error, All letters must be uppercase. Got: Jason [type=value_error, input_value='Jason', input_type=str]
        For further information visit https://errors.pydantic.dev/2.11/v/value_error
    """
```

We can see that we lose the original completion when validation fails. This leaves developers without the means to implement retry logic so that the LLM can provide a targeted correction and regenerate its response.

Without robust validation, applications risk producing inconsistent outputs and losing valuable context for error correction. This leads to degraded user experience and missed opportunities for targeted improvements in LLM responses.

### Streaming Challenges

Streaming with Structured Outputs is complex. It requires manual parsing, lacks partial validation, and needs a context manager to be used with. Effective implementation with the `beta.chat.completions.stream` method demands significant effort.

Let's see an example below.

```python
import openai
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = openai.OpenAI()
with client.beta.chat.completions.stream(
    response_format=User,
    messages=[
        {
            "role": "user",
            "content": "Extract the following user: Jason is 25 years old.",
        },
    ],
    model="gpt-4o-mini",
) as stream:
    for event in stream:
        if event.type == "content.delta":
            print(event.snapshot, flush=True, end="\n")
            # >
            #> {"
            #> {"name
            #> {"name":"
            #> {"name":"Jason
            #> {"name":"Jason","
            #> {"name":"Jason","age
            #> {"name":"Jason","age":
            #> {"name":"Jason","age":25
            #> {"name":"Jason","age":25}
            # >
            #> {"
            #> {"name
            #> {"name":"
            #> {"name":"Jason
            #> {"name":"Jason","
            #> {"name":"Jason","age
            #> {"name":"Jason","age":
            #> {"name":"Jason","age":25
            #> {"name":"Jason","age":25}
            # >
            #> {"
            #> {"name
            #> {"name":"
            #> {"name":"Jason
            #> {"name":"Jason","
            #> {"name":"Jason","age
            #> {"name":"Jason","age":
            #> {"name":"Jason","age":25
            #> {"name":"Jason","age":25}
            # >
            #> {"
            #> {"name
            #> {"name":"
            #> {"name":"Jason
            #> {"name":"Jason","
            #> {"name":"Jason","age
            #> {"name":"Jason","age":
            #> {"name":"Jason","age":25
            #> {"name":"Jason","age":25}
```

### Unpredictable Latency Spikes

In order to benchmark the two modes, we made 200 identical requests to OpenAI and noted the time taken for each request to complete. The results are summarized in the following table:

| mode               | mean  | min   | max    | std_dev | variance |
| ------------------ | ----- | ----- | ------ | ------- | -------- |
| Tool Calling       | 6.84  | 6.21  | 12.84  | 0.69    | 0.47     |
| Structured Outputs | 28.20 | 14.91 | 136.90 | 9.27    | 86.01    |

Structured Outputs suffers from unpredictable latency spikes while Tool Calling maintains consistent performance. This could cause users to occasionally experience significant delays in response times, potentially impacting the overall user satisfication and retention rates.

## Why use `instructor`

`instructor` is fully compatible with Structured Outputs and provides three main benefits to developers.

1. **Automatic Validation and Retries**: Regenerates LLM responses on Pydantic validation failures, ensuring data integrity.
2. **Real-time Streaming Validation**: Incrementally validates partial JSON against Pydantic models, enabling immediate use of validated properties.
3. **Provider-Agnostic API**: Switch between LLM providers and models with a single line of code.

Let's see this in action below

### Automatic Validation and Retries

With `instructor`, all it takes is a simple Pydantic Schema and a validator for you to get the extracted names as an upper case value.

```python
import instructor
from pydantic import BaseModel, field_validator


class User(BaseModel):
    name: str
    age: int

    @field_validator("name")
    def ensure_uppercase(cls, v: str) -> str:
        if not v.isupper():
            raise ValueError("All letters must be uppercase. Got: " + v)
        return v


client = instructor.from_provider(
    "openai/gpt-5-nano", mode=instructor.Mode.TOOLS_STRICT
)

resp = client.create(
    response_model=User,
    messages=[
        {
            "role": "user",
            "content": "Extract the following user: Jason is 25 years old.",
        }
    ],
    model="gpt-4o-mini",
)

print(resp)
#> name='JASON' age=25
```

This built-in retry logic allows for targeted correction to the generated response, ensuring that outputs are not only consistent with your schema but also correct for your use-case. This is invaluable in building reliable LLM systems.

### Real-time Streaming Validation

A common use-case is to define a single schema and extract multiple instances of it. With `instructor`, doing this is relatively straightforward by using [our `create_iterable` method](../../concepts/lists.md).

```python
client = instructor.from_provider(
    "openai/gpt-5-nano", mode=instructor.Mode.TOOLS_STRICT
)


class User(BaseModel):
    name: str
    age: int


users = client.create_iterable(
    model="gpt-4o-mini",
    response_model=User,
    messages=[
        {
            "role": "system",
            "content": "You are a perfect entity extraction system",
        },
        {
            "role": "user",
            "content": (f"Extract `Jason is 10 and John is 10`"),
        },
    ],
)

for user in users:
    print(user)
    #> name='Jason' age=10
    #> name='John' age=10
```

Other times, we might also want to stream out information as it's dynamically generated into some sort of frontend component With `instructor`, you'll be able to do just that [using the `create_partial` method](../../concepts/partial.md).

```python
import instructor
from pydantic import BaseModel
from rich.console import Console

client = instructor.from_provider(
    "openai/gpt-5-nano", mode=instructor.Mode.TOOLS_STRICT
)

text_block = """
In our recent online meeting, participants from various backgrounds joined to discuss the upcoming tech conference. The names and contact details of the participants were as follows:

- Name: John Doe, Email: johndoe@email.com, Twitter: @TechGuru44
- Name: Jane Smith, Email: janesmith@email.com, Twitter: @DigitalDiva88
- Name: Alex Johnson, Email: alexj@email.com, Twitter: @CodeMaster2023

During the meeting, we agreed on several key points. The conference will be held on March 15th, 2024, at the Grand Tech Arena located at 4521 Innovation Drive. Dr. Emily Johnson, a renowned AI researcher, will be our keynote speaker.

The budget for the event is set at $50,000, covering venue costs, speaker fees, and promotional activities. Each participant is expected to contribute an article to the conference blog by February 20th.

A follow-up meeting is scheduled for January 25th at 3 PM GMT to finalize the agenda and confirm the list of speakers.
"""


class User(BaseModel):
    name: str
    email: str
    twitter: str


class MeetingInfo(BaseModel):
    users: list[User]
    date: str
    location: str
    budget: int
    deadline: str


extraction_stream = client.create_partial(
    model="gpt-4o-mini",
    response_model=MeetingInfo,
    messages=[
        {
            "role": "user",
            "content": f"Get the information about the meeting and the users {text_block}",
        },
    ],
    stream=True,
)


console = Console()

for extraction in extraction_stream:
    obj = extraction.model_dump()
    console.clear()
    console.print(obj)
```

This will output the following

![Structured Output Extraction](./img/Structured_Output_Extraction.gif)

### Provider-Agnostic API

With `instructor`, switching between different providers is easy due to our unified API.

For example, the switch from OpenAI to Anthropic requires only three adjustments

1. Import the Anthropic client
2. Use `from_anthropic` instead of `from_openai`
3. Update the model name (e.g., from gpt-4o-mini to claude-3-5-sonnet)

This makes it incredibly flexible for users looking to migrate and test different providers for their use cases. Let's see this in action with an example below.

```python
import instructor
from pydantic import BaseModel

client = instructor.from_provider("openai/gpt-5-nano")


class User(BaseModel):
    name: str
    age: int


resp = client.create(
    model="gpt-4o-mini",
    response_model=User,
    messages=[
        {
            "role": "user",
            "content": "Extract the user from the string belo - Chris is a 27 year old engineer in San Francisco",
        }
    ],
    max_tokens=100,
)

print(resp)
#> name='Chris' age=27
```

Now let's see how we can achieve the same with Anthropic.

```python hl_lines="2 5 14"
import instructor
from pydantic import BaseModel

client = instructor.from_provider("anthropic/claude-3-5-haiku-latest")  # (2)!


class User(BaseModel):
    name: str
    age: int


resp = client.create(
    model="claude-3-5-sonnet-20240620",  # (3)!
    response_model=User,
    messages=[
        {
            "role": "user",
            "content": "Extract the user from the string belo - Chris is a 27 year old engineer in San Francisco",
        }
    ],
    max_tokens=100,
)

print(resp)
#> name='Chris' age=27
```

1.  Import the Anthropic client
2.  Use `from_anthropic` instead of `from_openai`
3.  Update the model name to `claude-3-5-sonnet-20240620`

## Conclusion

While OpenAI's Structured Outputs shows promise, it has key limitations. The system lacks support for extra JSON fields to provide output examples, default value factories, and pattern matching in defined schemas. These constraints limit developers' ability to express complex return types, potentially impacting application performance and flexibility.

If you're interested in Structured Outputs, `instructor` addresses these critical issues. It provides automatic retries, real-time input validation, and multi-provider integration, allowing developers to more effectively implement Structured Outputs in their AI projects.

if you haven't given `instructor` a shot, try it today!
