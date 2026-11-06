---
authors:
- jxnl
categories:
- Pydantic
comments: true
date: 2024-03-08
description: Learn to generate synthetic data using Pydantic and OpenAI's models with
  practical examples and configurations.
draft: false
tags:
- Synthetic Data
- Pydantic
- OpenAI
- Data Generation
- Python
---

# Simple Synthetic Data Generation

What that people have been using instructor for is to generate synthetic data rather than extracting data itself. We can even use the J-Schemo extra fields to give specific examples to control how we generate data.

Consider the example below. We'll likely generate very simple names.

```python
from typing import Iterable
from pydantic import BaseModel
import instructor


# Define the UserDetail model
class UserDetail(BaseModel):
    name: str
    age: int


# Patch the OpenAI client to enable the response_model functionality
client = instructor.from_provider("openai/gpt-5-nano")


def generate_fake_users(count: int) -> Iterable[UserDetail]:
    return client.create(
        model="gpt-5.4-mini",
        response_model=Iterable[UserDetail],
        messages=[
            {"role": "user", "content": f"Generate a {count} synthetic users"},
        ],
    )


for user in generate_fake_users(5):
    print(user)
    #> name='Alice' age=25
    #> name='Bob' age=30
    #> name='Charlie' age=22
    #> name='David' age=28
    #> name='Eve' age=35
```

## Leveraging Simple Examples

We might want to set examples as part of the prompt by leveraging Pydantics configuration. We can set examples directly in the JSON scheme itself.

```python
from typing import Iterable
from pydantic import BaseModel, Field
import instructor


# Define the UserDetail model
class UserDetail(BaseModel):
    name: str = Field(examples=["Timothee Chalamet", "Zendaya"])
    age: int


# Patch the OpenAI client to enable the response_model functionality
client = instructor.from_provider("openai/gpt-5-nano")


def generate_fake_users(count: int) -> Iterable[UserDetail]:
    return client.create(
        model="gpt-5.4-mini",
        response_model=Iterable[UserDetail],
        messages=[
            {"role": "user", "content": f"Generate a {count} synthetic users"},
        ],
    )


for user in generate_fake_users(5):
    print(user)
    #> name='John Doe' age=25
    #> name='Alice Smith' age=30
    #> name='Bob Johnson' age=28
    #> name='Emily Brown' age=35
    #> name='Michael Williams' age=27
```

By incorporating names of celebrities as examples, we have shifted towards generating synthetic data featuring well-known personalities, moving away from the simplistic, single-word names previously used.

## Leveraging Complex Example

To effectively generate synthetic examples with more nuance, lets upgrade to the "gpt-5.4-mini" model, use model level examples rather than attribute level examples:

```Python
import instructor

from typing import Iterable
from pydantic import BaseModel, ConfigDict


# Define the UserDetail model
class UserDetail(BaseModel):
    """Old Wizards"""

    name: str
    age: int

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"name": "Gandalf the Grey", "age": 1000},
                {"name": "Albus Dumbledore", "age": 150},
            ]
        }
    )


# Patch the OpenAI client to enable the response_model functionality
client = instructor.from_provider("openai/gpt-5-nano")


def generate_fake_users(count: int) -> Iterable[UserDetail]:
    return client.create(
        model="gpt-5.4-mini",
        response_model=Iterable[UserDetail],
        messages=[
            {"role": "user", "content": f"Generate `{count}` synthetic examples"},
        ],
    )


for user in generate_fake_users(5):
    print(user)
    #> name='Merlin' age=600
    #> name='Radagast the Brown' age=950
    #> name='Rincewind' age=70
    #> name='Harry Potter' age=17
    #> name='Elminster Aumar' age=1200
```

## Leveraging Descriptions

By adjusting the descriptions within our Pydantic models, we can subtly influence the nature of the synthetic data generated. This method allows for a more nuanced control over the output, ensuring that the generated data aligns more closely with our expectations or requirements.

For instance, specifying "Fancy French sounding names" as a description for the `name` field in our `UserDetail` model directs the generation process to produce names that fit this particular criterion, resulting in a dataset that is both diverse and tailored to specific linguistic characteristics.


```python
import instructor

from typing import Iterable
from pydantic import BaseModel, Field


# Define the UserDetail model
class UserDetail(BaseModel):
    name: str = Field(description="Fancy French sounding names")
    age: int


# Patch the OpenAI client to enable the response_model functionality
client = instructor.from_provider("openai/gpt-5-nano")


def generate_fake_users(count: int) -> Iterable[UserDetail]:
    return client.create(
        model="gpt-5.4-mini",
        response_model=Iterable[UserDetail],
        messages=[
            {"role": "user", "content": f"Generate `{count}` synthetic users"},
        ],
    )


for user in generate_fake_users(5):
    print(user)
    #> name='Jean Luc' age=25
    #> name='Marcelle' age=30
    #> name='Antoinette' age=22
    #> name='Gaspard' age=28
    #> name='Eloise' age=35
```