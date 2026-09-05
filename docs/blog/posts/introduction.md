---
authors:
- jxnl
categories:
- Pydantic
comments: true
date: 2023-09-11
description: Learn how Pydantic simplifies working with LLMs and structured JSON outputs
  in Python, enhancing developer experience and code organization.
draft: false
tags:
- Pydantic
- LLMs
- Python
- OpenAI
- JSON
---

# Generating Structured Output / JSON from LLMs

Language models have seen significant growth. Using them effectively often requires complex frameworks. This post discusses how Instructor simplifies this process using Pydantic.

<!-- more -->

## The Problem with Existing LLM Frameworks

Current frameworks for Language Learning Models (LLMs) have complex setups. Developers find it hard to control interactions with language models. Some frameworks require complex JSON Schema setups.

## The OpenAI Function Calling Game-Changer

OpenAI's Function Calling feature provides a constrained interaction model. However, it has its own complexities, mostly around JSON Schema.

## Why Pydantic?

Instructor uses Pydantic to simplify the interaction between the programmer and the language model.

- **Widespread Adoption**: Pydantic is a popular tool among Python developers.
- **Simplicity**: Pydantic allows model definition in Python.
- **Framework Compatibility**: Many Python frameworks already use Pydantic.

```python
import pydantic
import instructor

# Enables the response_model
client = instructor.from_provider("openai/gpt-5-nano")


class UserDetail(pydantic.BaseModel):
    name: str
    age: int

    def introduce(self):
        return f"Hello I'm {self.name} and I'm {self.age} years old"


user: UserDetail = client.create(
    model="gpt-5.4-mini",
    response_model=UserDetail,
    messages=[
        {"role": "user", "content": "Extract Jason is 25 years old"},
    ],
)
```

## Simplifying Validation Flow with Pydantic

Pydantic validators simplify features like re-asking or self-critique. This makes these tasks less complex compared to other frameworks.

```python
from typing_extensions import Annotated
from pydantic import BaseModel, BeforeValidator
from instructor import llm_validator


class QuestionAnswerNoEvil(BaseModel):
    question: str
    answer: Annotated[
        str,
        BeforeValidator(llm_validator("don't say objectionable things")),
    ]
```

## The Modular Approach

Pydantic allows for modular output schemas. This leads to more organized code.

### Composition of Schemas

```python
class UserDetails(BaseModel):
    name: str
    age: int


class UserWithAddress(UserDetails):
    address: str
```

### Defining Relationships

```python
class UserDetail(BaseModel):
    id: int
    age: int
    name: str
    friends: List[int]


class UserRelationships(BaseModel):
    users: List[UserDetail]
```

### Using Enums

```python
from enum import Enum, auto


class Role(Enum):
    PRINCIPAL = auto()
    TEACHER = auto()
    STUDENT = auto()
    OTHER = auto()


class UserDetail(BaseModel):
    age: int
    name: str
    role: Role
```

### Flexible Schemas

```python
from typing import List


class Property(BaseModel):
    key: str
    value: str


class UserDetail(BaseModel):
    age: int
    name: str
    properties: List[Property]
```

### Chain of Thought

```python
class TimeRange(BaseModel):
    chain_of_thought: str
    start_time: int
    end_time: int


class UserDetail(BaseModel):
    id: int
    age: int
    name: str
    work_time: TimeRange
    leisure_time: TimeRange
```

## Language Models as Microservices

The architecture resembles FastAPI. Most code can be written as Python functions that use Pydantic objects. This eliminates the need for prompt chains.

### FastAPI Stub

```python
import fastapi
from pydantic import BaseModel


class UserDetails(BaseModel):
    name: str
    age: int


app = fastapi.FastAPI()


@app.get("/user/{user_id}", response_model=UserDetails)
async def get_user(user_id: int) -> UserDetails:
    raise fastapi.HTTPException(status_code=404, detail=f"User {user_id} not found")
```

### Using Instructor as a Function

```python
def extract_user(text: str) -> UserDetails:
    return client.chat.completions.create(
        model="gpt-4.1-mini",
        response_model=UserDetails,
        messages=[{"role": "user", "content": text}],
    )
```

### Response Modeling

```python
class MaybeUser(BaseModel):
    result: Optional[UserDetail]
    error: bool
    message: Optional[str]
```

## Conclusion

Instructor, with Pydantic, simplifies interaction with language models. It is usable for both experienced and new developers.

## Related Concepts

- [Getting Started Guide](../../index.md) - Learn how to install and use Instructor
- [Model Providers](../../integrations/index.md) - Explore supported LLM providers
- [Validation Context](../../concepts/reask_validation.md) - Understand how to validate LLM outputs
- [Response Models](../../concepts/models.md) - Deep dive into defining structured outputs

## See Also

- [Why Instructor is the Best Library](best_framework.md) - Learn about Instructor's philosophy and advantages
- [Structured Outputs and Prompt Caching with Anthropic](structured-output-anthropic.md) - See how Instructor works with Claude
- [Chain of Density Tutorial](../../tutorials/6-chain-of-density.ipynb) - Learn advanced prompting techniques

If you enjoy the content or want to try out `instructor` please check out the [github](https://github.com/jxnl/instructor) and give us a star!
