---
title: Philosophy
description: The principles behind Instructor - why simple beats complex every time.
---

# Philosophy

Great tools make hard things easy without making easy things hard. That's Instructor.

## Start with what developers know

Most AI frameworks invent their own abstractions. We don't.

```python
import instructor
from pydantic import BaseModel


# What you already know (Pydantic)
class User(BaseModel):
    name: str
    age: int


# What Instructor adds
client = instructor.from_provider("openai/gpt-4.1-mini")
_user = client.create(
    response_model=User,
    messages=[{"role": "user", "content": "Jane is 33"}],
)  # That's it
```

If you know Pydantic, you know Instructor. No new concepts, no new syntax, no 200-page manual.

## Your escape hatch is always there

The worst frameworks are roach motels - easy to get in, impossible to get out. Instructor is different:

```python
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


# With Instructor
client = instructor.from_provider("openai/gpt-4.1-mini")
_result = client.create(
    response_model=User,
    messages=[{"role": "user", "content": "Jane is 33"}],
)

# Want to go back to raw API? Just remove response_model:
client = instructor.from_provider("openai/gpt-4.1-mini")
_result = client.create(messages=[{"role": "user", "content": "Say hello"}])

# Or use the provider directly:
from openai import OpenAI

_raw_client = OpenAI()  # Back to vanilla
```

We patch, we don't wrap. Your code, your control.

## Show, don't hide

Bad frameworks hide complexity. Good tools help you understand it.

```python
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


# See exactly what Instructor sends
instructor.logfire.configure()  # Full observability

client = instructor.from_provider("openai/gpt-4.1-mini")
result = client.create(
    response_model=User,
    messages=[{"role": "user", "content": "Jane is 33"}],
)

# Access raw responses
_raw_response = result._raw_response  # See what the LLM actually returned
```

When something goes wrong (and it will), you can see exactly what happened.

## Composition beats configuration

No YAML files. No decorators. No magic. Just functions.

```python
import instructor
from pydantic import BaseModel

client = instructor.from_provider("openai/gpt-4.1-mini")


class User(BaseModel):
    name: str
    age: int


class Company(BaseModel):
    name: str
    industry: str


class Analysis(BaseModel):
    user: User
    company: Company


# Build complex systems with simple functions
def extract_user(text: str) -> User:
    return client.create(
        response_model=User, messages=[{"role": "user", "content": text}]
    )


def extract_company(text: str) -> Company:
    return client.create(
        response_model=Company, messages=[{"role": "user", "content": text}]
    )


def analyze_email(email: str) -> Analysis:
    user = extract_user(email)
    company = extract_company(email)
    return Analysis(user=user, company=company)


# Compose however makes sense for YOUR application
_analysis = analyze_email("Please introduce Jane from Acme.")
```

## Start simple, grow naturally

The best code is code that grows with your needs:

```python
import instructor
from instructor import Partial
from pydantic import BaseModel, field_validator

client = instructor.from_provider("openai/gpt-4.1-mini")


class User(BaseModel):
    name: str
    age: int


# Day 1: Just get it working
_user = client.create(
    response_model=User,
    messages=[{"role": "user", "content": "Jane is 33"}],
)


# Day 7: Add validation
class User(BaseModel):
    name: str
    age: int

    @field_validator("age")
    def check_age(cls, value: int) -> int:
        if value < 0 or value > 150:
            raise ValueError("Invalid age")
        return value


# Day 14: Add retries for production
_user = client.create(
    response_model=User,
    messages=[{"role": "user", "content": "Jane is 33"}],
    max_retries=3,
)


# Day 30: Add streaming for better UX
def update_ui(_partial: Partial[User]) -> None:
    pass


for partial in client.create(
    response_model=Partial[User],
    messages=[{"role": "user", "content": "Jane is 33"}],
    stream=True,
):
    update_ui(partial)
```

Each addition is one line. No refactoring. No migration guide.

## What we intentionally DON'T do

### No prompt engineering

We don't write prompts for you. You know your domain better than we do.

```python
# We DON'T do this:
# @instructor.prompt("Extract the user information carefully")
# def extract_user(text: str):
#     ...


# You write your own prompts:
text = "Jane is 33"
_messages = [
    {"role": "system", "content": "You are a precise data extractor"},
    {"role": "user", "content": f"Extract user from: {text}"},
]
```

### No new abstractions

We don't invent concepts like "Agents", "Chains", or "Tools". Those are your domain concepts.

```python
import instructor
from pydantic import BaseModel

# We DON'T do this:
# class UserExtractionAgent(instructor.Agent):
#     tools = [instructor.WebSearch(), instructor.Calculator()]


class User(BaseModel):
    name: str
    age: int


def search_web(query: str) -> str:
    return f"Results for {query}"


client = instructor.from_provider("openai/gpt-4.1-mini")


# You build what makes sense:
def extract_user_with_search(query: str) -> User:
    # Your logic, your way
    search_results = search_web(query)
    return client.create(
        response_model=User, messages=[{"role": "user", "content": search_results}]
    )


_user = extract_user_with_search("Find Jane")
```

### No framework lock-in

Your code should work with or without us:

```python
import instructor
from pydantic import BaseModel


# This is just a Pydantic model
class User(BaseModel):
    name: str
    age: int


# This is just a function
def process_user(user: User) -> dict:
    return {"name": user.name.upper(), "adult": user.age >= 18}


client = instructor.from_provider("openai/gpt-4.1-mini")

# Instructor just connects them to LLMs
user = client.create(
    response_model=User,
    messages=[{"role": "user", "content": "Jane is 33"}],
)

_result = process_user(user)  # Works with or without Instructor
```

## The result

By following these principles, we get:

- **Tiny API surface**: Learn it in minutes, not days
- **Zero vendor lock-in**: Switch providers or remove Instructor anytime
- **Debuggable**: When things break, you can see why
- **Composable**: Build complex systems from simple parts
- **Pythonic**: If it feels natural in Python, it feels natural in Instructor

## In practice

Here's what building with Instructor actually looks like:

```python
from enum import Enum
from typing import List

import instructor
from pydantic import BaseModel


# Your domain models (not ours)
class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Ticket(BaseModel):
    title: str
    description: str
    priority: Priority
    estimated_hours: float


# Your business logic (not ours)
def prioritize_tickets(tickets: List[Ticket]) -> List[Ticket]:
    return sorted(tickets, key=lambda t: (t.priority.value, -t.estimated_hours))


# Connect to LLM (one line)
client = instructor.from_provider("openai/gpt-4.1-mini")

# Extract structured data (simple function call)
tickets = client.create(
    response_model=List[Ticket],
    messages=[{"role": "user", "content": "Parse these support tickets: ..."}],
)

# Use your business logic
_prioritized = prioritize_tickets(tickets)
```

No framework. No abstractions. Just Python.

## The philosophy in one sentence

**Make structured LLM outputs as easy as defining a Pydantic model.**

Everything else follows from that.
