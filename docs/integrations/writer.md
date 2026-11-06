---
title: Structured Outputs with Writer, a complete guide with instructor
description: Learn how to use Writer for structured outputs using their latest Palmyra-X-004 model for more reliable system outputs
---

# Structured Outputs with Writer, a complete guide with instructor

This guide demonstrates how to use Writer for structured outputs using their latest Palmyra-X-004 model for more reliable system outputs.

You'll need to sign up for an account and get an API key. You can do that [here](https://writer.com).

```bash
export WRITER_API_KEY=<your-api-key-here>
pip install "instructor[writer]"
```

## Palmyra-X-004

Writer supports structured outputs with their latest Palmyra-X-004 model that introduces tool calling functionality

### Sync Example

```python
import instructor
from pydantic import BaseModel

# Initialize Writer client
client = instructor.from_provider("writer/palmyra-x5")


class User(BaseModel):
    name: str
    age: int


# Extract structured data
user = client.create(
    messages=[{"role": "user", "content": "Extract: John is 30 years old"}],
    response_model=User,
)

print(user)
#> name='John' age=30
```

### Async Example

```python
import instructor
from pydantic import BaseModel
import asyncio

client = instructor.from_provider(
    "writer/palmyra-x5",
    async_client=True,
)


class User(BaseModel):
    name: str
    age: int


async def extract_user():
    # Extract structured data
    user = await client.create(
        messages=[{"role": "user", "content": "Extract: John is 30 years old"}],
        response_model=User,
    )

    print(user)
    #> name='John' age=30


if __name__ == "__main__":
    import asyncio

    asyncio.run(extract_user())
```

## Nested Objects

Writer also supports nested objects, which is useful for extracting data from more complex responses.

```python
import instructor
from pydantic import BaseModel

# Initialize Writer client
client = instructor.from_provider("writer/palmyra-x5")


class Address(BaseModel):
    street: str
    city: str
    country: str


class User(BaseModel):
    name: str
    age: int
    addresses: list[Address]


# Create structured output with nested objects
user = client.create(
    messages=[
        {
            "role": "user",
            "content": """
            Extract: Jason is 25 years old.
            He lives at 123 Main St, New York, USA
            and has a summer house at 456 Beach Rd, Miami, USA
        """,
        },
    ],
    response_model=User,
)
print(user)
#> {
#>     'name': 'Jason',
#>     'age': 25,
#>     'addresses': [
#>         {
#>             'street': '123 Main St',
#>             'city': 'New York',
#>             'country': 'USA'
#>         },
#>         {
#>             'street': '456 Beach Rd',
#>             'city': 'Miami',
#>             'country': 'USA'
#>         }
#>     ]
#> }
```

## Streaming Support

Instructor has two main ways that you can use to stream responses out

1. **Iterables**: These are useful when you'd like to stream a list of objects of the same type (Eg. use structured outputs to extract multiple users)
2. **Partial Streaming**: This is useful when you'd like to stream a single object and you'd like to immediately start processing the response as it comes in.

We currently support streaming for Writer with native tool for both methods listed above.

### Partial Streaming

```python
import instructor
from pydantic import BaseModel

client = instructor.from_provider("writer/palmyra-x5")


class Person(BaseModel):
    name: str
    age: int


resp = client.create_partial(
    messages=[
        {
            "role": "user",
            "content": "Ivan is 27 and lives in Singapore",
        }
    ],
    response_model=Person,
)

for person in resp:
    print(person)
    #> name=None age=None
    #> name='Ivan' age=None
    #> name='Ivan' age=27
```
