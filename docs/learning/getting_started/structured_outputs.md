---
title: Getting Started with Structured LLM Outputs
description: Learn the basics of extracting structured data from language models using Instructor. Understand the difference between unstructured and structured outputs.
---

# Getting Started with Structured Outputs

Large language models (LLMs) are powerful tools for generating text, but extracting structured data from their outputs can be challenging. Structured outputs solve this problem by having LLMs return data in consistent, machine-readable formats.

## The Problem with Unstructured Outputs

Let's look at what happens when we ask an LLM to extract information without any structure:

```python
from openai import OpenAI

client = OpenAI()
response = client.create(
    model="gpt-5.4-mini",
    messages=[
        {
            "role": "user",
            "content": "Extract customer: John Doe, age 35, email: john@example.com",
        }
    ],
)

print(response.choices[0].message.content)
```

The output might look like:
```
Customer Name: John Doe
Age: 35
Email: john@example.com
```

Or it could be:
```
I found the following customer information:
- Name: John Doe
- Age: 35
- Email address: john@example.com
```

This inconsistency makes it difficult to reliably parse the information in downstream applications.

## The Solution: Structured Outputs with Instructor

Instructor solves this problem by using Pydantic models to define the expected structure of the output:

```python
import instructor
from pydantic import BaseModel, Field, EmailStr
class Customer(BaseModel):
    name: str = Field(description="Customer's full name")
    age: int = Field(description="Customer's age in years", ge=0, le=120)
    email: EmailStr = Field(description="Customer's email address")

client = instructor.from_provider("openai/gpt-5-nano")
customer = client.create(
    model="gpt-5.4-mini",
    messages=[
        {
            "role": "user",
            "content": "Extract customer: John Doe, age 35, email: john@example.com",
        }
    ],
    response_model=Customer,  # This is the key part
)

print(customer)  # Customer(name='John Doe', age=35, email='john@example.com')
print(f"Name: {customer.name}, Age: {customer.age}, Email: {customer.email}")
```

The benefits of this approach include:

1. **Consistency**: Always get data in the same format
2. **Validation**: Age must be between 0 and 120, email must be valid
3. **Type Safety**: `age` is always an integer, not a string
4. **Documentation**: Model fields are self-documenting with descriptions

## Complex Example: Nested Structures

Instructor shines with complex data structures:

```python
from typing import List, Optional
from pydantic import BaseModel, Field
import instructor
client = instructor.from_provider("openai/gpt-5-nano")

class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str

class Contact(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None

class Person(BaseModel):
    name: str
    age: int
    occupation: str
    address: Address
    contact: Contact
    skills: List[str] = Field(description="List of professional skills")

person = client.create(
    model="gpt-5.4-mini",
    messages=[
        {
            "role": "user",
            "content": """
        Extract detailed information for this person:
        John Smith is a 42-year-old software engineer living at 123 Main St, San Francisco, CA 94105.
        His email is john.smith@example.com and phone is 555-123-4567.
        John is skilled in Python, JavaScript, and cloud architecture.
        """,
        }
    ],
    response_model=Person,
)

print(f"Name: {person.name}")
print(f"Location: {person.address.city}, {person.address.state}")
print(f"Skills: {', '.join(person.skills)}")
```

## Installation

To get started with Instructor, install it via pip:

```shell
pip install instructor pydantic
```

You'll also need to set up your API keys for the LLM provider you're using.

## Next Steps

In the next sections, you'll learn how to:

1. Create your [first extraction](first_extraction.md)
2. Understand the different [response models](response_models.md) you can create
3. Set up [clients for various LLM providers](../../concepts/from_provider.md)