---
title: Understanding Response Models in Instructor
description: Learn how to create response models with Pydantic to define structure, validation rules, and extract complex data from LLMs.
---

# Understanding Response Models

Response models are at the core of Instructor's functionality. They define the structure of the data you want to extract and provide validation rules. This guide explains how to create different types of response models for various use cases.

## Basic Models

Let's start with a simple model similar to what we've seen before:

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
```

This defines a model with two required fields: `name` (a string) and `age` (an integer).

## Adding Field Metadata

You can add metadata to fields using the `Field` class:

```python
from pydantic import BaseModel, Field

class WeatherForecast(BaseModel):
    """Weather forecast for a specific location"""

    temperature: float = Field(
        description="Current temperature in Celsius"
    )
    condition: str = Field(
        description="Weather condition (sunny, cloudy, rainy, etc.)"
    )
    humidity: int = Field(
        description="Humidity percentage from 0-100"
    )
```

Field descriptions help the LLM understand what information to extract for each field.

## Field Validation

You can add validation rules to ensure the extracted data meets your requirements:

```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    name: str = Field(min_length=3)
    price: float = Field(gt=0)  # greater than 0
    quantity: int = Field(ge=0)  # greater than or equal to 0
    description: str = Field(max_length=500)
```

Common validation parameters include:
- `min_length`/`max_length`: For strings
- `ge`/`gt`/`le`/`lt`: For numbers (greater/less than or equal/than)
- `pattern`: For regex pattern matching

For more on validation, see the [Field Validation](../patterns/field_validation.md) and [Validation Basics](../validation/basics.md) guides.

## Nested Models

You can create complex data structures with nested models:

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class Address(BaseModel):
    street: str
    city: str
    state: Optional[str] = None
    country: str

class User(BaseModel):
    name: str
    age: int
    addresses: List[Address]
```

This allows you to extract hierarchical data structures. For more examples, check out the [Simple Nested Structure](../patterns/nested_structure.md) guide.

## Using Enums

Enums help when you want to restrict a field to a set of specific values:

```python
from enum import Enum
from pydantic import BaseModel

class UserType(str, Enum):
    ADMIN = "admin"
    REGULAR = "regular"
    GUEST = "guest"

class User(BaseModel):
    name: str
    user_type: UserType
```

## Optional Fields

For fields that might not be present in the source text:

```python
from typing import Optional
from pydantic import BaseModel

class Contact(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    address: Optional[str] = None
```

For more about working with optional fields, see the [Optional Fields](../patterns/optional_fields.md) guide.

## Lists and Arrays

To extract multiple items of the same type:

```python
from typing import List
from pydantic import BaseModel

class BlogPost(BaseModel):
    title: str
    content: str
    tags: List[str]
```

For more about working with lists, see the [List Extraction](../patterns/list_extraction.md) guide.

## Using Your Models with Instructor

Once you've defined your model, you can use it for extraction:

```python
import instructor
client = instructor.from_provider("openai/gpt-5-nano")

forecast = client.create(
    model="gpt-5.4-mini",
    response_model=WeatherForecast,
    messages=[
        {"role": "user", "content": "What's the weather in New York today?"}
    ]
)

print(forecast.model_dump_json(indent=2))
```

## Model Documentation

You can add documentation to your models using docstrings and field descriptions:

```python
from pydantic import BaseModel, Field

class Investment(BaseModel):
    """Represents an investment opportunity with risk and return details."""

    name: str = Field(description="Name of the investment")
    amount: float = Field(description="Investment amount in USD")
    expected_return: float = Field(description="Expected annual return percentage")
    risk_level: str = Field(description="Risk level (low, medium, high)")
```

This documentation helps both the LLM understand what to extract and makes your code more maintainable.

## Advanced Validation with Validators

For more complex validation rules, you can use validator methods:

```python
from pydantic import BaseModel, Field, field_validator
from datetime import date

class Reservation(BaseModel):
    check_in: date
    check_out: date
    guests: int = Field(ge=1)

    @field_validator("check_out")
    def check_dates(cls, v, values):
        if "check_in" in values.data and v <= values.data["check_in"]:
            raise ValueError("check_out must be after check_in")
        return v
```

For more advanced validation techniques, check out the [Custom Validators](../validation/custom_validators.md) guide.

## Next Steps

In the next section, learn about [from_provider](../../concepts/from_provider.md) to configure different LLM providers and understand the various modes of operation.