---
title: Working with Optional Fields in Instructor
description: Learn how to use optional fields in Pydantic models to handle missing or uncertain information from LLM outputs.
---

# Optional Fields

This guide explains how to work with optional fields in your data models. Optional fields allow the model to skip fields when information is unavailable or uncertain.

## Why Use Optional Fields?

Optional fields are useful when:

1. Some information is missing from the input text
2. Certain fields are only relevant in specific contexts
3. The LLM can't confidently extract all fields
4. You want to allow partial success instead of complete failure

## Basic Optional Fields

To make a field optional, use Python's `Optional` type and provide a default value:

```python
from typing import Optional
from pydantic import BaseModel
import instructor
client = instructor.from_provider("openai/gpt-5-nano")

class Person(BaseModel):
    name: str  # Required field
    age: Optional[int] = None  # Optional field with None default
    occupation: Optional[str] = None  # Optional field with None default
```

Here, `name` is required, while `age` and `occupation` are optional and will default to `None` if not found.

## Using Default Values

You can provide meaningful default values for optional fields:

```python
from typing import List
from pydantic import BaseModel
import instructor
client = instructor.from_provider("openai/gpt-5-nano")

class Product(BaseModel):
    name: str
    price: float
    currency: str = "USD"  # Default value
    in_stock: bool = True  # Default value
    tags: List[str] = []  # Default empty list
```

## Optional Fields with Validation

You can add the `Field` class for more control and validation:

```python
from typing import Optional
from pydantic import BaseModel, Field
import instructor
client = instructor.from_provider("openai/gpt-5-nano")

class UserProfile(BaseModel):
    username: str
    email: str
    bio: Optional[str] = Field(
        None,  # Default value
        max_length=200,  # Validation applies if present
        description="User's biography, limited to 200 characters"
    )
```

## Optional Nested Structures

Entire nested structures can be optional:

```python
from typing import Optional
from pydantic import BaseModel
import instructor
client = instructor.from_provider("openai/gpt-5-nano")

class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str

class Contact(BaseModel):
    email: str
    phone: Optional[str] = None
    address: Optional[Address] = None  # Optional nested structure

class Person(BaseModel):
    name: str
    contact: Contact
```

When using nested optional structures, check if they exist before accessing:

```python
# Access nested data safely
if person.contact.address:
    print(f"Address: {person.contact.address.city}")
else:
    print("No address information available")
```

## Using `Maybe` for Uncertain Fields

Instructor provides a `Maybe` model factory for extractions that might not produce a result. Use `Optional` for individual nullable fields. To represent a potentially missing extraction, pass a Pydantic model to `Maybe`:

```python
from typing import Optional

from instructor import Maybe
from pydantic import BaseModel


class PersonInfo(BaseModel):
    name: str
    age: Optional[int] = None


MaybePerson = Maybe(PersonInfo)
person = MaybePerson(result=PersonInfo(name="Alex", age=32))
```

Use `MaybePerson` as the `response_model` when extracting with an Instructor client. The example above constructs a response locally, without an LLM call.

The returned model has `result`, `error`, and `message` fields. Check `result` before accessing the extracted data:

```python
if person.result is not None:
    print(f"Name: {person.result.name}")
    print(f"Age: {person.result.age}")
elif person.error:
    print(person.message or "Could not extract person information")
else:
    print("No person information available")
```

When an extraction fails, `result` can be `None` and `message` can explain why:

```python
missing_person = MaybePerson(error=True, message="No person information found")
print(missing_person.result)
print(missing_person.message)
```

This prints `None` and `No person information found`.

For more about the `Maybe` type, see the [Missing Concepts](../../concepts/maybe.md) page.

## Handling Optional Values

Always handle the possibility of `None` values in your code:

```python
# Check for None before using
if person.age is not None:
    drinking_age = "Legal" if person.age >= 21 else "Underage"
else:
    drinking_age = "Unknown"

# Use conditional expressions
price_display = f"${product.price}" if product.price is not None else "Price unavailable"

# Provide defaults with 'or'
display_name = user.nickname or user.username
```

## Validation with Optional Fields

Optional fields can still have validation when they're present:

```python
from typing import Optional
from pydantic import BaseModel, field_validator
import instructor
import re
client = instructor.from_provider("openai/gpt-5-nano")

class ContactInfo(BaseModel):
    email: str
    phone: Optional[str] = None

    @field_validator('phone')
    @classmethod
    def validate_phone(cls, v):
        if v is not None and not re.match(r'^\+?[1-9]\d{1,14}$', v):
            raise ValueError("Invalid phone format")
        return v
```

## Related Resources

- [Simple Object Extraction](./simple_object.md) - Extracting basic objects
- [Field Validation](./field_validation.md) - Adding validation to fields
- [Nested Structure](./nested_structure.md) - Working with complex data
- [Missing Concepts](../../concepts/maybe.md) - Using the Maybe type for uncertain fields

## Next Steps

- Learn about [Field Validation](./field_validation.md)
- Explore [Nested Structure](./nested_structure.md) for complex data
- Check out [Prompt Templates](./prompt_templates.md) for crafting prompts
