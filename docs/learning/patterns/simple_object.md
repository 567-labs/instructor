---
title: Simple Object Extraction Pattern
description: Learn the fundamental pattern of extracting simple objects from text using Instructor with type-safe validation.
---

# Simple Object Extraction: LLM Tutorial for Structured Data

Learn how to extract structured objects from text using LLMs in this comprehensive tutorial. We'll cover the fundamental pattern of transforming unstructured text into validated Python objects using Instructor with GPT-4, Claude, and other language models.

## Basic LLM Object Extraction Tutorial

```python
from pydantic import BaseModel
import instructor
# Define your LLM extraction schema
class Person(BaseModel):
    name: str
    age: int
    occupation: str

# Extract structured data from LLM
client = instructor.from_provider("openai/gpt-5-nano")
person = client.create(
    model="gpt-5.4-mini",  # Works with GPT-4, Claude, Gemini
    messages=[
        {"role": "user", "content": "John Smith is a 35-year-old software engineer."}
    ],
    response_model=Person  # Type-safe LLM extraction
)

print(f"Name: {person.name}")
print(f"Age: {person.age}")
print(f"Occupation: {person.occupation}")
```

```
┌───────────────┐            ┌───────────────┐
│ Define Model  │            │ Extracted     │
│ name: str     │  Extract   │ name: "John"  │
│ age: int      │ ─────────> │ age: 35       │
│ occupation: str│            │ occupation:   │
└───────────────┘            │ "software..." │
                             └───────────────┘
```

## Enhance LLM Extraction with Field Descriptions

Guide your LLM with clear field descriptions for more accurate extraction:

```python
from pydantic import BaseModel, Field

class Book(BaseModel):
    title: str = Field(description="The full title of the book")
    author: str = Field(description="The author's full name")
    publication_year: int = Field(description="The year the book was published")
```

Field descriptions serve as prompts for the LLM, improving extraction accuracy and reducing errors in your structured outputs.

## Handle Missing Data in LLM Responses

Real-world LLM extractions often encounter missing information. Here's how to handle it gracefully:

```python
from typing import Optional
from pydantic import BaseModel

class MovieReview(BaseModel):
    title: str
    director: Optional[str] = None  # Optional field
    rating: float
```

Using `Optional` fields ensures your LLM extraction remains robust when dealing with incomplete or partial information.

## Validate LLM Outputs with Pydantic

Ensure LLM outputs meet your requirements with built-in validation:

```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    name: str
    price: float = Field(gt=0, description="The product price in USD")
    in_stock: bool
```

Pydantic validation ensures your LLM outputs are not just structured, but also correct and business-rule compliant.

## Production-Ready LLM Extraction Example

Here's a complete example showing nested object extraction from LLMs:

```python
from pydantic import BaseModel
from typing import Optional

class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    address: Optional[Address] = None

# Extract structured data
client = instructor.from_provider("openai/gpt-5-nano")
contact = client.create(
    model="gpt-5.4-mini",
    messages=[
        {"role": "user", "content": """
        Contact information:
        Name: Sarah Johnson
        Email: sarah.j@example.com
        Phone: (555) 123-4567
        Address: 123 Main St, Boston, MA 02108
        """}
    ],
    response_model=ContactInfo
)

print(f"Name: {contact.name}")
print(f"Email: {contact.email}")
```

## Common LLM Object Extraction Use Cases

- **Contact Information**: Extract names, emails, phones from unstructured text
- **Product Details**: Parse product descriptions into structured catalogs
- **Event Information**: Extract dates, locations, attendees from event descriptions
- **Entity Recognition**: Identify and structure people, places, organizations

## Continue Your LLM Tutorial Journey

- **[List Extraction Tutorial](list_extraction.md)** - Extract multiple objects from LLM responses
- **[Nested Structures](nested_structure.md)** - Handle complex hierarchical data from LLMs
- **[Advanced Validation](field_validation.md)** - Implement business rules for LLM outputs

Master these patterns to build production-ready LLM applications with reliable structured outputs!