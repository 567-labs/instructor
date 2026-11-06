---
title: Your First LLM Extraction with Instructor
description: Step-by-step tutorial for your first structured data extraction from language models using Instructor and Pydantic.
---

# Your First LLM Extraction: Structured Outputs Tutorial

Learn how to extract structured data from LLMs using Instructor in this hands-on tutorial. We'll build a simple yet powerful example that demonstrates how to transform unstructured text into validated Python objects using GPT-4, Claude, or any supported LLM.

## Quick Start: Extract Structured Data from LLMs

This LLM tutorial shows you how to extract structured information from natural language. We'll parse a person's name and age - a perfect starting point for understanding Instructor's power:

```python
from pydantic import BaseModel
import instructor
# 1. Define your data model for LLM extraction
class Person(BaseModel):
    name: str
    age: int

# 2. Initialize Instructor with your LLM provider
client = instructor.from_provider("openai/gpt-5-nano")

# 3. Extract structured data from LLM
person = client.create(
    response_model=Person,   # Type-safe extraction
    messages=[
        {"role": "user", "content": "John Doe is 30 years old"}
    ]
)

# 4. Use validated, structured data from LLM
print(f"Name: {person.name}, Age: {person.age}")
# Output: Name: John Doe, Age: 30
```

## How Instructor LLM Extraction Works

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ Define      │ -> │ Instruct LLM │ -> │ Get Typed   │
│ Structure   │    │ to Extract   │    │ Response    │
└─────────────┘    └──────────────┘    └─────────────┘
```

Understanding the LLM structured output pipeline:

### Step 1: Define Your LLM Output Schema

```python
class Person(BaseModel):
    name: str
    age: int
```

Pydantic models define the structure for LLM outputs:
- `name`: String field for extracting names from LLM
- `age`: Integer field with automatic type validation

### Step 2: Configure Your LLM Client

```python
client = instructor.from_provider("openai/gpt-5-nano")
```

Instructor enhances your LLM client with structured output capabilities. Works with OpenAI, Anthropic, Google, and 15+ providers.

### Step 3: Execute LLM Extraction

```python
person = client.create(
    response_model=Person,
    messages=[
        {"role": "user", "content": "John Doe is 30 years old"}
    ]
)
```

Key parameters for structured LLM outputs:
- `response_model`: Pydantic model for type-safe extraction
- `messages`: Input text for the LLM to process

Note: The model is already specified when creating the client with `from_provider()`, so you don't need to pass it again.

### Step 4: Work with Validated LLM Data

```python
print(f"Name: {person.name}, Age: {person.age}")
```

Get back a fully validated Python object from your LLM - no JSON parsing, no validation errors, just clean data ready to use.

## Enhance LLM Extraction with Field Descriptions

Improve LLM accuracy by providing clear field descriptions:

```python
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age in years")
```

Field descriptions act as prompts, guiding the LLM to extract exactly what you need.

## Handle Optional Data in LLM Responses

Real-world LLM extractions often have missing data. Handle it gracefully:

```python
from typing import Optional

class Person(BaseModel):
    name: str
    age: Optional[int] = None  # Now age is optional
```

## Continue Your LLM Tutorial Journey

You've successfully extracted structured data from an LLM! Next steps:

1. **[Advanced Response Models](response_models.md)** - Complex schemas for LLM outputs
2. **[Multi-Provider Setup](../../concepts/from_provider.md)** - Use GPT-4, Claude, Gemini interchangeably
3. **[Production Patterns](../patterns/simple_object.md)** - Real-world LLM extraction examples

## Common LLM Extraction Patterns

- **Entity Extraction**: Names, dates, locations from unstructured text
- **Sentiment Analysis**: Structured sentiment scores with reasoning
- **Data Classification**: Categorize text into predefined schemas
- **Information Parsing**: Convert documents into structured databases

Ready to build more complex LLM extractions? Continue to [Response Models](response_models.md) →
