---
title: LLM Validation Basics with Instructor
description: Master the fundamentals of validating LLM outputs to ensure reliable, business-compliant structured data from GPT-4, Claude, and other models.
---

# LLM Validation Tutorial: Ensure Data Quality with Instructor

Master the fundamentals of validating LLM outputs in this comprehensive tutorial. Learn how to use Instructor's validation system to ensure GPT-4, Claude, and other language models produce reliable, business-compliant structured data.

## Why LLM Output Validation is Critical

When extracting structured data from LLMs, validation ensures:

1. **Data Integrity**: LLM outputs contain all required fields with correct formats
2. **Business Compliance**: Extracted data adheres to your domain rules and constraints
3. **Production Reliability**: LLM responses meet quality standards before entering your system

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ LLM         │ -> │ Instructor   │ -> │ Validated   │
│ Generates   │    │ Validates    │    │ Structured  │
│ Response    │    │ Structure    │    │ Data        │
└─────────────┘    └──────────────┘    └─────────────┘
                          │
                          │ If validation fails
                          ▼
                   ┌─────────────┐
                   │ Retry with  │
                   │ Feedback    │
                   └─────────────┘
```

## Basic LLM Validation Example

See how Instructor validates LLM outputs automatically:

```python
from pydantic import BaseModel, Field
import instructor
# Define validation rules for LLM extraction
class UserProfile(BaseModel):
    name: str
    age: int = Field(ge=13, description="User's age in years")

# Extract and validate LLM output
client = instructor.from_provider("openai/gpt-5-nano")
response = client.create(
    model="gpt-5.4-mini",  # Works with GPT-4, Claude, Gemini
    messages=[
        {"role": "user", "content": "My name is Jane Smith and I'm 25 years old."}
    ],
    response_model=UserProfile  # Automatic validation
)

print(f"User: {response.name}, Age: {response.age}")
```

Key validation features in this LLM tutorial:
- **Constraint Validation**: Age must be ≥ 13 years
- **Automatic Retry**: If LLM output fails validation, Instructor retries with error context
- **Type Safety**: Ensures LLM returns proper data types

## Essential LLM Validation Patterns

Common validation rules for LLM outputs:

| Validation | Example | What It Does |
|------------|---------|-------------|
| Type checking | `age: int` | Ensures value is an integer |
| Required fields | `name: str` | Field must be present |
| Optional fields | `middle_name: Optional[str] = None` | Field can be missing |
| Minimum value | `age: int = Field(ge=18)` | Value must be ≥ 18 |
| Maximum value | `rating: float = Field(le=5.0)` | Value must be ≤ 5.0 |
| String length | `username: str = Field(min_length=3)` | String must be at least 3 chars |

## How LLM Output Validation Works

The LLM validation pipeline in Instructor:

1. **LLM Generation**: Language model produces structured output
2. **Schema Matching**: Instructor maps LLM response to your Pydantic model
3. **Validation Check**: Pydantic validates against defined constraints
4. **Smart Retry**: On failure, errors are sent back to the LLM with context
5. **Success or Timeout**: Process continues until valid output or retry limit

## Enhance LLM Validation with Custom Messages

Guide LLMs with specific error messages for better corrections:

```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    name: str
    price: float = Field(
        gt=0,
        description="Product price in USD",
        json_schema_extra={"error_msg": "Price must be greater than zero"}
    )
```

## Common LLM Validation Use Cases

- **Age Verification**: Ensure extracted ages meet minimum requirements
- **Price Validation**: Verify LLM-extracted prices are positive numbers
- **Email Format**: Validate email addresses from unstructured text
- **Date Constraints**: Ensure dates are within valid ranges
- **Business Rules**: Enforce domain-specific constraints on LLM outputs

## Continue Your LLM Validation Journey

- **[Custom Validators](custom_validators.md)** - Build complex validation logic for LLM outputs
- **[Retry Mechanisms](retry_mechanisms.md)** - Configure how Instructor handles validation failures
- **[Field-Level Validation](field_level_validation.md)** - Validate individual fields in LLM responses

Master validation to ensure your LLM applications produce reliable, production-ready data!