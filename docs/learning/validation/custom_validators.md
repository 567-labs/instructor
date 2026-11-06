---
title: Custom Validators for LLM Outputs
description: Learn to build custom validators for LLM outputs using rule-based and semantic validation techniques with Instructor.
---

# Custom LLM Validators Tutorial: Advanced Data Quality Control

Learn how to build custom validators for LLM outputs in this advanced tutorial. Master both rule-based and semantic validation techniques to ensure GPT-4, Claude, and other language models produce data that meets your exact requirements.

## Basic Custom Validator

Custom validators are functions that validate field values and can be applied using Pydantic's field validators.

```python
from pydantic import BaseModel, field_validator
import instructor

# Initialize the client
client = instructor.from_provider("openai/gpt-5-nano")

class Person(BaseModel):
    name: str
    age: int

    @field_validator('age')
    @classmethod
    def validate_age(cls, value):
        if value < 0 or value > 120:
            raise ValueError("Age must be between 0 and 120")
        return value

# Extract data with validation
response = client.create(
    model="gpt-5.4-mini",
    messages=[
        {"role": "user", "content": "The person's name is John and they are 150 years old."}
    ],
    response_model=Person
)
```

If the model returns an age outside the valid range, Instructor will retry the request with specific feedback about the validation failure.

For more information on how Instructor handles validation and retries, see [Validation Basics](../../concepts/validation.md) and the [Retrying](../../concepts/retrying.md) concepts page.

## Complex Validation

You can create more complex validators that check multiple fields or have conditional logic:

```python
from pydantic import BaseModel, field_validator, model_validator
import instructor
from typing import List, Optional
from datetime import date
client = instructor.from_provider("openai/gpt-5-nano")

class Employee(BaseModel):
    name: str
    hire_date: date
    termination_date: Optional[date] = None
    skills: List[str]

    @field_validator('skills')
    @classmethod
    def validate_skills(cls, skills):
        if len(skills) < 1:
            raise ValueError("Employee must have at least one skill")
        return skills

    @model_validator(mode='after')
    def validate_dates(self):
        if self.termination_date and self.termination_date < self.hire_date:
            raise ValueError("Termination date cannot be before hire date")
        return self
```

For more advanced validation approaches, check out [Field-level Validation](../../concepts/fields.md) and the [Validators](../../concepts/reask_validation.md) concepts page.

## Handling Complex Data Types

Custom validators can also process more complex data types and perform transformations:

```python
from pydantic import BaseModel, field_validator
import instructor
import re
client = instructor.from_provider("openai/gpt-5-nano")

class Contact(BaseModel):
    name: str
    email: str
    phone: str

    @field_validator('email')
    @classmethod
    def validate_email(cls, value):
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, value):
            raise ValueError("Invalid email format")
        return value

    @field_validator('phone')
    @classmethod
    def validate_phone(cls, value):
        # Remove non-digit characters and validate
        digits_only = re.sub(r'\D', '', value)
        if len(digits_only) < 10:
            raise ValueError("Phone number must have at least 10 digits")
        return digits_only  # Return the cleaned version
```

For a practical example of extraction with validation, see the [Contact Information Extraction](../../examples/extract_contact_info.md) example.

## Using External Services for Validation

You can also use external services or APIs for validation:

```python
from pydantic import BaseModel, field_validator
import instructor
import requests
client = instructor.from_provider("openai/gpt-5-nano")

class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str

    @field_validator('zip_code')
    @classmethod
    def validate_zip_code(cls, value):
        # Example of validation using an external service (simplified)
        # In a real app, you might use a postal code validation API
        if not (value.isdigit() and len(value) == 5):
            raise ValueError("Zip code must be 5 digits")
        return value
```

## Semantic Validation with LLMs

For complex validation scenarios where rule-based validation is difficult, Instructor provides semantic validation capabilities using LLMs via the `llm_validator` function. For a comprehensive guide on this topic, see the dedicated [Semantic Validation](../../concepts/semantic_validation.md) page:

```python
from typing import Annotated
from pydantic import BaseModel, BeforeValidator
import instructor
from instructor import llm_validator

client = instructor.from_provider("openai/gpt-5-nano")

class ProductDescription(BaseModel):
    product_name: str
    description: Annotated[
        str,
        BeforeValidator(
            llm_validator(
                "The description must be professional, accurate, and free of hyperbole. "
                "It should not make unsubstantiated claims or use superlatives excessively.",
                client=client
            )
        )
    ]

# This would fail validation because it uses excessive hyperbole
try:
    product = ProductDescription(
        product_name="SuperClean 3000",
        description="The absolute BEST cleaning product in the world! Will change your life FOREVER! Makes every other cleaning product completely OBSOLETE!"
    )
except ValueError as e:
    print(e)  # The validation error would explain the issue with the hyperbolic language
```

Semantic validation is particularly useful for validating against criteria that are:

1. **Subjective** - Such as tone, style, or appropriateness
2. **Contextual** - Requiring understanding of relationships between elements
3. **Complex** - Where multiple interrelated factors need to be evaluated together
4. **Hard to formalize** - When rules would be too numerous or complex to express programmatically

Unlike rule-based validators that check against predefined criteria, semantic validators leverage LLMs to evaluate content based on natural language instructions. They can understand nuance and context in ways that traditional validation cannot.

### When to Use Semantic Validation

Consider using semantic validation when:

- You need to enforce style guidelines or content policies
- Validating natural language content against subjective criteria
- Checking for consistency across multiple fields or complex relationships
- Traditional validation would require hundreds of individual rules

Remember that semantic validation requires additional API calls, which adds cost and latency to your application. Use it strategically for high-value validation needs rather than for simple constraints that can be handled with standard validators.

## Handling Validation Failures

When validation fails, Instructor can handle it in different ways. Learn more about:

- [Retry Mechanisms](../../concepts/retrying.md) for automatic retries with feedback
- [Self-Correction](../../examples/self_critique.md) for AI model self-correction techniques

## Best Practices for Custom Validators

1. **Be specific in error messages**: Provide clear error messages that explain exactly what went wrong
2. **Validate early**: Apply validators to individual fields when possible before model-level validation
3. **Keep validators focused**: Each validator should have a single responsibility
4. **Use type hints**: Proper type hints help both Pydantic and Instructor understand your data better
5. **Consider both validation and transformation**: Validators can both validate and transform data
6. **Choose appropriate validation type**: Use rule-based validation for simple, objective criteria and semantic validation for complex, subjective, or context-dependent validation
7. **Balance cost and benefits**: Consider the additional cost and latency of semantic validation against the value it provides

For more information on validation in general, check out the [Validation](../../concepts/validation.md) concepts page.

## Related Resources

- [Fields](../../concepts/fields.md) - Learn about field definitions and properties
- [Models](../../concepts/models.md) - Understand model creation and configuration
- [Types](../../concepts/types.md) - Explore the different data types you can use

Custom validators are a powerful way to ensure the data you extract meets your specific requirements, improving the reliability and quality of structured outputs from LLMs.