---
title: Field-level Validation with Instructor
description: Learn how to create specific validation rules for individual fields in your Pydantic models to ensure data quality.
---

# Field-level Validation

Field-level validation lets you create specific rules for individual fields in your data models. This guide shows how to use field-level validation with Instructor.

## What is Field-level Validation?

Field-level validation in Instructor uses Pydantic's validation features to:

1. Check individual fields with custom rules
2. Transform field values (like formatting or cleaning data)
3. Apply business rules to specific fields
4. Give clear feedback when values are invalid

Validation happens when your model is being processed, and if it fails, Instructor will retry with better instructions.

## Basic Field Validation

You can apply simple validation using Pydantic's Field constraints:

```python
from pydantic import BaseModel, Field
import instructor
client = instructor.from_provider("openai/gpt-5-nano")

class User(BaseModel):
    name: str = Field(..., min_length=2, description="User's full name")
    age: int = Field(..., ge=18, le=120, description="User's age in years")
    email: str = Field(
        ...,
        pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        description="Valid email address"
    )
```

For more details, see the [Fields](../../concepts/fields.md) concepts page.

## Custom Field Validators

For more complex rules, use the `field_validator` decorator:

```python
from pydantic import BaseModel, field_validator
import instructor
import re
client = instructor.from_provider("openai/gpt-5-nano")

class Product(BaseModel):
    name: str
    sku: str
    price: float

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if len(v.strip()) < 3:
            raise ValueError("Product name must be at least 3 characters long")
        return v.strip().title()  # Clean up and format

    @field_validator('sku')
    @classmethod
    def validate_sku(cls, v):
        pattern = r'^[A-Z]{3}-\d{4}$'
        if not re.match(pattern, v):
            raise ValueError("SKU must be in format XXX-0000 (3 uppercase letters, dash, 4 digits)")
        return v
```

## Validating Multiple Fields Together

Sometimes one field's validity depends on other fields. Use `model_validator` for this:

```python
from pydantic import BaseModel, model_validator
import instructor
from datetime import date
client = instructor.from_provider("openai/gpt-5-nano")

class Reservation(BaseModel):
    check_in: date
    check_out: date
    room_type: str
    guests: int

    @model_validator(mode='after')
    def validate_dates(self):
        if self.check_out <= self.check_in:
            raise ValueError("Check-out date must be after check-in date")

        if self.room_type == "Standard" and self.guests > 2:
            raise ValueError("Standard rooms can only fit 2 guests")

        return self
```

## How Validation Errors Are Handled

When validation fails, Instructor adds error details to help the AI fix the problem:

```
The following errors occurred during validation:
- product_sku: Product not found
- quantity: Quantity must be at least 1

Please fix these errors and ensure the response is valid.
```

## Best Practices

1. **Order matters**: Validators run in the order they're defined
2. **Clear messages**: Write specific error messages
3. **Clean first**: Handle data cleaning before validation
4. **Validate early**: Check fields before model-level validation
5. **Transform wisely**: Field validators can both check and change values

## Related Resources

- [Fields](../../concepts/fields.md) - Basic field properties
- [Custom Validators](../../concepts/reask_validation.md) - Creating custom validation logic
- [Validation Basics](../../concepts/validation.md) - Fundamental validation concepts
- [Retry Mechanisms](../../concepts/retrying.md) - How validation retries work
- [Fallback Strategies](../../concepts/error_handling.md) - Handling persistent validation failures
- [Types](../../concepts/types.md) - Understanding data types in Pydantic models

