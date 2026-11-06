# Field Validation

This guide covers how to add validation to fields when extracting structured data with Instructor. Field validation ensures that your extracted data meets specific criteria and constraints.

## Why Field Validation Matters

Field validation helps you:

1. Ensure data quality and consistency
2. Enforce business rules
3. Prevent errors in downstream processing
4. Provide clear feedback for invalid data

Instructor uses Pydantic's validation system, which is applied automatically during extraction.

## Basic Field Constraints

You can add basic constraints to fields using Pydantic's `Field` function:

```python
from pydantic import BaseModel, Field
import instructor
from openai import OpenAI

client = instructor.from_provider("openai/gpt-5-nano")

class User(BaseModel):
    name: str = Field(..., min_length=2, max_length=50)
    age: int = Field(..., ge=0, le=120)  # greater than or equal to 0, less than or equal to 120
    email: str = Field(..., pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

# Extract with validation
response = client.create(
    model="gpt-5.4-mini",
    messages=[
        {"role": "user", "content": "I'm John Smith, 35 years old, with email john@example.com"}
    ],
    response_model=User
)
```

Common Field constraints include:

| Constraint | Description | Example |
|------------|-------------|---------|
| `min_length` | Minimum string length | `min_length=2` |
| `max_length` | Maximum string length | `max_length=50` |
| `pattern` | Regex pattern to match | `pattern=r'^[0-9]+$'` |
| `gt` | Greater than | `gt=0` (for numbers) |
| `ge` | Greater than or equal | `ge=18` |
| `lt` | Less than | `lt=100` |
| `le` | Less than or equal | `le=120` |
| `min_items` | Minimum list items | `min_items=1` |
| `max_items` | Maximum list items | `max_items=10` |

For more information on field definitions, see the [Fields](../../concepts/fields.md) concepts page.

## Validation with Field Validators

For more complex validation logic, use Pydantic's `field_validator` decorator:

```python
from pydantic import BaseModel, Field, field_validator
import instructor
from openai import OpenAI
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
            raise ValueError("Product name must be at least 3 characters")
        return v.strip()

    @field_validator('sku')
    @classmethod
    def validate_sku(cls, v):
        if not re.match(r'^[A-Z]{3}-\d{4}$', v):
            raise ValueError("SKU must be in format XXX-0000")
        return v

    @field_validator('price')
    @classmethod
    def validate_price(cls, v):
        if v <= 0:
            raise ValueError("Price must be greater than zero")
        if v > 10000:
            raise ValueError("Price exceeds maximum allowed value")
        return v

# Extract validated data
response = client.create(
    model="gpt-5.4-mini",
    messages=[
        {"role": "user", "content": "Product: Wireless Headphones, SKU: ABC-1234, Price: $79.99"}
    ],
    response_model=Product
)
```

Field validators can:
- Perform complex validation logic
- Clean and normalize data
- Transform values
- Check values against external data sources

For more on custom validators, see the [Custom Validators](../validation/custom_validators.md) guide.

## Model-level Validation

Sometimes validation needs to check relationships between fields. For this, use `model_validator`:

```python
from pydantic import BaseModel, Field, model_validator
import instructor
from openai import OpenAI
from datetime import date

client = instructor.from_provider("openai/gpt-5-nano")

class DateRange(BaseModel):
    start_date: date
    end_date: date

    @model_validator(mode='after')
    def validate_date_range(self):
        if self.end_date < self.start_date:
            raise ValueError("End date must be after start date")
        return self
```

## Validation in Nested Structures

You can apply validation at any level in nested structures:

```python
from pydantic import BaseModel, Field, field_validator
import instructor
from openai import OpenAI
from typing import List

client = instructor.from_provider("openai/gpt-5-nano")

class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str

    @field_validator('state')
    @classmethod
    def validate_state(cls, v):
        valid_states = {"CA", "NY", "TX", "FL"}  # Example: just a few states
        if v not in valid_states:
            raise ValueError(f"State must be one of: {', '.join(valid_states)}")
        return v

    @field_validator('zip_code')
    @classmethod
    def validate_zip(cls, v):
        if not v.isdigit() or len(v) != 5:
            raise ValueError("ZIP code must be 5 digits")
        return v

class Person(BaseModel):
    name: str
    addresses: List[Address]  # Nested structure with validation
```

For more on nested structures, see the [Nested Structure](nested_structure.md) guide.

## List Item Validation

You can validate items in a list:

```python
from typing import List
from pydantic import BaseModel, Field, field_validator
import instructor
from openai import OpenAI

client = instructor.from_provider("openai/gpt-5-nano")

class TagList(BaseModel):
    tags: List[str] = Field(..., min_items=1, max_items=5)

    @field_validator('tags')
    @classmethod
    def validate_tags(cls, tags):
        # Convert all tags to lowercase
        tags = [tag.lower() for tag in tags]

        # Check for minimum length of each tag
        for tag in tags:
            if len(tag) < 2:
                raise ValueError("Each tag must be at least 2 characters")

        # Check for duplicates
        if len(tags) != len(set(tags)):
            raise ValueError("Tags must be unique")

        return tags
```

For more on lists, see the [List Extraction](list_extraction.md) guide.

## Using Enumerations for Validation

Enums provide a way to validate fields against a predefined set of values:

```python
from enum import Enum
from pydantic import BaseModel
import instructor
from openai import OpenAI

client = instructor.from_provider("openai/gpt-5-nano")

class Status(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class Task(BaseModel):
    title: str
    description: str
    status: Status  # Must be one of the enum values
    priority: Priority  # Must be one of the enum values

# Extract with enum validation
response = client.create(
    model="gpt-5.4-mini",
    messages=[
        {"role": "user", "content": "Task: Update website, Description: Refresh content on homepage, Status: pending, Priority: high"}
    ],
    response_model=Task
)
```

For more information on enums, see the [Enums](../../concepts/enums.md) concepts page.

## Custom Error Messages

You can customize validation error messages for better feedback:

```python
from pydantic import BaseModel, Field
import instructor
from openai import OpenAI

client = instructor.from_provider("openai/gpt-5-nano")

class CreditCard(BaseModel):
    number: str = Field(
        ...,
        pattern=r'^\d{16}$',
        json_schema_extra={"error_msg": "Credit card number must be exactly 16 digits"}
    )
    expiry_month: int = Field(
        ...,
        ge=1,
        le=12,
        json_schema_extra={"error_msg": "Expiry month must be between 1 and 12"}
    )
    expiry_year: int = Field(
        ...,
        ge=2023,
        le=2030,
        json_schema_extra={"error_msg": "Expiry year must be between 2023 and 2030"}
    )
    cvv: str = Field(
        ...,
        pattern=r'^\d{3,4}$',
        json_schema_extra={"error_msg": "CVV must be 3 or 4 digits"}
    )
```

## Handling Validation Failures

When validation fails, Instructor will:

1. Capture the validation error
2. Add the error message to the context
3. Retry the request with this feedback (if retries are enabled)

To control retry behavior:

```python
client = instructor.from_provider(
    "openai/gpt-4o",
    max_retries=2,  # Number of retries after the initial attempt
    throw_error=True  # Whether to raise an exception on validation failure
)
```

For more on retries, see the [Retry Mechanisms](../validation/retry_mechanisms.md) guide.

## Real-world Example: Form Data Validation

Here's a more complete example validating form inputs:

```python
from pydantic import BaseModel, Field, field_validator, model_validator
import instructor
import re
from datetime import date, datetime
from typing import Optional
client = instructor.from_provider("openai/gpt-5-nano")

class RegistrationForm(BaseModel):
    username: str = Field(..., min_length=3, max_length=20)
    email: str
    password: str
    confirm_password: str
    birth_date: date

    @field_validator('username')
    @classmethod
    def validate_username(cls, v):
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError("Username can only contain letters, numbers, and underscores")
        return v

    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', v):
            raise ValueError("Invalid email format")
        return v

    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not re.search(r'[A-Z]', v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r'[a-z]', v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search(r'[0-9]', v):
            raise ValueError("Password must contain at least one number")
        return v

    @field_validator('birth_date')
    @classmethod
    def validate_age(cls, v):
        today = date.today()
        age = today.year - v.year - ((today.month, today.day) < (v.month, v.day))
        if age < 18:
            raise ValueError("You must be at least 18 years old to register")
        return v

    @model_validator(mode='after')
    def passwords_match(self):
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self
```

## Related Resources

- [Validation Basics](../validation/basics.md) - Core validation concepts
- [Custom Validators](../validation/custom_validators.md) - Creating custom validation logic
- [Field-level Validation](../validation/field_level_validation.md) - Advanced field validation
- [Retry Mechanisms](../validation/retry_mechanisms.md) - Handling validation failures
- [Fields](../../concepts/fields.md) - Understanding field definitions
- [Enums](../../concepts/enums.md) - Using enumeration types

## Next Steps

- Learn about [Optional Fields](optional_fields.md) for handling missing data
- Explore [Custom Validators](../validation/custom_validators.md) for complex validation
- Check out [Nested Structure](nested_structure.md) for complex data relationships