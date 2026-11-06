# Retry Mechanisms

Retry mechanisms in Instructor handle validation failures by giving the LLM another chance to generate valid responses. This guide explains how retries work and how to customize them for your use case.

## How Retries Work

When validation fails, Instructor:

1. Captures the validation error(s)
2. Formats them as feedback
3. Adds the feedback to the prompt context
4. Asks the LLM to try again with this new information

This creates a feedback loop that helps the LLM correct its output until it produces a valid response.

## Basic Retry Example

Here's a simple example showing retries in action:

```python
import instructor
from pydantic import BaseModel, Field, field_validator

# Initialize the client with max_retries
client = instructor.from_provider(
    "openai/gpt-4o",
    max_retries=2  # Will try up to 3 times (initial + 2 retries)
)

class Product(BaseModel):
    name: str
    price: float = Field(..., gt=0)

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if len(v) < 3:
            raise ValueError("Product name must be at least 3 characters")
        return v

# This will automatically retry if validation fails
response = client.create(
    model="gpt-5.4-mini",
    messages=[
        {"role": "user", "content": "Product: Pen, Price: -5"}
    ],
    response_model=Product
)
```

In this example, the initial response will likely fail validation because:
- The price is negative (violating the `gt=0` constraint)
- Instructor will automatically retry with feedback about these issues

For more details on max_retries configuration, see the [Retrying](../../concepts/retrying.md) concepts page.

## Customizing Retry Behavior

You can customize retry behavior when initializing the Instructor client:

```python
import instructor

# Customize retry behavior
client = instructor.from_provider(
    "openai/gpt-4o",
    max_retries=3,                   # Maximum number of retries
    retry_if_parsing_fails=True,     # Retry on JSON parsing failures
    throw_error=True                 # Throw an error if all retries fail
)
```

### Retry Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `max_retries` | Maximum number of retry attempts | 0 |
| `retry_if_parsing_fails` | Whether to retry if JSON parsing fails | True |
| `throw_error` | Whether to throw an error if all retries fail | True |

## Handling Retry Failures

When all retries fail, Instructor raises an `InstructorRetryException` that contains comprehensive information about all failed attempts:

```python
from instructor.core.exceptions import InstructorRetryException

try:
    response = client.create(
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": "Product: Invalid data"}],
        response_model=Product,
        max_retries=3
    )
except InstructorRetryException as e:
    print(f"Failed after {e.n_attempts} attempts")
    print(f"Total usage: {e.total_usage}")
    
    # New: Access detailed information about each failed attempt
    for attempt in e.failed_attempts:
        print(f"Attempt {attempt.attempt_number}: {attempt.exception}")
        if attempt.completion:
            # Analyze the raw completion that failed validation
            print(f"Raw response: {attempt.completion}")
```

The `InstructorRetryException` now includes:

- `failed_attempts`: A list of `FailedAttempt` objects containing:
  - `attempt_number`: The retry attempt number
  - `exception`: The specific exception that occurred
  - `completion`: The raw LLM response (when available)
- `n_attempts`: Total number of attempts made
- `total_usage`: Total token usage across all attempts
- `last_completion`: The final failed completion
- `messages`: The conversation history

This comprehensive tracking enables better debugging and analysis of retry patterns.

For more on handling validation failures, see [Fallback Strategies](../../concepts/error_handling.md).

## Error Messages and Feedback

Instructor provides detailed error messages to the LLM during retries:

```
The following errors occurred during validation:
- price: ensure this value is greater than 0
- name: Product name must be at least 3 characters

Please fix these errors and ensure the response is valid.
```

This feedback helps the LLM understand exactly what needs to be fixed.

## Retry Limitations

While retries are powerful, they have some limitations:

1. **Retry Budget**: Each retry consumes tokens and time
2. **Persistent Errors**: Some errors might not be fixable by the LLM
3. **Model Limitations**: Some models may consistently struggle with certain validations

For complex validation scenarios, consider implementing [Custom Validators](custom_validators.md) or [Field-level Validation](field_level_validation.md).

## Advanced Retry Pattern: Progressive Validation

For complex schemas, you can implement a progressive validation pattern:

```python
import instructor
from pydantic import BaseModel, Field

# Initialize with moderate retries
client = instructor.from_provider(
    "openai/gpt-4o",
    max_retries=2
)

# Basic validation first
class BasicProduct(BaseModel):
    name: str
    price: float = Field(..., gt=0)

# Advanced validation second
class DetailedProduct(BasicProduct):
    description: str = Field(..., min_length=10)
    category: str
    in_stock: bool

# Two-step extraction with validation
try:
    # First get basic fields
    basic = client.create(
        model="gpt-5.4-mini",
        messages=[
            {"role": "user", "content": "Product: Mini Pen, Price: $2.50"}
        ],
        response_model=BasicProduct
    )

    # Then get full details with context from the first step
    detailed = client.create(
        model="gpt-5.4-mini",
        messages=[
            {"role": "user", "content": f"Provide more details about {basic.name} which costs ${basic.price}"}
        ],
        response_model=DetailedProduct
    )
except Exception as e:
    # Handle validation failures
    print(f"Validation failed: {e}")
```

## Related Resources

- [Retrying](../../concepts/retrying.md) - Core retry concepts
- [Validation](../../concepts/validation.md) - Main validation documentation
- [Custom Validators](../../concepts/reask_validation.md) - Creating custom validation logic
- [Fallback Strategies](../../concepts/error_handling.md) - Handling persistent validation failures
- [Self Critique](../../examples/self_critique.md) - Example of model self-correction

## Next Steps

- Learn about [Field-level Validation](field_level_validation.md)
- Implement [Custom Validators](custom_validators.md)