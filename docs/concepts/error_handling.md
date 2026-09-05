---
title: Error Handling
description: Learn how to handle errors and exceptions when using Instructor for structured outputs.
---

# Error Handling

Instructor provides a comprehensive exception hierarchy to help you handle errors gracefully. All Instructor exceptions inherit from `InstructorError`.

## Exception Reference

| Exception | Description | Key Attributes |
|-----------|-------------|----------------|
| `InstructorError` | Base exception for all Instructor errors | - |
| `IncompleteOutputException` | Output truncated due to token limit | `last_completion` |
| `InstructorRetryException` | All retry attempts exhausted | `n_attempts`, `failed_attempts`, `total_usage` |
| `ValidationError` | Response validation failed | - |
| `ResponseParsingError` | Cannot parse LLM response | `mode`, `raw_response` |
| `ProviderError` | Provider-specific error | `provider` |
| `ConfigurationError` | Invalid configuration | - |
| `ModeError` | Invalid mode for provider | `mode`, `provider`, `valid_modes` |
| `ClientError` | Client initialization failed | - |
| `MultimodalError` | Processing image/audio/PDF failed | `content_type`, `file_path` |
| `AsyncValidationError` | Async validation failed | `errors` |

## Common Exceptions

### Incomplete Output

Raised when the LLM output is truncated due to reaching the token limit:

```python
import instructor
from pydantic import BaseModel
from instructor.core.exceptions import (
    IncompleteOutputException,
    InstructorRetryException,
)


class Report(BaseModel):
    content: str


client = instructor.from_provider("openai/gpt-4.1-mini", mode=instructor.Mode.JSON)

try:
    response = client.create(
        response_model=Report,
        messages=[{"role": "user", "content": "Write a long report..."}],
        max_tokens=50,
        max_retries=0,
    )
except (IncompleteOutputException, InstructorRetryException) as e:
    print(f"Output truncated: {e}")
    print(f"Last completion: {e.last_completion}")
```

### Retry Exhausted

Raised when all retry attempts fail:

```python
import instructor
from pydantic import BaseModel
from instructor.core.exceptions import InstructorRetryException


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider("openai/gpt-4.1-mini")

try:
    response = client.create(
        response_model=User,
        messages=[{"role": "user", "content": "Extract user info..."}],
        max_retries=3,
    )
except InstructorRetryException as e:
    print(f"Failed after {e.n_attempts} attempts")
    for attempt in e.failed_attempts:
        print(f"  Attempt {attempt.attempt_number}: {attempt.exception}")
```

### Validation Error

Raised when the response fails validation:

```python
import instructor
from pydantic import BaseModel, field_validator
from instructor.core.exceptions import ValidationError


class StrictModel(BaseModel):
    value: int

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Value must be positive")
        return v


client = instructor.from_provider("openai/gpt-4.1-mini")

try:
    response = client.create(
        response_model=StrictModel,
        messages=[{"role": "user", "content": "Extract data..."}],
    )
except ValidationError as e:
    print(f"Validation failed: {e}")
```

### Provider and Configuration Errors

Raised for provider-specific issues or invalid configuration:

```python
import instructor
from instructor.core.exceptions import ConfigurationError, ModeError

# Invalid provider format
try:
    client = instructor.from_provider("invalid-format")
except ConfigurationError as e:
    print(f"Configuration error: {e}")

# Wrong mode for provider
try:
    client = instructor.from_provider(
        "openai/gpt-4.1-mini",
        mode=instructor.Mode.TOOLS,
    )
except ModeError as e:
    print(f"Invalid mode. Valid modes: {e.valid_modes}")
```

## Best Practices

### Catch Specific Exceptions

```python
import logging
import instructor
from pydantic import BaseModel
from instructor.core.exceptions import (
    IncompleteOutputException,
    InstructorRetryException,
    ValidationError,
)

logger = logging.getLogger(__name__)


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider("openai/gpt-4.1-mini")

try:
    response = client.create(
        response_model=User,
        messages=[{"role": "user", "content": "Extract: Sam is 34"}],
    )
except IncompleteOutputException:
    logger.warning("Output truncated, retrying with more tokens")
    response = client.create(
        response_model=User,
        messages=[{"role": "user", "content": "Extract: Sam is 34"}],
        max_tokens=2000,
    )
except InstructorRetryException as e:
    logger.error(f"Failed after {e.n_attempts} attempts")
    response = None
except ValidationError as e:
    logger.error(f"Validation failed: {e}")
    raise
```

### Use Base Exception for General Handling

```python
import instructor
from pydantic import BaseModel
from instructor.core.exceptions import InstructorError


class Data(BaseModel):
    value: str


client = instructor.from_provider("openai/gpt-4.1-mini")

try:
    response = client.create(
        response_model=Data,
        messages=[{"role": "user", "content": "Extract data"}],
    )
except InstructorError as e:
    # Catches any Instructor-specific error
    print(f"Instructor error: {type(e).__name__}: {e}")
```

### Graceful Degradation

```python
import instructor
from pydantic import BaseModel, field_validator
from instructor.core.exceptions import ValidationError, InstructorRetryException


class StrictData(BaseModel):
    value: int

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Value must be positive")
        return v


class RelaxedData(BaseModel):
    value: str


client = instructor.from_provider("openai/gpt-4.1-mini")


def extract_with_fallback(content: str):
    try:
        return client.create(
            response_model=StrictData,
            messages=[{"role": "user", "content": content}],
        )
    except ValidationError:
        # Fall back to less strict model
        return client.create(
            response_model=RelaxedData,
            messages=[{"role": "user", "content": content}],
        )
    except InstructorRetryException:
        return None
```

## Backwards Compatibility

New exceptions inherit from both `ValueError` and `InstructorError`, so existing code continues to work:

```python
import instructor
from pydantic import BaseModel
from instructor.core.exceptions import ResponseParsingError


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider("openai/gpt-4.1-mini")

# Old code still works
try:
    response = client.create(
        response_model=User,
        messages=[{"role": "user", "content": "Extract: Kai is 41"}],
    )
except ValueError as e:
    print(f"Error: {e}")

# New code can access additional context
try:
    response = client.create(
        response_model=User,
        messages=[{"role": "user", "content": "Extract: Kai is 41"}],
    )
except ResponseParsingError as e:
    print(f"Mode: {e.mode}, Raw: {e.raw_response}")
```

## Integration with Hooks

Monitor errors using the hooks system:

```python
import instructor
from instructor.core.exceptions import ValidationError


def on_parse_error(error: Exception):
    if isinstance(error, ValidationError):
        print(f"Validation error: {error}")


client = instructor.from_provider("openai/gpt-4.1-mini")
client.hooks.on("parse:error", on_parse_error)
```

## See Also

- [Retrying](./retrying.md) - Retry strategies with Tenacity
- [Validation](./validation.md) - Validation patterns
- [Hooks](./hooks.md) - Error monitoring with hooks
