---
title: "Retry Logic with Tenacity"
description: "Learn how to implement retry logic with Tenacity for LLM applications, including exponential backoff, conditional retries, and error handling."
---

# Retry Logic with Tenacity

Tenacity is a Python library for adding retry logic to your applications. Combined with Instructor, it helps handle API failures, rate limits, and validation errors.

## Limit Validation Retry Cost

Use `token_budget` to stop validation retries after cumulative provider usage
reaches a positive token limit:

```python
import instructor
from instructor.core import TokenBudgetExceeded
from pydantic import BaseModel

client = instructor.from_provider("openai/gpt-4.1-mini")


class UserInfo(BaseModel):
    name: str
    age: int


try:
    user = client.create(
        response_model=UserInfo,
        messages=[{"role": "user", "content": "Extract: Jason is 25"}],
        max_retries=3,
        token_budget=2_000,
    )
except TokenBudgetExceeded as error:
    print(error.total_usage)
```

The budget is checked after a response fails validation and before Instructor
prepares another request. Reaching the exact budget stops the retry. A response
that validates successfully is returned even if that completed request takes
the cumulative total over the budget.

`token_budget` is a retry budget, not a hard per-request limit. The provider may
use more than the remaining budget while completing the current request. Use
the provider's output-token setting when you also need a per-request limit.

Budgeted retries currently require a structured, non-streaming response and
compatible provider usage metadata. Instructor raises
`TokenUsageUnavailableError` instead of making another request when it cannot
account for usage safely.

## Basic Retry with Exponential Backoff

The most common pattern uses exponential backoff to delay retries:

```python
import instructor
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

client = instructor.from_provider("openai/gpt-4.1-mini")


class UserInfo(BaseModel):
    name: str
    age: int
    email: str


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def extract_user_info(text: str) -> UserInfo:
    """Extract user information with retry logic."""
    return client.create(
        response_model=UserInfo,
        messages=[{"role": "user", "content": f"Extract user info: {text}"}],
    )


try:
    user = extract_user_info("John is 30 years old with email john@example.com")
    print(f"Success: {user.name}, {user.age}, {user.email}")
    #> Success: John, 30, john@example.com
except Exception as e:
    print(f"Failed after retries: {e}")
```

## Error-Specific Retries

Retry only on specific error types for better control:

```python
import instructor
from openai import APIError, RateLimitError
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

client = instructor.from_provider("openai/gpt-4.1-mini")


class UserInfo(BaseModel):
    name: str
    age: int
    email: str


# Retry on API errors with longer delays
@retry(
    retry=retry_if_exception_type((RateLimitError, APIError)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=1, max=60),
)
def handle_api_errors(text: str) -> UserInfo:
    return client.create(
        response_model=UserInfo,
        messages=[{"role": "user", "content": text}],
    )


# Retry on validation errors with shorter delays
@retry(
    retry=retry_if_exception_type(ValidationError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
)
def handle_validation_errors(text: str) -> UserInfo:
    return client.create(
        response_model=UserInfo,
        messages=[{"role": "user", "content": text}],
    )
```

## Custom Retry Conditions

Retry based on the result content rather than exceptions:

```python
import instructor
from pydantic import BaseModel
from tenacity import retry, retry_if_result, stop_after_attempt

client = instructor.from_provider("openai/gpt-4.1-mini")


class UserInfo(BaseModel):
    name: str
    age: int
    email: str


def should_retry(result: UserInfo) -> bool:
    """Retry if the result doesn't meet quality criteria."""
    return result.age < 0 or result.age > 150 or not result.email


@retry(retry=retry_if_result(should_retry), stop=stop_after_attempt(3))
def extract_valid_user(text: str) -> UserInfo:
    return client.create(
        response_model=UserInfo,
        messages=[{"role": "user", "content": text}],
    )
```

## Context-Based Validation with Retries

Use the `context` parameter to pass runtime data to validators:

```python
import instructor
from pydantic import BaseModel, ValidationInfo, field_validator, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

client = instructor.from_provider("openai/gpt-4.1-mini")


class Citation(BaseModel):
    """A claim with a supporting quote from source text."""

    claim: str
    quote: str

    @field_validator('quote')
    @classmethod
    def verify_quote_exists(cls, v: str, info: ValidationInfo):
        context = info.context
        if context:
            source_text = context.get('source_text', '')
            if v not in source_text:
                raise ValueError(f"Quote '{v}' not found in source text.")
        return v


@retry(
    retry=retry_if_exception_type(ValidationError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def extract_citation(claim: str, source_text: str) -> Citation:
    return client.create(
        response_model=Citation,
        messages=[
            {
                "role": "system",
                "content": "Extract the claim and find an exact quote from the source.",
            },
            {
                "role": "user",
                "content": "Source: {{ source_text }}\n\nClaim: {{ claim }}",
            },
        ],
        context={"source_text": source_text, "claim": claim},
    )


source = "The Eiffel Tower was completed in 1889 and stands 330 meters tall."
citation = extract_citation("The tower is over 300 meters", source)
print(f"Quote: {citation.quote}")
```

## Logging and Monitoring

Add logging to track retry attempts:

```python
import logging
import instructor
from pydantic import BaseModel
from tenacity import after_log, before_log, retry, stop_after_attempt, wait_exponential

client = instructor.from_provider("openai/gpt-4.1-mini")


class UserInfo(BaseModel):
    name: str
    age: int
    email: str


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    before=before_log(logger, logging.INFO),
    after=after_log(logger, logging.ERROR),
)
def logged_extraction(text: str) -> UserInfo:
    return client.create(
        response_model=UserInfo,
        messages=[{"role": "user", "content": text}],
    )
```

## Instructor's Built-in Retries

Instructor has built-in retry support that works alongside Tenacity:

```python
import instructor
from instructor import Mode
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt

client = instructor.from_provider(
    "openai/gpt-4.1-mini",
    mode=Mode.JSON,
    max_retries=3,
    retry_delay=1,
)


class UserInfo(BaseModel):
    name: str
    age: int
    email: str


# Combine Instructor and Tenacity retries for additional resilience
@retry(stop=stop_after_attempt(2))
def double_retry_extraction(text: str) -> UserInfo:
    return client.create(
        response_model=UserInfo,
        messages=[{"role": "user", "content": text}],
    )
```

## Failed Attempts Tracking

When retries fail, Instructor provides detailed failure history:

```python
import instructor
from instructor.core.exceptions import InstructorRetryException
from pydantic import BaseModel, field_validator

client = instructor.from_provider("openai/gpt-4.1-mini")


class UserInfo(BaseModel):
    name: str
    age: int

    @field_validator('age')
    @classmethod
    def validate_age(cls, v):
        if v < 0 or v > 150:
            raise ValueError(f"Age {v} is invalid")
        return v


try:
    result = client.create(
        response_model=UserInfo,
        messages=[{"role": "user", "content": "Extract: John is -5 years old"}],
        max_retries=3,
    )
except InstructorRetryException as e:
    print(f"Failed after {e.n_attempts} attempts")
    for attempt in e.failed_attempts:
        print(f"Attempt {attempt.attempt_number}: {attempt.exception}")
```

Failed attempts are automatically propagated to reask handlers, enabling contextual error messages and progressive corrections.

## Best Practices

### Choose Appropriate Strategies

| Error Type | Attempts | Min Delay | Max Delay |
|------------|----------|-----------|-----------|
| Rate limits | 5 | 1s | 60-120s |
| Validation errors | 2-3 | 1s | 10s |
| Network errors | 4 | 2s | 30s |

### Always Set Stop Conditions

```python
from tenacity import retry, stop_after_attempt


# Good: bounded retries
@retry(stop=stop_after_attempt(3))
def bounded_retry():
    pass


# Bad: could retry forever
@retry()  # Don't do this!
def unbounded_retry():
    pass
```

## Troubleshooting

**Infinite retries**: Always set `stop_after_attempt()` or `stop_after_delay()`.

**Too many retries**: Use `retry_if_exception_type()` to retry only on specific errors.

**Still hitting rate limits**: Increase max delay and use `wait_exponential()` with higher multipliers.

## Related Resources

- [Tenacity Documentation](https://tenacity.readthedocs.io/)
- [Error Handling](./error_handling.md)
- [Validation](./validation.md)
