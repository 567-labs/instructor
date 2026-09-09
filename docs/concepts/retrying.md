---
title: "Retry Logic with Tenacity"
description: "Learn how to implement retry logic with Tenacity for LLM applications, including exponential backoff, conditional retries, and error handling."
---

# Retry Logic with Tenacity

Tenacity is a Python library for adding retry logic to your applications. Combined with Instructor, it helps handle API failures, rate limits, and validation errors.

## Count SDK Requests Separately from Extraction Attempts

There are two retry settings with the same name:

- `OpenAI(max_retries=2)` or `Anthropic(max_retries=2)` configures SDK transport
  retries after the first HTTP request.
- `client.create(max_retries=2, ...)` configures Instructor validation retries
  after the first extraction attempt: at most three SDK calls. Negative integer
  values are treated as zero retries.

With integer `max_retries`, Instructor retries supported parsing and validation
errors. An exhausted SDK rate-limit, connection, or server error is wrapped in
`InstructorRetryException`; Instructor does not restart it automatically. A
custom `Retrying` / `AsyncRetrying` instance controls its own retry predicate,
wait, and stop conditions, and can retry SDK errors as well.

Do not multiply the configured limits to report actual calls. For example,
`429 → invalid JSON response → 500 → valid JSON response` takes four HTTP
requests and two Instructor attempts. Persistent `429` with two SDK retries
ends after three HTTP requests and one Instructor attempt. The product of the
limits is only an upper bound for these SDKs under an attempt-bounded policy;
SDK retry decisions and successful responses determine the actual count.

`completion:kwargs`, `completion:response`, and exception `n_attempts` describe
Instructor attempts. SDK-internal retries do not emit additional Instructor
hooks. `failed_attempts` records parsing failures, not every transport failure.
Use SDK transport instrumentation or server request logs for HTTP attempt
counts, and elapsed time around the entire extraction for latency. A missing
measurement must remain unknown rather than being reported as zero.

For custom policies, `max_attempts` is unknown (`None`), and an error hook's
`is_last_attempt=False` means finality has not yet been established. Use
`completion:last_attempt` to observe exhausted retries. This distinction also
matters when elapsed-time stopping ends retries before the integer count limit.
Cancellation propagates without an error or last-attempt hook.

The [OpenAI SDK retry documentation](https://github.com/openai/openai-python#retries)
and [Anthropic SDK documentation](https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python)
describe their own retry and timeout behavior. Both document two default retries
for selected transport/status failures; consult the installed SDK version before
extending these rules to another provider or transport.

## Timeouts and Whole-Operation Deadlines

For integer Instructor retries, a numeric `timeout` passed to `create` serves two
purposes: it is forwarded to the SDK, and it supplies an elapsed-time stop
condition for further validation retries. The elapsed condition is evaluated
after a failed attempt. It does not interrupt an in-flight SDK request, SDK
backoff, parsing, or user validation. A successful response can return after that
elapsed limit. SDK socket timeouts may themselves trigger SDK retries.

An SDK timeout object, such as `httpx.Timeout` for OpenAI 2.x, is forwarded but
does not enable Instructor's numeric elapsed stop condition. A timeout configured
only on the SDK client likewise does not set that condition. With a custom
Tenacity instance, specify its stop policy explicitly; Instructor does not add
an elapsed stop condition to it. `stop_after_delay` also checks between attempts
and is not an interrupting deadline.

For an async operation, the application can bound the wait using cancellation:

```python
import asyncio
import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel


class Answer(BaseModel):
    value: int


async def extract():
    async with AsyncOpenAI(max_retries=0, timeout=2.0) as sdk:
        client = instructor.from_openai(sdk)
        return await asyncio.wait_for(
            client.create(
                model="gpt-4.1-mini",
                response_model=Answer,
                messages=[{"role": "user", "content": "Extract seven"}],
                max_retries=1,
            ),
            timeout=5.0,
        )
```

`asyncio.wait_for` cancels the task and waits for cancellation cleanup, so cleanup
or blocking synchronous validators can extend the observed wall time. It does
not guarantee that the remote provider stops work. Cancelling a thread that runs
a synchronous client does not cancel that client's HTTP request. For lazy
streaming, place the stream's consumption and cleanup inside the application's
deadline too; returning a stream does not mean extraction has completed.

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

Pass the policy into Instructor to select the original SDK and validation
exceptions before Instructor wraps exhaustion in `InstructorRetryException`:

```python
import instructor
from openai import APIConnectionError, InternalServerError, OpenAI, RateLimitError
from pydantic import BaseModel, ValidationError
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

client = instructor.from_openai(OpenAI(max_retries=0))


class UserInfo(BaseModel):
    name: str
    age: int


user = client.create(
    model="gpt-4.1-mini",
    response_model=UserInfo,
    messages=[{"role": "user", "content": "John is 30"}],
    max_retries=Retrying(
        retry=retry_if_exception_type(
            (APIConnectionError, InternalServerError, RateLimitError, ValidationError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
    ),
)
```

Use `AsyncRetrying` with async clients and the relevant exception classes for
your SDK (for example, `anthropic.RateLimitError`). This explicit policy makes
up to three Instructor attempts and disables nested OpenAI transport retries.
An outer decorator sees `InstructorRetryException`, not its underlying SDK or
validation error, and restarting the function starts a new extraction budget.

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
    Retrying,
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
        max_retries=Retrying(
            retry=retry_if_exception_type(ValidationError),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
        ),
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

Set validation retry limits on the extraction call:

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel

client = instructor.from_openai(OpenAI(max_retries=2), mode=instructor.Mode.JSON)


class UserInfo(BaseModel):
    name: str
    age: int


user = client.create(
    model="gpt-4.1-mini",
    response_model=UserInfo,
    messages=[{"role": "user", "content": "John is 30"}],
    max_retries=1,  # First extraction plus at most one validation retry.
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
