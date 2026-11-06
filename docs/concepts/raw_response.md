---
title: Creating a Model with OpenAI Completions
description: Learn how to create a custom model using OpenAI's API to extract user data efficiently with Python.
---


# Creating a model with completions

In instructor>1.0.0 we have a custom client, if you wish to use the raw response you can do the following

```python
import instructor

from pydantic import BaseModel

client = instructor.from_provider("openai/gpt-4.1-mini")


class UserExtract(BaseModel):
    name: str
    age: int


user, completion = client.create_with_completion(
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old"},
    ],
)

print(user)
#> name='jason' age=25

print(completion)
"""
ChatCompletion(
    id='chatcmpl-D1KqvmcGn5zeYfqRdquwERAH0wIVB',
    choices=[
        Choice(
            finish_reason='stop',
            index=0,
            logprobs=None,
            message=ChatCompletionMessage(
                content=None,
                refusal=None,
                role='assistant',
                annotations=[],
                audio=None,
                function_call=None,
                tool_calls=[
                    ChatCompletionMessageFunctionToolCall(
                        id='call_8VastKJ2gYWNrYEQmBXGWnRv',
                        function=Function(
                            arguments='{"name":"jason","age":25}', name='UserExtract'
                        ),
                        type='function',
                    )
                ],
            ),
        )
    ],
    created=1769210857,
    model='gpt-4.1-mini-2025-04-14',
    object='chat.completion',
    service_tier='default',
    system_fingerprint='fp_376a7ccef1',
    usage=CompletionUsage(
        completion_tokens=10,
        prompt_tokens=79,
        total_tokens=89,
        completion_tokens_details=CompletionTokensDetails(
            accepted_prediction_tokens=None,
            audio_tokens=0,
            reasoning_tokens=0,
            rejected_prediction_tokens=None,
        ),
        prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0),
    ),
)
"""
```

## Raw response with a list response model

If your response model is a list (for example, `list[UserExtract]`), you can still use `create_with_completion()`. Instructor wraps the list in a `ResponseList` (also called `ListResponse`) that behaves like a normal list but also preserves the raw response.

### What is ResponseList?

`ResponseList` is a special list type that Instructor uses when your `response_model` is a list. It extends Python's built-in `list` type and adds a `_raw_response` attribute to store the provider's raw response object.

This is necessary because `create_with_completion()` needs to return both the parsed result and the raw response. For single objects, this is straightforward: `(model_instance, raw_response)`. For lists, we need a way to attach the raw response to the list itself, which is what `ResponseList` does.

### Using ResponseList

The returned value behaves exactly like a normal Python list, but you can access the raw response using `get_raw_response()`:

```python
import instructor
from pydantic import BaseModel

client = instructor.from_provider("openai/gpt-4.1-mini")


class UserExtract(BaseModel):
    name: str
    age: int


users, completion = client.create_with_completion(
    response_model=list[UserExtract],
    messages=[
        {"role": "user", "content": "Extract users: Jason is 25, Ivan is 30"},
    ],
)

# Use it like a normal list
print(users[0])
#> name='Jason' age=25
print(len(users))
#> 2

# Access the raw response
raw = users.get_raw_response()
assert raw == completion

# ResponseList supports all list operations
for user in users:
    print(user.name)
#> Jason
#> Ivan
```

## See Also

- [Hooks](./hooks.md) - Monitor LLM interactions without accessing raw responses
- [Debugging](../debugging.md) - Debugging techniques for LLM outputs
- [Response Models](./models.md) - Working with structured response models

## Anthropic Raw Response

You can also access the raw response from Anthropic models. This is useful for debugging or when you need to access additional information from the response.

```python
import instructor

client = instructor.from_provider("anthropic/claude-3-5-sonnet-latest")


user, completion = client.create_with_completion(
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old"},
    ],
)

print(user)
#> name='Jason' age=25

print(completion)
"""