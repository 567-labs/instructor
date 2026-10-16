---
title: Union Types in Instructor
description: Learn how to use Union types to handle multiple possible response types in Instructor
---

# Working with Union Types in Instructor

This guide explains how to work with union types in Instructor, allowing you to handle multiple possible response types from language models. Union types are particularly useful when you need the LLM to choose between different response formats or action types.

!!! note "Union vs. union"
    The content from the original `union.md` page has been consolidated into this more comprehensive guide. That page showed a basic example of using Union types for multiple action types.

## Basic Union Types

Union types let you specify that a field can be one of several types:

```python
from typing import Union
from pydantic import BaseModel


class Response(BaseModel):
    value: Union[str, int]  # Can be either string or integer
```

## Discriminated Unions

Use discriminated unions to handle different response types:

```python
from typing import Literal, Union
from pydantic import BaseModel
import instructor


class UserQuery(BaseModel):
    type: Literal["user"]
    username: str


class SystemQuery(BaseModel):
    type: Literal["system"]
    command: str


Query = Union[UserQuery, SystemQuery]

# Usage with Instructor
client = instructor.from_provider("openai/gpt-4.1-mini")

response = client.create(
    response_model=Query,
    messages=[{"role": "user", "content": "Parse: user lookup jsmith"}],
)
```

## Optional Fields

Combine Union with Optional for nullable fields:

```python
from typing import Optional
from pydantic import BaseModel


class User(BaseModel):
    name: str
    email: Optional[str] = None  # Same as Union[str, None]
```

## Best Practices

1. **Type Hints**: Use proper type hints for clarity and better IDE support
2. **Discriminators**: Add discriminator fields (like `type`) for complex unions to help the LLM choose correctly
3. **Validation**: Add validators for union fields to ensure the data is valid
4. **Documentation**: Document expected types clearly in your models with docstrings
5. **Field Names**: Use descriptive field names to guide the model's output
6. **Examples**: Include examples in your Pydantic models to help the LLM understand the expected format

## Common Patterns

### Multiple Response Types
```python
from typing import Union, Literal
from pydantic import BaseModel


class SuccessResponse(BaseModel):
    status: Literal["success"]
    data: dict


class ErrorResponse(BaseModel):
    status: Literal["error"]
    message: str


Response = Union[SuccessResponse, ErrorResponse]
```

### Nested Unions
```python
from typing import Literal, Union, List
from pydantic import BaseModel


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class ImageContent(BaseModel):
    type: Literal["image"]
    url: str


class Message(BaseModel):
    content: List[Union[TextContent, ImageContent]]
```

## Dynamic Action Selection with Unions

You can use Union types to write "agents" that dynamically choose actions by selecting an output class. For example, in a search and lookup function:

```python
from pydantic import BaseModel
from typing import Union


class Search(BaseModel):
    query: str

    def execute(self):
        # Implementation for search
        return f"Searching for: {self.query}"


class Lookup(BaseModel):
    key: str

    def execute(self):
        # Implementation for lookup
        return f"Looking up key: {self.key}"


class Action(BaseModel):
    action: Union[Search, Lookup]

    def execute(self):
        return self.action.execute()
```

With this pattern, the LLM can decide whether to perform a search or a lookup based on the user's input:

```python
import instructor
from pydantic import BaseModel
from typing import Union


class Search(BaseModel):
    query: str

    def execute(self):
        # Implementation for search
        return f"Searching for: {self.query}"


class Lookup(BaseModel):
    key: str

    def execute(self):
        # Implementation for lookup
        return f"Looking up key: {self.key}"


class Action(BaseModel):
    action: Union[Search, Lookup]

    def execute(self):
        return self.action.execute()


client = instructor.from_provider("openai/gpt-4.1-mini")

# Let the LLM decide what action to take
result = client.create(
    response_model=Action,
    messages=[
        {
            "role": "system",
            "content": "You're an assistant that helps search or lookup information.",
        },
        {"role": "user", "content": "Find information about climate change"},
    ],
)

# Execute the chosen action
print(result.execute())  # Likely outputs: "Searching for: climate change"
#> Searching for: climate change
```

## Integration with Instructor

```python
import instructor
from typing import Union, Literal
from pydantic import BaseModel


class SuccessResponse(BaseModel):
    status: Literal["success"]
    data: dict


class ErrorResponse(BaseModel):
    status: Literal["error"]
    message: str


Response = Union[SuccessResponse, ErrorResponse]

client = instructor.from_provider("openai/gpt-4.1-mini")

result = client.create(
    response_model=Response,
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that processes requests and returns either a success or error response.",
        },
        {
            "role": "user",
            "content": "Process this request: Get user information for id 123",
        },
    ],
)

# Check the result type
if isinstance(result, ErrorResponse):
    print(f"Error: {result.message}")
    #> Error: Request not supported: Get user information for id 123
else:
    print(f"Success: {result.data}")
```

### Streaming with Unions
```python
import instructor


def stream_content():
    response = client.create(
        response_model=instructor.Partial[Message],
        stream=True,
        messages=[{"role": "user", "content": "Generate mixed content"}],
    )
    for partial in response:
        if partial.content:
            for item in partial.content:
                if isinstance(item, TextContent):
                    print(f"Text: {item.text}")
                elif isinstance(item, ImageContent):
                    print(f"Image: {item.url}")
```

### Validation Errors

```python
from pydantic import ValidationError, BaseModel
from typing import Union, Literal


class SuccessResponse(BaseModel):
    status: Literal["success"]
    data: dict


class ErrorResponse(BaseModel):
    status: Literal["error"]
    message: str


Response = Union[SuccessResponse, ErrorResponse]

try:
    # This will fail because "invalid" is not a valid status
    response = SuccessResponse(status="invalid", data={"key": "value"})
except ValidationError as e:
    print(f"Validation error: {e}")
    """
    Validation error: 1 validation error for SuccessResponse
    status
      Input should be 'success' [type=literal_error, input_value='invalid', input_type=str]
    """
```

## Type Checking

Use isinstance() for runtime type checking:

```python
from typing import Union, Literal
from pydantic import BaseModel


class SuccessResponse(BaseModel):
    status: Literal["success"]
    data: dict


class ErrorResponse(BaseModel):
    status: Literal["error"]
    message: str


Response = Union[SuccessResponse, ErrorResponse]


def process_response(response: Response):
    if isinstance(response, SuccessResponse):
        # Handle success case
        print(f"Success: {response.data}")
    elif isinstance(response, ErrorResponse):
        # Handle error case
        print(f"Error: {response.message}")
```

For more information about union types, check out the [Pydantic documentation on unions](https://docs.pydantic.dev/latest/concepts/types/#unions).

```python
from typing import Literal, Union
from pydantic import BaseModel
import instructor
from openai import OpenAI


class Action(BaseModel):
    """Base action class."""

    type: str


class SendMessage(BaseModel):
    type: Literal["send_message"]
    message: str
    recipient: str


class MakePayment(BaseModel):
    type: Literal["make_payment"]
    amount: float
    recipient: str


Action = Union[SendMessage, MakePayment]

# Usage with Instructor
client = instructor.from_provider("openai/gpt-4o")
response = client.create(
    response_model=Action,
    messages=[{"role": "user", "content": "Send a payment of $50 to John."}],
)
```

```python
from typing import Literal, Union
from pydantic import BaseModel
import instructor
from openai import OpenAI


class SearchAction(BaseModel):
    type: Literal["search"]
    query: str


class EmailAction(BaseModel):
    type: Literal["email"]
    to: str
    subject: str
    body: str


Action = Union[SearchAction, EmailAction]

# The model can choose which action to take
client = instructor.from_provider("openai/gpt-4o")
response = client.create(
    response_model=Action,
    messages=[{"role": "user", "content": "Find me information about climate change."}],
)
```

```python
from typing import Literal, Union
from pydantic import BaseModel
import instructor
from openai import OpenAI


class TextResponse(BaseModel):
    type: Literal["text"]
    content: str


class ImageResponse(BaseModel):
    type: Literal["image"]
    url: str
    caption: str


Response = Union[TextResponse, ImageResponse]

# Patched client
client = instructor.from_provider("openai/gpt-4o")
response = client.create(
    response_model=Response,
    messages=[{"role": "user", "content": "Tell me a joke about programming."}],
)
```

```python
from typing import Union
from pydantic import BaseModel


class Response(BaseModel):
    """A more complex example showing nested Union fields."""

    result: Union[str, int, float, bool]
```

```python
from typing import Dict, List, Union, Any
from pydantic import BaseModel


class Response(BaseModel):
    """A more complex example showing nested Union fields."""

    data: Dict[str, Union[str, int, List[Any]]]
```

## See Also

- [Types](./types.md) - Working with different data types in Instructor
- [Enums](./enums.md) - Using enumerated types for structured choices
- [Optional Fields](../learning/patterns/optional_fields.md) - Handling optional data
- [Validation](./validation.md) - Validating union type responses
- [Union Examples](../examples/index.md) - Practical union type examples
