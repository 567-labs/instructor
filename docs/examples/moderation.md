---
title: OpenAI Moderation Example for Content Compliance
description: Learn how to use OpenAI's moderation endpoint to filter harmful content and ensure compliance with usage policies.
---

# OpenAI Moderation

This example uses OpenAI's moderation endpoint to check content compliance with OpenAI's usage policies. It can identify and filter harmful content that violates the policies.

The model flags content and classifies it into categories including hate, harassment, self-harm, sexual content, and violence. Each category has subcategories for detailed classification.

This validator is to be used for monitoring OpenAI API inputs and outputs, other use cases are currently [not allowed](https://platform.openai.com/docs/guides/moderation).

## Incorporating OpenAI moderation validator

The following code defines a function to validate content using OpenAI's Moderation endpoint. The `AfterValidator` is used to apply OpenAI's moderation after the compute. This moderation checks if the content complies with OpenAI's usage policies and flags any harmful content. Here's how it works:

1. Generate the OpenAI client and patch it with the `instructor`. Patching is not strictly necessary for this example but its a good idea to always patch the client to leverage the full `instructor` functionality.

2. Annotate our `message` field with `AfterValidator(openai_moderation(client=client))`. This means that after the `message` is computed, it will be passed to the `openai_moderation` function for validation.

```python
import instructor

from instructor import openai_moderation

from typing_extensions import Annotated
from pydantic import BaseModel, AfterValidator

client = instructor.from_provider("openai/gpt-5-nano")


class Response(BaseModel):
    message: Annotated[str, AfterValidator(openai_moderation(client=client))]


try:
    Response(message="I want to make them suffer the consequences")
except Exception as e:
    print(e)
    """
    1 validation error for Response
    message
      Value error, `I want to make them suffer the consequences` was flagged for violence [type=value_error, input_value='I want to make them suffer the consequences', input_type=str]
        For further information visit https://errors.pydantic.dev/2.9/v/value_error
    """

try:
    Response(message="I want to hurt myself.")
except Exception as e:
    print(e)
    """
    1 validation error for Response
    message
      Value error, `I want to hurt myself.` was flagged for self_harm, self_harm_intent, self-harm, self-harm/intent [type=value_error, input_value='I want to hurt myself.', input_type=str]
        For further information visit https://errors.pydantic.dev/2.9/v/value_error
    """
```
