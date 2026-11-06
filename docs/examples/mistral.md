---
title: Using MistralAI for Structured Outputs
description: Learn how to use MistralAI models for inference, including setup, API key generation, and example code.
---

# Structured Outputs using Mistral
You can use MistralAI models for inference with Instructor using `from_provider`.

The examples use `mistral-large-latest`.

## MistralAI API
To use mistral you need to obtain a mistral API key.
Goto [mistralai](https://mistral.ai/) click on Build Now and login. Select API Keys from the left menu and then select
Create API key to create a new key.

## Use example
Some pip packages need to be installed to use the example:
```
pip install instructor mistralai pydantic
```
You need to export the mistral API key:
```
export MISTRAL_API_KEY=<your-api-key>
```

An example:
```python
import instructor
from pydantic import BaseModel


class UserDetails(BaseModel):
    name: str
    age: int


# Using from_provider (recommended)
client = instructor.from_provider("mistral/mistral-large-latest")

resp = client.create(
    response_model=UserDetails,
    messages=[{"role": "user", "content": "Jason is 10"}],
    temperature=0,
)

print(resp)
#> name='Jason' age=10

# output: UserDetails(name='Jason', age=10)
```
