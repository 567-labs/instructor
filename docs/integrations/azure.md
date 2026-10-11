---
title: Structured outputs with Azure OpenAI, a complete guide w/ instructor
description: Learn how to use Azure OpenAI with instructor for structured outputs, including async/sync implementations, streaming, and validation.
---

# Structured Outputs with Azure OpenAI

This guide demonstrates how to use Azure OpenAI with instructor for structured outputs. Azure OpenAI provides the same powerful models as OpenAI but with enterprise-grade security and compliance features through Microsoft Azure.

## Installation

We can use the same installation as we do for OpenAI since the default `openai` client ships with an AzureOpenAI client.

First, install the required dependencies:

```bash
pip install instructor
```

Next, make sure that you've enabled Azure OpenAI in your Azure account and have a deployment for the model you'd like to use. [Here is a guide to get started](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/create-resource?pivots=web-portal)

Once you've done so, you'll have an endpoint and a API key to be used to configure the client.

```bash
instructor.exceptions.InstructorRetryException: Error code: 401 - {'statusCode': 401, 'message': 'Unauthorized. Access token is missing, invalid, audience is incorrect (https://cognitiveservices.azure.com), or have expired.'}
```

If you see an error like the one above, make sure you've set the correct endpoint and API key in the client.

## Authentication

To use Azure OpenAI, you'll need:

1. Azure OpenAI endpoint
2. API key
3. Deployment name

```python
import os
from openai import AzureOpenAI
import instructor

# Configure Azure OpenAI client
client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-02-01",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"]
)

# Patch the client with instructor
client = instructor.from_provider("azure_openai/gpt-4o-mini")
```

## Using Auto Client (Recommended)

The easiest way to get started with Azure OpenAI is using the `from_provider` method:

```python
import instructor
import os

# Set your Azure OpenAI credentials
os.environ["AZURE_OPENAI_API_KEY"] = "your-api-key"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-resource.openai.azure.com/"

# Create client using the provider string
client = instructor.from_provider("azure_openai/gpt-4o-mini")

# Or async client
async_client = instructor.from_provider("azure_openai/gpt-4o-mini", async_client=True)
```

You can also pass credentials as parameters:

```python
import instructor

client = instructor.from_provider(
    "azure_openai/gpt-4o-mini",
    api_key="your-api-key",
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_version="2024-02-01"  # Optional, defaults to 2024-02-01
)
```

## Basic Usage

Here's a simple example using a Pydantic model:

```python
import os
import instructor
from openai import AzureOpenAI
from pydantic import BaseModel

client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-02-01",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
)
client = instructor.from_provider("azure_openai/gpt-4o-mini")


class User(BaseModel):
    name: str
    age: int


# Synchronous usage
user = client.create(
    messages=[{"role": "user", "content": "John is 30 years old"}],
    response_model=User,
)

print(user)
# > name='John' age=30
```

## Async Implementation

Azure OpenAI supports async operations:

```python
import os
import instructor
import asyncio
from openai import AsyncAzureOpenAI
from pydantic import BaseModel

client = AsyncAzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-02-15-preview",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
)
client = instructor.from_provider("azure_openai/gpt-4o-mini")


class User(BaseModel):
    name: str
    age: int


async def get_user_async():
    return await client.create(
        messages=[{"role": "user", "content": "John is 30 years old"}],
        response_model=User,
    )


# Run async function
user = asyncio.run(get_user_async())
print(user)
# > name='John' age=30
```

## Nested Models

Azure OpenAI handles complex nested structures:

```python
import os
import instructor
from openai import AzureOpenAI
from pydantic import BaseModel

client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-02-01",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
)
client = instructor.from_provider("azure_openai/gpt-4o-mini")


class Address(BaseModel):
    street: str
    city: str
    country: str


class UserWithAddress(BaseModel):
    name: str
    age: int
    addresses: list[Address]


resp = client.create(
    messages=[
        {
            "role": "user",
            "content": """
        John is 30 years old and has two addresses:
        1. 123 Main St, New York, USA
        2. 456 High St, London, UK
        """,
        }
    ],
    response_model=UserWithAddress,
)

print(resp)
# {
#     'name': 'John',
#     'age': 30,
#     'addresses': [
#         {
#             'street': '123 Main St',
#             'city': 'New York',
#             'country': 'USA'
#         },
#         {
#             'street': '456 High St',
#             'city': 'London',
#             'country': 'UK'
#         }
#     ]
# }
```

## Streaming Support

Instructor has two main ways that you can use to stream responses out

1. **Iterables**: These are useful when you'd like to stream a list of objects of the same type (Eg. use structured outputs to extract multiple users)
2. **Partial Streaming**: This is useful when you'd like to stream a single object and you'd like to immediately start processing the response as it comes in.

### Partials

You can use our `create_partial` method to stream a single object. Note that validators should not be declared in the response model when streaming objects because it will break the streaming process.

```python
import instructor
from pydantic import BaseModel

client = instructor.from_provider("azure_openai/gpt-4o-mini")


class User(BaseModel):
    name: str
    age: int
    bio: str


# Stream partial objects as they're generated
user = client.create_partial(
    messages=[
        {"role": "user", "content": "Create a user profile for Jason, age 25"},
    ],
    response_model=User,
)

for user_partial in user:
    print(user_partial)

# > name='Jason' age=None bio='None'
# > name='Jason' age=25 bio='A tech'
# > name='Jason' age=25 bio='A tech enthusiast'
# > name='Jason' age=25 bio='A tech enthusiast who loves coding, gaming, and exploring new'
# > name='Jason' age=25 bio='A tech enthusiast who loves coding, gaming, and exploring new technologies'

```

## Iterable Responses

```python
import instructor
from pydantic import BaseModel

client = instructor.from_provider("azure_openai/gpt-4o-mini")


class User(BaseModel):
    name: str
    age: int


# Extract multiple users from text
users = client.create_iterable(
    messages=[
        {
            "role": "user",
            "content": """
            Extract users:
            1. Jason is 25 years old
            2. Sarah is 30 years old
            3. Mike is 28 years old
        """,
        },
    ],
    response_model=User,
)

for user in users:
    print(user)
#> name='Jason' age=25
# > name='Sarah' age=30
# > name='Mike' age=28

```

## Instructor Modes

We provide several modes to make it easy to work with the different response models that OpenAI supports

1. `instructor.Mode.TOOLS` : This uses the [tool calling API](https://platform.openai.com/docs/guides/function-calling) to return structured outputs to the client
2. `instructor.Mode.JSON` : This forces the model to return JSON by using [OpenAI's JSON mode](https://platform.openai.com/docs/guides/structured-outputs#json-mode).
3. `instructor.Mode.FUNCTIONS` : This uses OpenAI's function calling API to return structured outputs and will be deprecated in the future.
4. `instructor.Mode.PARALLEL_TOOLS` : This uses the [parallel tool calling API](https://platform.openai.com/docs/guides/function-calling#configuring-parallel-function-calling) to return structured outputs to the client. This allows the model to generate multiple calls in a single response.
5. `instructor.Mode.MD_JSON` : This makes a simple call to the OpenAI chat completion API and parses the raw response as JSON.
6. `instructor.Mode.TOOLS_STRICT` : This uses the new Open AI structured outputs API to return structured outputs to the client using constrained grammar sampling. This restricts users to a subset of the JSON schema.
7. `instructor.Mode.JSON_O1` : This is a mode for the `O1` model. We created a new mode because `O1` doesn't support any system messages, tool calling or streaming so you need to use this mode to use Instructor with `O1`.

In general, we recommend using `Mode.Tools` because it's the most flexible and future-proof mode. It has the largest set of features that you can specify your schema in and makes things significantly easier to work with.

## Best Practices

## Additional Resources

- [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Instructor Documentation](https://python.useinstructor.com/)
- [Azure OpenAI Pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/)
