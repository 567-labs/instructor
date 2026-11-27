---
title: "Gemini structured outputs"
description: "Connect to the Gemini API, validate nested Pydantic models, and migrate from google-generativeai to google-genai."
---

# Gemini API structured outputs with Pydantic in Python

Use `instructor.from_provider("google/gemini-3.8-flash")` and set `GOOGLE_API_KEY` to call the Gemini API. If you authenticate through Google Cloud, follow the [Vertex AI guide](vertex.md).

## Google GenAI SDK

The examples use `google-genai`. For native SDK clients, file uploads, and streaming, see the [Google Gen AI SDK examples](genai.md).

!!! info "Choosing a provider prefix"

    - `google/<model>` is recommended. It uses the current `google-genai` SDK for the Gemini API and, with `vertexai=True`, Vertex AI.
    - `vertexai/<model>` is deprecated. Migrate to `google/<model>` with `vertexai=True`.
    - `gemini/<model>` is legacy. It uses the older `google-generativeai` package; migrate to `google/<model>`.

```bash
pip install "instructor[google-genai]"
```

## Simple User Example (Sync)

```python
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


# Using from_provider (recommended)
client = instructor.from_provider(
    "google/gemini-3.8-flash",
)

resp = client.create(
    response_model=User,
    messages=[
        {
            "role": "user",
            "content": "Extract Jason is 25 years old.",
        }
    ],
)

print(resp)  # User(name='Jason', age=25)
```

## Simple User Example (Async)

!!! info "Async Support"

    Instructor supports async mode for the Google GenAI SDK. If you're using the async client, make sure that your client is declared within the same event loop as the function that calls it. If not you'll get a bunch of errors.

```python
import instructor
from pydantic import BaseModel
import asyncio


class User(BaseModel):
    name: str
    age: int


async def extract_user():
    client = instructor.from_provider(
        "google/gemini-3.8-flash",
        async_client=True,
    )

    user = await client.create(
        messages=[
            {
                "role": "user",
                "content": "Extract Jason is 25 years old.",
            }
        ],
        response_model=User,
    )
    return user


# Run async function
user = asyncio.run(extract_user())
print(user)  # User(name='Jason', age=25)
```

## Configuration Options

You can customize the model's behavior using generation configuration parameters. These parameters control aspects like temperature, token limits, and sampling methods. Pass these parameters as a dictionary to the `generation_config` parameter when creating the response.

The most common parameters include:
- `temperature`: Controls randomness in the output (0.0 to 1.0)
- `max_tokens`: Maximum number of tokens to generate
- `top_p`: Nucleus sampling parameter
- `top_k`: Number of highest probability tokens to consider

For more details on configuration options, see [Google's documentation on Gemini configuration parameters](https://cloud.google.com/vertex-ai/generative-ai/docs/samples/generativeaionvertexai-gemini-pro-config-example){target="_blank"}.


```python
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider(
    "google/gemini-3.8-flash",
    mode=instructor.Mode.JSON,
)

resp = client.create(
    response_model=User,
    messages=[
        {
            "role": "user",
            "content": "Extract Jason is 25 years old.",
        },
    ],
    generation_config={
        "temperature": 0.5,
        "max_tokens": 1000,
        "top_p": 1,
        "top_k": 32,
    },
)

print(resp)
```

## Safety settings with images

Google GenAI uses a different set of harm categories for image inputs (for example, `HARM_CATEGORY_IMAGE_HATE`).

When your request includes image content, Instructor will:

- Use the image-specific categories in the request config
- Map thresholds you pass for text categories (like `HARM_CATEGORY_HATE_SPEECH`) to the matching image category (like `HARM_CATEGORY_IMAGE_HATE`)

This avoids `400 INVALID_ARGUMENT` errors when you combine `safety_settings` with images.

```python
import instructor
from google.genai.types import HarmBlockThreshold, HarmCategory
from instructor.processing.multimodal import Image
from pydantic import BaseModel


class Result(BaseModel):
    summary: str


client = instructor.from_provider("google/gemini-3.8-flash")

result = client.create(
    response_model=Result,
    messages=[
        {
            "role": "user",
            "content": [
                "Describe the image in one sentence.",
                Image.autodetect("path/to/image.png"),
            ],
        }
    ],
    # You can still pass text categories. Instructor will map them for image inputs.
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    },
)

print(result)
```

## Nested Example

```python
import instructor
from pydantic import BaseModel


class Address(BaseModel):
    street: str
    city: str
    country: str


class User(BaseModel):
    name: str
    age: int
    addresses: list[Address]


client = instructor.from_provider(
    "google/gemini-3.8-flash",
)

user = client.create(
    messages=[
        {
            "role": "user",
            "content": """
            Extract: Jason is 25 years old.
            He lives at 123 Main St, New York, USA
            and has a summer house at 456 Beach Rd, Miami, USA
        """,
        },
    ],
    response_model=User,
)

print(user)
#> {
#>     'name': 'Jason',
#>     'age': 25,
#>     'addresses': [
#>         {
#>             'street': '123 Main St',
#>             'city': 'New York',
#>             'country': 'USA'
#>         },
#>         {
#>             'street': '456 Beach Rd',
#>             'city': 'Miami',
#>             'country': 'USA'
#>         }
#>     ]
#> }
```

## Streaming Support

Instructor has two main ways that you can use to stream responses out

1. **Iterables**: These are useful when you'd like to stream a list of objects of the same type (Eg. use structured outputs to extract multiple users)
2. **Partial Streaming**: This is useful when you'd like to stream a single object and you'd like to immediately start processing the response as it comes in.

### Partials

```python
import instructor
from pydantic import BaseModel

client = instructor.from_provider(
    "google/gemini-3.8-flash",
)


class User(BaseModel):
    name: str
    age: int
    bio: str


user = client.create_partial(
    messages=[
        {
            "role": "user",
            "content": "Create a user profile for Jason and 1 sentence bio, age 25",
        },
    ],
    response_model=User,
)

for user_partial in user:
    print(user_partial)
    #> name=None age=None bio=None
    #> name=None age=25 bio='Jason is a great guy'
    #> name='Jason' age=25 bio='Jason is a great guy'
```

### Iterable Example

```python
import instructor
from pydantic import BaseModel

client = instructor.from_provider(
    "google/gemini-3.8-flash",
)


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
    #> name='Sarah' age=30
    #> name='Mike' age=28
```

## Known Limitations

Google Gemini has the following known limitations when used with Instructor:

1. **Union Types**: Gemini does not support Union types (except for `Optional`). Use separate response models or `Literal` types instead.
2. **Union Streaming**: Streaming is not supported for Union types with Iterable.

These limitations are specific to Google Gemini and do not affect other providers like OpenAI or Anthropic. Tests automatically skip these features for Google to prevent failures.

## Instructor Modes

We provide several modes to make it easy to work with the different response models that Gemini supports:

1. `instructor.Mode.TOOLS` : This uses Gemini's tool calling API to return structured outputs (default)
2. `instructor.Mode.JSON` : This uses Gemini's JSON schema mode for structured outputs

!!! note "Backwards Compatibility"

    Legacy provider-specific modes (such as `Mode.GENAI_TOOLS`, `Mode.GENAI_JSON`, `Mode.GENAI_STRUCTURED_OUTPUTS`) are deprecated. They emit warnings and automatically map to the generic modes (`Mode.TOOLS`, `Mode.JSON`).

!!! info "Mode Selection"
    When using `from_provider`, the appropriate mode is automatically selected based on the provider and model capabilities.

## Available Models

Check [Google's model catalog](https://ai.google.dev/gemini-api/docs/models) for supported inputs and model availability.

## Using Gemini's Multimodal Capabilities

We've written an extensive list of guides on how to use gemini's multimodal capabilities with instructor.

- [Using Gemini To Extract Travel Video Recommendations](../blog/posts/multimodal-gemini.md)
- [Parsing PDFs with Gemini](../blog/posts/chat-with-your-pdf-with-gemini.md)
- [Generating Citations with Gemini](../blog/posts/generating-pdf-citations.md)

Stay tuned to the blog for more guides on using Gemini with instructor.

## Related Resources

- [Google AI Documentation](https://ai.google.dev/)
- [Instructor Core Concepts](../concepts/index.md)
- [Type Validation Guide](../concepts/validation.md)
- [Advanced Usage Examples](../examples/index.md)

## Migration from google-generativeai

Install `instructor[google-genai]` and change the provider prefix from `gemini/` to `google/`. If you manage your own SDK client, replace `google.generativeai.GenerativeModel` with `google.genai.Client` and wrap it with [`from_genai`](genai.md).

### Vertex AI Migration

Use `google/<model>` with `vertexai=True`. See [Vertex AI migration](vertex.md#migration-to-google-genai) for project and location settings.
