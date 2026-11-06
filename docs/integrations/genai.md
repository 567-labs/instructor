---
draft: False
date: 2025-03-15
title: "Structured outputs with Google's genai SDK"
description: "Learn how to use Instructor with Google's Generative AI SDK to extract structured data from Gemini models."
slug: genai
tags:
  - patching
authors:
  - instructor
---

# Structured Outputs with Google's genai SDK

!!! info "Recommended SDK"

    The `genai` SDK is Google's recommended Python client for working with Gemini models. It provides a unified interface for both the Gemini API and Vertex AI. For detailed setup instructions, including how to use it with Vertex AI, please refer to the [official Google AI documentation for the GenAI SDK](https://googleapis.github.io/python-genai/).

This guide demonstrates how to use Instructor with Google's `genai` SDK to extract structured data from Gemini models.

We currently have two modes for Gemini

- `Mode.TOOLS` : This leverages function calling under the hood and returns a structured response
- `Mode.JSON` : This provides Gemini with a JSON Schema that it will use to respond in a structured format with

!!! info "Gemini Thought Parts Filtering"

    When using `Mode.TOOLS`, Instructor automatically filters out thought parts from Gemini responses. Gemini 2.5 models include internal reasoning parts with `thought: true` by default, which cannot be disabled. Instructor removes these thought parts before processing the structured output to prevent runtime errors.

    This filtering happens automatically and requires no additional configuration. For more information about Gemini's thinking feature, see the [official documentation](https://ai.google.dev/gemini-api/docs/thinking).

!!! note "Backwards Compatibility"

    The legacy provider-specific modes (`Mode.GENAI_TOOLS`, `Mode.GENAI_JSON`, `Mode.GENAI_STRUCTURED_OUTPUTS`) are still supported but emit deprecation warnings and automatically map to the generic modes (`Mode.TOOLS`, `Mode.JSON`).

## Installation

```bash
pip install "instructor[google-genai]"
```

## Basic Usage

!!! warning "Union Types"

    Gemini does not support Union types (except for `Optional`) in structured outputs and tool calling. Use separate response models or `Literal` types instead. `Optional` fields (i.e., `X | None`) work correctly.

Getting started with Instructor and the genai SDK is straightforward. Just create a Pydantic model defining your output structure, patch the genai client, and make your request with a response_model parameter:

```python
from google import genai
import instructor
from pydantic import BaseModel

# Define your Pydantic model
class User(BaseModel):
    name: str
    age: int

# Initialize and patch the client
client = instructor.from_provider("google/gemini-2.5-flash")

# Extract structured data
response = client.create(
    messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
    response_model=User,
)

print(response)  # User(name='Jason', age=25)
```

## Alternative: Using the v2 GenAI client

!!! note "Recommended: Use `from_provider`"

    The `from_provider` approach shown above is recommended for most use cases. The `from_genai` helper below is available if you need to work directly with the native `google.genai.Client` and keep the Google request format intact.

```python
from google.genai import Client
from instructor import Mode
from instructor.v2 import from_genai
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


raw_client = Client(api_key="YOUR_KEY")
client = from_genai(raw_client, mode=Mode.TOOLS)

result = client.chat.completions.create(
    messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
    response_model=User,
)

print(result)
```

Behind the scenes the v2 client registers the correct mode handler, converts OpenAI-style messages to the GenAI `contents` format, and parses the response while filtering Gemini thought parts.

## Message Formatting

Genai supports multiple message formats, and Instructor seamlessly works with all of them. This flexibility allows you to use whichever format is most convenient for your application:

```python
from google import genai
import instructor
from pydantic import BaseModel
from google.genai import types

# Define your Pydantic model
class User(BaseModel):
    name: str
    age: int

# Initialize and patch the client
client = instructor.from_provider("google/gemini-2.5-flash")

# Single string (converted to user message)
response = client.create(
    messages="Jason is 25 years old",
    response_model=User,
)

print(response)
# > name='Jason' age=25

# Standard format
response = client.create(
    messages=[
        {"role": "user", "content": "Jason is 25 years old"}
    ],
    response_model=User,
)

print(response)
# > name='Jason' age=25

# Using genai's Content type
response = client.create(
    messages=[
        genai.types.Content(
            role="user",
            parts=[genai.types.Part.from_text(text="Jason is 25 years old")]
        )
    ],
    response_model=User,
)

print(response)
# > name='Jason' age=25
```

### System Messages

System messages help set context and instructions for the model. With Gemini models, you can provide system messages in two different ways:

```python
from google import genai
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider("google/gemini-2.5-flash")

# As a parameter
response = client.create(
    system="Jason is 25 years old",
    messages=[{"role": "user", "content": "You are a data extraction assistant"}],
    response_model=User,
)

print(response)
# > name='Jason' age=25

# Or as a message with role "system"
response = client.create(
    messages=[
        {"role": "system", "content": "Jason is 25 years old"},
        {"role": "user", "content": "You are a data extraction assistant"},
    ],
    response_model=User,
)

print(response)
# > name='Jason' age=25

```

## Template Variables

Template variables make it easy to reuse prompts with different values. This is particularly useful for dynamic content or when testing different inputs:

```python
from google import genai
import instructor
from pydantic import BaseModel
from google.genai import types


# Define your Pydantic model
class User(BaseModel):
    name: str
    age: int


# Initialize and patch the client
client = instructor.from_provider("google/gemini-2.5-flash")

# Single string (converted to user message)
response = client.create(
    messages=[{"role": "user", "content": "{{ name }} is {{ age }} years old"}],
    response_model=User,
    context={
        "name": "Jason",
        "age": 25,
    },
)

print(response)
# > name='Jason' age=25

# Standard format
response = client.create(
    messages=[{"role": "user", "content": "{{ name }} is {{ age }} years old"}],
    response_model=User,
    context={
        "name": "Jason",
        "age": 25,
    },
)

print(response)
# > name='Jason' age=25

# Using genai's Content type
response = client.create(
    messages=[
        genai.types.Content(
            role="user",
            parts=[genai.types.Part.from_text(text="{{name}} is {{age}} years old")],
        )
    ],
    response_model=User,
    context={
        "name": "Jason",
        "age": 25,
    },
)

print(response)
# > name='Jason' age=25
```

## Validation and Retries

Instructor can automatically retry requests when validation fails, ensuring you get properly formatted data. This is especially helpful when enforcing specific data requirements:

```python
from typing import Annotated
from pydantic import AfterValidator, BaseModel
import instructor
from google import genai


def uppercase_validator(v: str) -> str:
    if v.islower():
        raise ValueError("Name must be ALL CAPS")
    return v


class UserDetail(BaseModel):
    name: Annotated[str, AfterValidator(uppercase_validator)]
    age: int


client = instructor.from_provider("google/gemini-2.5-flash")

response = client.create(
    messages=[{"role": "user", "content": "Extract: jason is 25 years old"}],
    response_model=UserDetail,
    max_retries=3,
)

print(response)  # UserDetail(name='JASON', age=25)
```

## Multimodal Capabilities

> We've provided a few different sample files for you to use to test out these new features. All examples below use these files.
>
> - (Audio) : A Recording of the Original Gettysburg Address : [gettysburg.wav](https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/gettysburg.wav)
> - (Image) : An image of some blueberry plants [image.jpg](https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/image.jpg)
> - (PDF) : A sample PDF file which contains a fake invoice [invoice.pdf](https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/invoice.pdf)

Instructor provides a unified, provider-agnostic interface for working with multimodal inputs like images, PDFs, and audio files. With Instructor's multimodal objects, you can easily load media from URLs, local files, or base64 strings using a consistent API that works across different AI providers (OpenAI, Anthropic, Mistral, etc.).

Instructor handles all the provider-specific formatting requirements behind the scenes, ensuring your code remains clean and future-proof as provider APIs evolve.

Let's see how to use the Image, Audio and PDF classes.

### Image Processing

!!! info "Autodetect Images"

    For convenient handling of images, you can enable automatic image conversion using the `autodetect_images` parameter. When enabled, Instructor will automatically detect and convert file paths and HTTP URLs provided as strings into the appropriate format required by the Google GenAI SDK. This makes working with images seamless and straightforward. ( see examples below )

Instructor makes it easy to analyse and extract semantic information from images using the Gemini series of models. [Click here](https://ai.google.dev/gemini-api/docs/models) to check if the model you'd like to use has vison capabilities.

Let's see an example below with the sample image above where we'll load it in using our `from_url` method.

Note that we support local files and base64 strings too with the `from_path` and the `from_base64` class methods.

```python
from instructor.processing.multimodal import Image
from pydantic import BaseModel, Field
import instructor
from google.genai import Client


class ImageDescription(BaseModel):
    objects: list[str] = Field(..., description="The objects in the image")
    scene: str = Field(..., description="The scene of the image")
    colors: list[str] = Field(..., description="The colors in the image")


client = instructor.from_provider("google/gemini-2.5-flash")
url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/image.jpg"
# Multiple ways to load an image:
response = client.create(
    response_model=ImageDescription,
    messages=[
        {
            "role": "user",
            "content": [
                "What is in this image?",
                # Option 1: Direct URL with autodetection
                Image.from_url(url),
                # Option 2: Local file
                # Image.from_path("path/to/local/image.jpg")
                # Option 3: Base64 string
                # Image.from_base64("base64_encoded_string_here")
                # Option 4: Autodetect
                # Image.autodetect(<url|path|base64>)
            ],
        },
    ],
)

print(response)
# Example output:
# ImageDescription(
#     objects=['blueberries', 'leaves'],
#     scene='A blueberry bush with clusters of ripe blueberries and some unripe ones against a cloudy sky',
#     colors=['green', 'blue', 'purple', 'white']
# )

```

### Audio Processing

Instructor makes it easy to analyse and extract semantic information from Audio files using the Gemini series of models. Let's see an example below with the sample Audio file above where we'll load it in using our `from_url` method.

Note that we support local files and base64 strings too with the `from_path`

```python
from instructor.processing.multimodal import Audio
from pydantic import BaseModel
import instructor
from google.genai import Client


class AudioDescription(BaseModel):
    transcript: str
    summary: str
    speakers: list[str]
    key_points: list[str]


url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/gettysburg.wav"

client = instructor.from_provider("google/gemini-2.5-flash")

response = client.create(
    response_model=AudioDescription,
    messages=[
        {
            "role": "user",
            "content": [
                "Please transcribe and analyze this audio:",
                # Multiple loading options:
                Audio.from_url(url),
                # Option 2: Local file
                # Audio.from_path("path/to/local/audio.mp3")
            ],
        },
    ],
)

print(response)
# > transcript='Four score and seven years ago our fathers..."]
```

### PDF

Instructor makes it easy to analyse and extract semantic information from PDFs using Gemini's new models.

Let's see an example below with the sample PDF above where we'll load it in using our `from_url` method. With this integration that we're passing in the raw bytes to gemini itself, we also support using the Files api with the `PDFWithGenaiFile` class.

Note that we support local files and base64 strings using this method too with the `from_path` and the `from_base64` class methods.

```python
from instructor.processing.multimodal import PDF
from pydantic import BaseModel
import instructor
from google.genai import Client


class Receipt(BaseModel):
    total: int
    items: list[str]


client = instructor.from_provider("google/gemini-2.5-flash")
url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/invoice.pdf"
# Multiple ways to load an PDF:
response = client.create(
    response_model=Receipt,
    messages=[
        {
            "role": "user",
            "content": [
                "Extract out the total and line items from the invoice",
                # Option 1: Direct URL
                PDF.from_url(url),
                # Option 2: Local file
                # PDF.from_path("path/to/local/invoice.pdf"),
                # Option 3: Base64 string
                # PDF.from_base64("base64_encoded_string_here")
                # Option 4: Autodetect
                # PDF.autodetect(<url|path|base64>)
            ],
        },
    ],
)

print(response)
# > Receipt(total=220, items=['English Tea', 'Tofu'])
```

We also support the use of PDFs with the Gemini `Files` api with the `PDFWithGenaiFile` that allows you to use existing uploaded files or local files.

Note that the `PdfWithGenaiFile.from_new_genai_file` operation is blocking and you can set the timeout and retry delay that we'll call while we await the upload to be registered as completed.

```python
PDFWithGenaiFile.from_new_genai_file(
    "./invoice.pdf",
    retry_delay=1,  # Time to wait before checking if file is ready to use
    max_retries=20 # Number of times to check before throwing an error
),
```

This makes it easier for you to work with the Gemini files API. You can use this in a normal chat completion as seen below

```python
from instructor.processing.multimodal import PDFWithGenaiFile
from pydantic import BaseModel
import instructor
from google.genai import Client


class Receipt(BaseModel):
    total: int
    items: list[str]


client = instructor.from_provider("google/gemini-2.5-flash")
url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/invoice.pdf"
# Multiple ways to load an PDF:
response = client.create(
    response_model=Receipt,
    messages=[
        {
            "role": "user",
            "content": [
                "Extract out the total and line items from the invoice",
                # Option 1: Direct URL
                PDFWithGenaiFile.from_new_genai_file("./invoice.pdf"),

                # Option 2 : Existing Genai File
                # PDFWithGenaiFile.from_existing_genai_file("invoice.pdf"),
            ],
        },
    ],
)

print(response)
```

If you'd like more fine-grained control over the files used, you can also use the `Files` api directly as seen below.

## Using Files

Our API integration also supports the use of files

```python
import instructor
from pydantic import BaseModel


class Summary(BaseModel):
    summary: str


client = instructor.from_provider("google/gemini-2.5-flash")

file1 = client.files.upload(
    file="./gettysburg.wav",
)

# As a parameter
response = client.create(
    messages=[
        {
            "role": "user",
            "content": [
                "Summarise the audio file.",
                file1,
            ]
        }
    ],
    response_model=Summary,
)

print(response)
# > summary="Abraham Lincoln's Gettysburg Address commences by stating that 87 years prior, the founding fathers created a new nation based on liberty and equality. It goes on to say that the Civil War is testing whether a nation so conceived can survive."
```

## Streaming Responses

!!! warning "Streaming Limitations"

    Google GenAI does not support streaming with tool/function calling (`Mode.TOOLS`) for regular models. To use streaming with Gemini:
    
    - Use `Mode.JSON` which supports streaming via `create_partial` and `create_iterable`
    - Or use `Partial[YourModel]` explicitly with `Mode.JSON`
    - Alternatively, set `stream=False` to disable streaming when using `Mode.TOOLS`

Streaming allows you to process responses incrementally rather than waiting for the complete result. This is extremely useful for making UI changes feel instant and responsive.

### Partial Streaming

Receive a stream of complete, validated objects as they're generated:

```python
from pydantic import BaseModel
import instructor


client = instructor.from_provider(
    "google/gemini-2.5-flash",
    mode=instructor.Mode.JSON,
)


class Person(BaseModel):
    name: str
    age: int


class PersonList(BaseModel):
    people: list[Person]


stream = client.create_partial(
    model="gemini-2.5-flash",
    response_model=PersonList,
    stream=True,
    messages=[
        {
            "role": "user",
            "content": "Ivan is 20 years old, Jason is 25 years old, and John is 30 years old",
        }
    ],
)

for extraction in stream:
    print(extraction)
    # > people=[PartialPerson(name='Ivan', age=None)]
    # > people=[PartialPerson(name='Ivan', age=20), PartialPerson(name='Jason', age=25), PartialPerson(name='John', age=None)]
    # > people=[PartialPerson(name='Ivan', age=20), PartialPerson(name='Jason', age=25), PartialPerson(name='John', age=30)]
```

### Iterable Streaming

For extracting multiple objects from a single response, use `create_iterable`:

```python
from pydantic import BaseModel
import instructor

client = instructor.from_provider("google/gemini-2.5-flash")

class User(BaseModel):
    name: str
    age: int

# Extract multiple users from a single response
stream = client.create_iterable(
    model="gemini-2.5-flash",
    response_model=User,
    stream=True,
    messages=[
        {
            "role": "user",
            "content": "Jason is 25 years old, Sarah is 30 years old, and Mike is 28 years old",
        }
    ],
)

for user in stream:
    print(user)
    # > User(name='Jason', age=25)
    # > User(name='Sarah', age=30)
    # > User(name='Mike', age=28)
```

### Async Streaming

Both partial and iterable streaming work with async clients:

```python
import asyncio
from pydantic import BaseModel
import instructor

class User(BaseModel):
    name: str
    age: int

async def async_partial_example():
    client = instructor.from_provider("google/gemini-2.5-flash", async_client=True)
    
    stream = client.create_partial(
        model="gemini-2.5-flash",
        response_model=User,
        stream=True,
        messages=[
            {"role": "user", "content": "Jason is 25 years old"}
        ],
    )
    
    async for chunk in stream:
        print(chunk)

async def async_iterable_example():
    client = instructor.from_provider("google/gemini-2.5-flash", async_client=True)
    
    stream = client.create_iterable(
        model="gemini-2.5-flash",
        response_model=User,
        stream=True,
        messages=[
            {
                "role": "user", 
                "content": "Jason is 25, Sarah is 30, Mike is 28"
            }
        ],
    )
    
    async for user in stream:
        print(user)

# Run async examples
asyncio.run(async_partial_example())
asyncio.run(async_iterable_example())
```

## Async Support

Instructor provides full async support for the genai SDK, allowing you to make non-blocking requests in async applications:

```python
import asyncio

import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


async def extract_user():
    client = instructor.from_provider(
        "google/gemini-2.5-flash",
        async_client=True,
    )

    response = await client.create(
        messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
        response_model=User,
    )
    return response


print(asyncio.run(extract_user()))
#> name = Jason age= 25
```
