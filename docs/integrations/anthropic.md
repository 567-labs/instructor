---
title: "Anthropic Claude Tutorial: Structured Outputs with Instructor"
description: "Use Claude Sonnet 5 with Instructor for structured data extraction and type-safe outputs in Python."
---

## See Also

- [Getting Started](../getting-started.md) - Quick start guide
- [from_provider Guide](../concepts/from_provider.md) - Detailed client configuration
- [Provider Examples](../index.md#provider-examples) - Quick examples for all providers
- [Mode Comparison](../modes-comparison.md) - Using Anthropic's tool calling

# Anthropic Claude Tutorial: Structured Outputs with Instructor

Use Claude Sonnet 5 with Instructor to extract structured, validated data, from basic setup through streaming and adaptive thinking.

## Quick Start: Install Instructor for Claude

Get started with Claude and Instructor for structured outputs:

```
pip install "instructor[anthropic]"
```

Once we've done so, getting started is as simple as using our `from_provider` method to patch the client up.

### Basic Usage

```python
# Standard library imports
from typing import List

# Third-party imports
import instructor
from pydantic import BaseModel, Field

# Set up environment (typically handled before script execution)
# os.environ["ANTHROPIC_API_KEY"] = "your-api-key"  # Uncomment and replace with your API key if not set


# Define your models with proper type annotations
class Properties(BaseModel):
    """Model representing a key-value property."""

    name: str = Field(description="The name of the property")
    value: str = Field(description="The value of the property")


class User(BaseModel):
    """Model representing a user with properties."""

    name: str = Field(description="The user's full name")
    age: int = Field(description="The user's age in years")
    properties: List[Properties] = Field(description="List of user properties")


client = instructor.from_provider(
    "anthropic/claude-sonnet-5", mode=instructor.Mode.TOOLS
)

try:
    # Extract structured data
    user_response = client.create(
        max_tokens=1024,
        messages=[
            {
                "role": "system",
                "content": "Extract structured information based on the user's request.",
            },
            {
                "role": "user",
                "content": "Create a user for a model with a name, age, and properties.",
            },
        ],
        response_model=User,
    )

    # Print the result as formatted JSON
    print(user_response.model_dump_json(indent=2))

    # Expected output:
    # {
    #   "name": "John Doe",
    #   "age": 35,
    #   "properties": [
    #     {
    #       "name": "City",
    #       "value": "New York"
    #     },
    #     {
    #       "name": "Occupation",
    #       "value": "Software Engineer"
    #     }
    #   ]
    # }
except instructor.exceptions.InstructorError as e:
    print(f"Validation error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Async Example

```python
import asyncio

async_client = instructor.from_provider(
    "anthropic/claude-sonnet-5",
    async_client=True,
    mode=instructor.Mode.TOOLS,
)


async def extract_user():
    return await async_client.create(
        messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
        response_model=User,
    )


user = asyncio.run(extract_user())
print(user)
```

### Parallel Tool Calling

Parallel tool mode is automatically detected when your response model is `Iterable[Union[Model1, Model2, ...]]`. Just use `Mode.TOOLS` (or let it default) and the handler will automatically:
- Set tool_choice to "auto" (required for parallel)
- Generate schemas for all union members
- Return a generator yielding each tool result

```python
from typing import Iterable, Literal
from pydantic import BaseModel
import instructor


class Weather(BaseModel):
    location: str
    units: Literal["imperial", "metric"]


class GoogleSearch(BaseModel):
    query: str


# No need to specify Mode.PARALLEL_TOOLS - it's auto-detected!
client = instructor.from_provider(
    "anthropic/claude-sonnet-5",
    mode=instructor.Mode.TOOLS,  # or just omit and use default
)

results = client.create(
    messages=[
        {"role": "system", "content": "You must always use tools"},
        {
            "role": "user",
            "content": "What is the weather in toronto and dallas and who won the super bowl?",
        },
    ],
    response_model=Iterable[Weather | GoogleSearch],  # Auto-detects parallel mode
)

for item in results:
    print(item)
```

**How it works**: When Instructor detects `Iterable[Union[...]]`, it automatically:
1. Sets `tool_choice` to `"auto"` (allows model to call any tool)
2. Generates tool schemas from all union members
3. Returns a generator that yields each extracted tool call
4. Each yielded item is validated against its corresponding Pydantic model

## Multimodal

> We've provided a few different sample files for you to use to test out these new features. All examples below use these files.
>
> - (Image) : An image of some blueberry plants [image.jpg](https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/image.jpg)
> - (PDF) : A sample PDF file which contains a fake invoice [invoice.pdf](https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/invoice.pdf)

Instructor provides a unified, provider-agnostic interface for working with multimodal inputs like images, PDFs, and audio files. With Instructor's multimodal objects, you can easily load media from URLs, local files, or base64 strings using a consistent API that works across different AI providers (OpenAI, Anthropic, Mistral, etc.).

Instructor handles all the provider-specific formatting requirements behind the scenes, ensuring your code remains clean and future-proof as provider APIs evolve.

Let's see how to use the Image and PDF classes.

### Image

> For a more in-depth walkthrough of the Image component, check out the [docs here](../concepts/multimodal.md)

Instructor makes it easy to analyse and extract semantic information from images using Anthropic's claude models. [Click here](https://docs.anthropic.com/en/docs/about-claude/models/all-models) to check if the model you'd like to use has vison capabilities.

Let's see an example below with the sample image above where we'll load it in using our `from_url` method.

Note that we support local files and base64 strings too with the `from_path` and the `from_base64` class methods.

```python
from instructor.processing.multimodal import Image
from pydantic import BaseModel, Field
import instructor


class ImageDescription(BaseModel):
    objects: list[str] = Field(..., description="The objects in the image")
    scene: str = Field(..., description="The scene of the image")
    colors: list[str] = Field(..., description="The colors in the image")


client = instructor.from_provider("anthropic/claude-sonnet-5")
url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/image.jpg"
# Multiple ways to load an image:
response = client.create(
    response_model=ImageDescription,
    max_tokens=1000,
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

### PDF

Instructor makes it easy to analyse and extract semantic information from PDFs using Anthropic's Claude line of models.

Let's see an example below with the sample PDF above where we'll load it in using our `from_url` method.

Note that we support local files and base64 strings too with the `from_path` and the `from_base64` class methods.

```python
from instructor.processing.multimodal import PDF
from pydantic import BaseModel
import instructor


class Receipt(BaseModel):
    total: int
    items: list[str]


client = instructor.from_provider("anthropic/claude-sonnet-5")
url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/invoice.pdf"
# Multiple ways to load an PDF:
response = client.create(
    response_model=Receipt,
    max_tokens=1000,
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
#> Receipt(total=220, items=['English Tea', 'Tofu'])
```

If you'd like to cache the PDF and use it across multiple different requests, we support that with the `PDFWithCacheControl` class which we can see below.

```python
from instructor.processing.multimodal import PDFWithCacheControl
from pydantic import BaseModel
import instructor


class Receipt(BaseModel):
    total: int
    items: list[str]


client = instructor.from_provider("anthropic/claude-sonnet-5")
url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/invoice.pdf"
# Multiple ways to load an PDF:
response, completion = client.create_with_completion(
    response_model=Receipt,
    max_tokens=1000,
    messages=[
        {
            "role": "user",
            "content": [
                "Extract out the total and line items from the invoice",
                # Option 1: Direct URL
                PDFWithCacheControl.from_url(url),
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

assert (
    completion.usage.cache_creation_input_tokens > 0
    or completion.usage.cache_read_input_tokens > 0
)
print(response)
#> Receipt(total=220, items=['English Tea', 'Tofu'])
```

## Streaming Support

Instructor has two main ways that you can use to stream responses out

1. **Iterables**: These are useful when you'd like to stream a list of objects of the same type (Eg. use structured outputs to extract multiple users)
2. **Partial Streaming**: This is useful when you'd like to stream a single object and you'd like to immediately start processing the response as it comes in.

### Partials

You can use our `create_partial` method to stream a single object. Note that validators should not be declared in the response model when streaming objects because it will break the streaming process.

```python
# Standard library imports

# Third-party imports
import instructor
from pydantic import BaseModel, Field

# Set up environment (typically handled before script execution)
# os.environ["ANTHROPIC_API_KEY"] = "your-api-key"  # Uncomment and replace with your API key if not set

# Initialize client with explicit mode
client = instructor.from_provider(
    "anthropic/claude-sonnet-5",
    mode=instructor.Mode.TOOLS,
)


# Define your model with proper annotations
class User(BaseModel):
    """Model representing a user profile."""

    name: str = Field(description="The user's full name")
    age: int = Field(description="The user's age in years")
    bio: str = Field(description="A biographical description of the user")


try:
    # Stream partial objects as they're generated
    for partial_user in client.create_partial(
        messages=[
            {
                "role": "system",
                "content": "Create a detailed user profile based on the information provided.",
            },
            {"role": "user", "content": "Create a user profile for Jason, age 25"},
        ],
        response_model=User,
        max_tokens=4096,
    ):
        print(f"Current state: {partial_user}")

    # Expected output:
    #> Current state: name='Jason' age=None bio=None
    #> Current state: name='Jason' age=25 bio='Jason is a 25-year-old with an adventurous spirit and a love for technology. He is'
    #> Current state: name='Jason' age=25 bio='Jason is a 25-year-old with an adventurous spirit and a love for technology. He is always on the lookout for new challenges and opportunities to grow both personally and professionally.'
except Exception as e:
    print(f"Error during streaming: {e}")
```

### Iterable Example

You can also use our `create_iterable` method to stream a list of objects. This is helpful when you'd like to extract multiple instances of the same response model from a single prompt.

```python
# Standard library imports

# Third-party imports
from instructor import from_provider
from pydantic import BaseModel, Field

# Set up environment (typically handled before script execution)
# os.environ["ANTHROPIC_API_KEY"] = "your-api-key"  # Uncomment and replace with your API key if not set

# Initialize client with explicit mode
client = from_provider(mode=instructor.Mode.TOOLS)


# Define your model with proper annotations
class User(BaseModel):
    """Model representing a basic user."""

    name: str = Field(description="The user's full name")
    age: int = Field(description="The user's age in years")


try:
    # Create an iterable of user objects
    users = client.create_iterable(
        messages=[
            {
                "role": "system",
                "content": "Extract all users from the provided text into structured format.",
            },
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
        max_tokens=4096,
        response_model=User,
    )

    # Process each user as it's extracted
    for user in users:
        print(user)

    # Expected output:
    #> name='Jason' age=25
    #> name='Sarah' age=30
    #> name='Mike' age=28
except Exception as e:
    print(f"Error during iteration: {e}")
```

## Instructor Modes

We provide several modes to make it easy to work with the different response models that Anthropic supports

1. `instructor.Mode.JSON` : This uses the text completion API from the Anthropic API and then extracts out the desired response model from the text completion model
2. `instructor.Mode.TOOLS` : This uses Anthropic's [tools calling API](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) to return structured outputs. Automatically detects parallel tools from `Iterable[Union[...]]` response models.
3. `instructor.Mode.PARALLEL_TOOLS` : **Deprecated** - Use `Mode.TOOLS` with `Iterable[Union[Model1, Model2, ...]]` instead. Auto-detected automatically.

### Mode Auto-Detection

`Mode.TOOLS` now intelligently adapts based on your response model and parameters:

| Response Model | Parameters | Behavior |
|---|---|---|
| `Model` | Regular | Single tool (forced) |
| `Model` | `thinking={...}` | Single tool with extended thinking (auto) |
| `Iterable[Union[Model1, Model2]]` | Regular | Parallel tools (auto) |
| `Iterable[Union[Model1, Model2]]` | `thinking={...}` | Parallel with thinking |

In general, we recommend using `Mode.TOOLS` because it automatically handles all these cases and is the best way to ensure you have the desired response schema.

## Caching

If you'd like to use caching with the Anthropic Client, we also support it for images and text input.

### Caching Text Input

Here's how you can implement caching for text input ( assuming you have a giant `book.txt` file that you read in).

We've written a comprehensive walkthrough of how to use caching to implement Anthropic's new Contextual Retrieval method that gives a significant bump to retrieval accuracy.

```python
# Standard library imports

# Third-party imports
import instructor
from pydantic import BaseModel, Field

# Set up environment (typically handled before script execution)
# os.environ["ANTHROPIC_API_KEY"] = "your-api-key"  # Uncomment and replace with your API key if not set


# Define your Pydantic model with proper annotations
class Character(BaseModel):
    """Model representing a character extracted from text."""

    name: str = Field(description="The character's full name")
    description: str = Field(description="A description of the character")


# Initialize client with explicit mode and prompt caching
client = instructor.from_provider(
    "anthropic/claude-sonnet-5",
    mode=instructor.Mode.TOOLS,
)

try:
    # Load your large context
    with open("./book.txt") as f:
        book = f.read()

    # Make multiple calls using the cached context
    for _ in range(2):
        # The first time processes the large text, subsequent calls use the cache
        resp, completion = client.create_with_completion(
            messages=[
                {
                    "role": "system",
                    "content": "Extract character information from the provided text.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "<book>" + book + "</book>",
                            "cache_control": {"type": "ephemeral"},  # Mark for caching
                        },
                        {
                            "type": "text",
                            "text": "Extract a character from the text given above",
                        },
                    ],
                },
            ],
            response_model=Character,
            max_tokens=1000,
        )

        # Process the result
        print(f"Character: {resp.name}")
        print(f"Description: {resp.description}")

        # The completion contains the raw response
        print(f"Raw completion length: {len(completion)}")

    # Note: Second iteration should be faster due to cache hit

except Exception as e:
    print(f"Error: {e}")
```

### Caching Images

We also support caching for images. This helps significantly, especially if you're using images repeatedly to save on costs. Read more about it [here](../concepts/caching.md)

```python
# Standard library imports

# Third-party imports
import instructor
from pydantic import BaseModel, Field

# Set up environment (typically handled before script execution)
# os.environ["ANTHROPIC_API_KEY"] = "your-api-key"  # Uncomment and replace with your API key if not set


# Define your model for image analysis
class ImageAnalyzer(BaseModel):
    """Model for analyzing image content."""

    content_description: str = Field(
        description="Description of what appears in the images"
    )
    objects: list[str] = Field(description="List of objects visible in the images")
    scene_type: str = Field(
        description="Type of scene shown in the images (indoor, outdoor, etc.)"
    )


# Initialize client with explicit mode and image caching enabled
client = instructor.from_provider(
    "anthropic/claude-sonnet-5",
    mode=instructor.Mode.TOOLS,
)

try:
    # Configure cache control for images
    cache_control = {"type": "ephemeral"}

    # Make a request with cached images
    response = client.create(
        response_model=ImageAnalyzer,
        messages=[
            {
                "role": "system",
                "content": "Analyze the content of the provided images in detail.",
            },
            {
                "role": "user",
                "content": [
                    "What is in these two images?",
                    # Remote image with caching
                    {
                        "type": "image",
                        "source": "https://example.com/image.jpg",
                        "cache_control": cache_control,
                    },
                    # Local image with caching
                    {
                        "type": "image",
                        "source": "path/to/image.jpg",
                        "cache_control": cache_control,
                    },
                ],
            },
        ],
        autodetect_images=True,  # Automatically handle image content
    )

    # Process the results
    print(f"Description: {response.content_description}")
    print(f"Objects: {', '.join(response.objects)}")
    print(f"Scene type: {response.scene_type}")

    # Subsequent identical requests will use cached images

except Exception as e:
    print(f"Error during image analysis: {e}")
```

## Thinking (Extended Thinking)

Anthropic supports extended thinking with their Claude models, enabling the model to think through complex problems before providing structured outputs. In Instructor, use `Mode.TOOLS` with the `thinking` parameter to enable this feature.

### Using Extended Thinking with TOOLS

```python
import instructor
from pydantic import BaseModel


class Answer(BaseModel):
    answer: float


client = instructor.from_provider("anthropic/claude-sonnet-5")
response = client.create(
    response_model=Answer,
    messages=[
        {
            "role": "user",
            "content": "Which is larger, 9.11 or 9.8?",
        },
    ],
    max_tokens=2000,
    thinking={"type": "adaptive"},
    tool_choice={"type": "auto"},
)

# Response is a validated Answer object
assert isinstance(response, Answer)
assert response.answer == 9.8
```

### How It Works

When you provide the `thinking` parameter with `type: "adaptive"` for Sonnet 5:

1. **Tool Choice**: Pass `tool_choice={"type": "auto"}` explicitly when using adaptive thinking. Thinking does not support a forced tool choice.
2. **Model Reasoning**: Claude decides how much reasoning the request needs; adaptive thinking does not use `budget_tokens`.
3. **Structured Output**: After reasoning, the model returns a valid tool call with your response model
4. **Validation**: The response is automatically validated against your Pydantic model

### Deprecation Notice

`Mode.ANTHROPIC_REASONING_TOOLS` is deprecated. Use `Mode.TOOLS` with the `thinking` parameter instead. Both modes now support thinking, but using the standard `TOOLS` mode is preferred and more flexible.
