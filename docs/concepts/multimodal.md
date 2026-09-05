---
title: Seamless Multimodal Interactions with Instructor
description: Learn how the Image, PDF and Audio class in Instructor enables seamless handling of multimodal content across different AI models.
---

---
title: Multimodal Processing with Instructor - Vision and Audio
description: Process images, audio, and video with Instructor for multimodal structured outputs. Extract data from visual content using GPT-4 Vision and Gemini models.
---

# Multimodal

> We've provided a few different sample files for you to use to test out these new features. All examples below use these files.
>
> - (Image) : An image of some blueberry plants [image.jpg](https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/image.jpg)
> - (Audio) : A Recording of the Original Gettysburg Address : [gettysburg.wav](https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/gettysburg.wav)
> - (PDF) : A sample PDF file which contains a fake invoice [invoice.pdf](https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/invoice.pdf)
>   Instructor provides a unified, provider-agnostic interface for working with multimodal inputs like images and PDFs.

Instructor provides a unified, provider-agnostic interface for working with multimodal inputs like images, PDFs, and audio files.

With Instructor's multimodal objects, you can easily load media from URLs, Google Cloud Storage URLs, local files, or base64 strings using a consistent API that works across different AI providers (OpenAI, Anthropic, Mistral, etc.).

Instructor handles all the provider-specific formatting requirements behind the scenes, ensuring your code remains clean and future-proof as provider APIs evolve. Let's see how to use the Image, Audio and PDF classes.

## `Image`

This class represents an image that can be loaded from a URL or file path. It provides a set of methods to create `Image` instances from different sources (Eg. URLs, paths and base64 strings). The following shows which methods are supported for the individual providers.

| Method            | OpenAI | Anthropic | Google GenAI |
| ----------------- | ------ | --------- | ------------ |
| `from_url()`      | ✅     | ✅        | ✅           |
| `from_gs_url()`   | ✅     | ✅        | ✅           |
| `from_path()`     | ✅     | ✅        | ✅           |
| `from_base64()`   | ✅     | ✅        | ✅           |
| `autodetect()`    | ✅     | ✅        | ✅           |

We also support Anthropic Prompt Caching for images with the `ImageWith

### Usage

By using the `Image` class, we can abstract away the differences between the different formats, allowing you to work with a unified interface.

You can create an `Image` instance from a URL, Google Cloud Storage (GCS) URL, or file path using the `from_url`, `from_gs_url`, or `from_path` methods. The `Image` class will automatically convert the image to a base64-encoded string and include it in the API request.

```python
import instructor
from instructor.processing.multimodal import Image
from pydantic import BaseModel


class ImageDescription(BaseModel):
    description: str
    items: list[str]


# Use our sample image provided above.
url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/image.jpg"

client = instructor.from_provider("openai/gpt-4.1-mini")

response = client.create(
    response_model=ImageDescription,
    messages=[
        {
            "role": "user",
            "content": [
                "What is in this image?",
                Image.from_url(url),
            ],
        }
    ],
)

print(response)
"""
description='Blueberry bushes with clusters of ripe and unripe blueberries. The berries are blue to purplish in color, and the leaves are green. The sky in the background is cloudy.' items=['blueberry bushes', 'ripe blueberries', 'unripe blueberries', 'green leaves', 'cloudy sky']
"""
```

### Google Cloud Storage Support

Instructor now supports loading images directly from Google Cloud Storage URLs. This is particularly useful when working with images stored in GCS buckets.

```python
import instructor
from instructor.processing.multimodal import Image
from pydantic import BaseModel


class ImageDescription(BaseModel):
    description: str
    items: list[str]


# Load image from GCS URL (must be publicly accessible)
gs_url = "gs://my-bucket/path/to/image.jpg"

client = instructor.from_provider("openai/gpt-4.1-mini")

response = client.create(
    response_model=ImageDescription,
    messages=[
        {
            "role": "user",
            "content": [
                "What is in this image?",
                Image.from_gs_url(gs_url),
            ],
        }
    ],
)

print(response)
"""
description='A sample image loaded from Google Cloud Storage.' items=['sample image']
"""
```

> **Note**: GCS URLs must point to publicly accessible objects. The `from_gs_url` method converts `gs://` URLs to `https://storage.googleapis.com/` URLs for access.

We also provide an `autodetect_images` keyword argument that allows you to provide URLs, GCS URLs, or file paths as normal strings when you set it to true. The system will automatically detect and handle different media types including images, audio, and PDFs.

You can see an example below.

```python
import instructor
from pydantic import BaseModel


class ImageDescription(BaseModel):
    description: str
    items: list[str]


# Download a sample image for demonstration
url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/image.jpg"

client = instructor.from_provider("openai/gpt-4.1-mini")

response = client.create(
    response_model=ImageDescription,
    autodetect_images=True,  # Set this to True
    messages=[
        {
            "role": "user",
            "content": ["What is in this image?", url],
        }
    ],
)

print(response)
"""
description='The image shows a close-up of a blueberry bush with ripe blueberries and green leaves. The background includes more blueberry bushes and a cloudy sky.' items=['Blueberry bush', 'Ripe blueberries', 'Green leaves', 'Cloudy sky']
"""
```

If you'll like to support Anthropic prompt caching with images, we provide the `ImageWithCacheControl` Object to do so. Simply use the `from_image_params` method and you'll be able to leverage Anthropic's prompt caching.

```python
import instructor
from instructor.processing.multimodal import ImageWithCacheControl
from pydantic import BaseModel


class ImageDescription(BaseModel):
    description: str
    items: list[str]


# Download a sample image for demonstration
url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/image.jpg"

client = instructor.from_provider("anthropic/claude-3-5-sonnet-20240620")

response, completion = client.create_with_completion(
    response_model=ImageDescription,
    autodetect_images=True,  # Set this to True
    messages=[
        {
            "role": "user",
            "content": [
                "What is in this image?",
                ImageWithCacheControl.from_image_params(
                    {
                        "source": url,
                        "cache_control": {
                            "type": "ephemeral",
                        },
                    }
                ),
            ],
        }
    ],
    max_tokens=1000,
)

print(response)
"""
description='A bush with numerous clusters of blueberries surrounded by green leaves, under a cloudy sky.' items=['blueberries', 'green leaves', 'cloudy sky']
"""

print(completion.usage.cache_creation_input_tokens)
#> 1820
```

By leveraging Instructor's multimodal capabilities, you can focus on building your application logic without worrying about the intricacies of each provider's image handling format. This not only saves development time but also makes your code more maintainable and adaptable to future changes in AI provider APIs.

## `Audio`

> Note : Only OpenAI and Gemini support audio files at the moment. For Gemini, we're passing in the raw bytes as bytes for this feature. If you'd like to use the `Files` API instead, we also support it, [read more at](../integrations/genai.md) to see how to do so.

Similar to the Image class, we provide methods to create `Audio` instances.

| Method          | OpenAI | Google GenAI |
| --------------- | ------ | ------------ |
| `from_url()`    | ✅     | ✅           |
| `from_gs_url()` | ✅     | ✅           |
| `from_path()`   | ✅     | ✅           |
| `from_base64()` | ✅     | ✅           |
| `autodetect()`  | ✅     | ✅           |

The `Audio` class represents an audio file that can be loaded from a URL, Google Cloud Storage URL, or file path. It provides methods to create `Audio` instances using the `from_path`, `from_url`, `from_gs_url`, `from_base64`, and `autodetect` methods.

The `Audio` class will automatically convert it to the right format and include it in the API request.

```python
from pydantic import BaseModel
import instructor
from instructor.processing.multimodal import Audio

# Initialize the client
client = instructor.from_provider("openai/gpt-4o-audio-preview")


# Define our response model
class AudioDescription(BaseModel):
    summary: str
    transcript: str


url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/gettysburg.wav"

# Make the API call with the audio file
resp = client.create(
    response_model=AudioDescription,
    modalities=["text"],
    audio={"voice": "alloy", "format": "wav"},
    messages=[
        {
            "role": "user",
            "content": [
                "Extract the following information from the audio:",
                Audio.from_url(url),
            ],
        },
    ],
)

print(resp)
"""
summary='This excerpt is from a famous historical speech discussing the founding principles of equality and liberty, and the ongoing civil war testing the endurance of those principles.' transcript='Four score and seven years ago our fathers brought forth on this continent a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure.'
"""
```

### Google Cloud Storage Support

You can also load audio files directly from Google Cloud Storage:

```python
from pydantic import BaseModel
import instructor
from instructor.processing.multimodal import Audio

# Initialize the client
client = instructor.from_provider("openai/gpt-4o-audio-preview")


# Define our response model
class AudioDescription(BaseModel):
    summary: str
    transcript: str


# Load audio from GCS URL (must be publicly accessible)
gs_url = "gs://my-bucket/path/to/audio.wav"

# Make the API call with the GCS audio file
resp = client.create(
    response_model=AudioDescription,
    modalities=["text"],
    audio={"voice": "alloy", "format": "wav"},
    messages=[
        {
            "role": "user",
            "content": [
                "Extract the following information from the audio:",
                Audio.from_gs_url(gs_url),
            ],
        },
    ],
)

print(resp)
"""
summary='A short historical speech about equality and liberty.' transcript='Four score and seven years ago our fathers brought forth...'
"""
```

## `PDF`

The `PDF` class represents a PDF file that can be loaded from a URL or file path.

It provides methods to create `PDF` instances and is currently supported for OpenAI, Mistral, GenAI, Anthropic, and Bedrock client integrations.

| Method            | OpenAI | Anthropic | Google GenAI | Mistral | Bedrock |
| ----------------- | ------ | --------- | ------------ | ------- | ------- |
| `from_url()`      | ✅     | ✅        | ✅           | ✅      | ✅      |
| `from_gs_url()`   | ✅     | ✅        | ✅           | ✅      | ✅      |
| `from_path()`     | ✅     | ✅        | ✅           | ❎      | ✅      |
| `from_base64()`   | ✅     | ✅        | ✅           | ❎      | ✅      |
| `autodetect()`    | ✅     | ✅        | ✅           | ✅      | ✅      |

For Gemini, we also provide two additional methods that make working with the google-genai files package easy which you can access in the `PDFWithGenaiFile` object.

For Anthropic, you can enable caching with the `PDFWithCacheControl` object. Note that this has caching configured by default for easy usage.

We provide examples of how to use all three object classes below.

For Bedrock, you can convert a `PDF` into the Bedrock-native document format with `PDF.to_bedrock()` and include the result in the message content list.

### Usage

```python
import instructor
from pydantic import BaseModel
from instructor.processing.multimodal import PDF

# Set up the client
url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/invoice.pdf"
client = instructor.from_provider("openai/gpt-4.1-mini")


# Create a model for analyzing PDFs
class Invoice(BaseModel):
    total: float
    items: list[str]


# Load and analyze a PDF
response = client.create(
    response_model=Invoice,
    messages=[
        {
            "role": "user",
            "content": [
                "Analyze this document",
                PDF.from_url(url),
            ],
        }
    ],
)

print(response)
"""
total=220.0 items=['English Tea - 2 units at $100 each', 'Tofu - 10 units at $2 each']
"""
```

### Google Cloud Storage Support

You can load PDF files directly from Google Cloud Storage URLs:

```python
import instructor
from pydantic import BaseModel
from instructor.processing.multimodal import PDF

# Set up the client
gs_url = "gs://my-bucket/path/to/document.pdf"
client = instructor.from_provider("openai/gpt-4.1-mini")


# Create a model for analyzing PDFs
class Invoice(BaseModel):
    total: float
    items: list[str]


# Load and analyze a PDF from GCS (must be publicly accessible)
response = client.create(
    response_model=Invoice,
    messages=[
        {
            "role": "user",
            "content": [
                "Analyze this document",
                PDF.from_gs_url(gs_url),
            ],
        }
    ],
)

print(f"Total = {response.total:.0f}, items = {response.items}")
#> Total = 220, items = ['English Tea', 'Tofu']
```

### Caching

If you'd like to cache the PDF for Anthropic, we provide the `PDFWithCacheControl` class which has caching configured by default.

```python
import instructor
from pydantic import BaseModel
from instructor.processing.multimodal import PDFWithCacheControl

# Set up the client
url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/invoice.pdf"
client = instructor.from_provider("anthropic/claude-3-5-sonnet-20240620")


# Create a model for analyzing PDFs
class Invoice(BaseModel):
    total: float
    items: list[str]


# Load and analyze a PDF
response, completion = client.create_with_completion(
    response_model=Invoice,
    messages=[
        {
            "role": "user",
            "content": [
                "Analyze this document",
                PDFWithCacheControl.from_url(url),
            ],
        }
    ],
    max_tokens=1000,
)

print(f"Total = {response.total:.0f}, items = {response.items}")
#> Total = 220, items = ['English Tea', 'Tofu']

print(completion.usage.cache_creation_input_tokens)
#> 2091
```

### Using Files

We also provide a convenient wrapper around the Files API - allowing you to use both uploaded files and to block the main thread while your file is uploading.

In this example below, we download the sample PDF and then upload it using the `Files` api provided by the `google.genai` sdk.

```python
import instructor
from pydantic import BaseModel
from instructor.processing.multimodal import PDFWithGenaiFile
import requests

# Set up the client
url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/invoice.pdf"
client = instructor.from_provider("google/gemini-2.5-flash")


# Create a model for analyzing PDFs
class Invoice(BaseModel):
    total: float
    items: list[str]


# Load and analyze a PDF
with requests.get(url) as download_response:
    pdf_data = download_response.content
    with open("./invoice.pdf", "wb") as f:
        f.write(pdf_data)

response = client.create(
    response_model=Invoice,
    messages=[
        {
            "role": "user",
            "content": [
                "Analyze this document",
                PDFWithGenaiFile.from_new_genai_file(
                    file_path="./invoice.pdf",
                    retry_delay=10,
                    max_retries=20,
                ),
            ],
        }
    ],
)

print(response)
#> total=220.0 items=['English Tea', 'Tofu']
```

If you've already uploaded your file ahead of time, we also support it. Just provide us with the file name as seen below

```python
import instructor
from pydantic import BaseModel
from instructor.processing.multimodal import PDFWithGenaiFile
import requests

# Set up the client
url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/invoice.pdf"
client = instructor.from_provider("google/gemini-2.5-flash")


# Create a model for analyzing PDFs
class Invoice(BaseModel):
    total: float
    items: list[str]


# Load and analyze a PDF
with requests.get(url) as download_response:
    pdf_data = download_response.content
    with open("./invoice.pdf", "wb") as f:
        f.write(pdf_data)

file = client.files.upload(
    file="invoice.pdf",
)

response = client.create(
    response_model=Invoice,
    messages=[
        {
            "role": "user",
            "content": [
                "Analyze this document",
                PDFWithGenaiFile.from_existing_genai_file(file_name=file.name),
            ],
        }
    ],
)

print(response)
#> total=220.0 items=['English Tea', 'Tofu']
```

This way you have more granular control over how the file is uploaded, potentially also processing multiple file uploads at once too.
