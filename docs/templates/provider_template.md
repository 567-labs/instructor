---
title: [Provider Name]
description: Guide to using instructor with [Provider Name]
---

# Structured outputs with [Provider Name], a complete guide w/ instructor

[Brief introduction to the provider, what models they offer, and why someone would use them]

## Quick Start

First, install the required packages:

```bash
pip install "instructor[provider-specific-extras]"
```

You'll need to set up authentication:

```bash
export PROVIDER_API_KEY=your_api_key_here
# Add any other environment variables needed
```

## Basic Example

Here's how to extract structured data using [Provider Name]:

```python
# Standard library imports
import os
from typing import Optional

# Third-party imports
import instructor
from provider_sdk import ClientClass
from pydantic import BaseModel, Field

# Set up environment (typically handled before script execution)
# os.environ["PROVIDER_API_KEY"] = "your-api-key"  # Uncomment and replace with your API key if not set

# Initialize the client with explicit mode
client = instructor.from_provider(
    ClientClass(
        api_key=os.environ.get("PROVIDER_API_KEY", "your_api_key_here"),
        # Other configuration options
    ),
    mode=instructor.Mode.PROVIDER_SPECIFIC_MODE,
)

# Define your data structure with proper annotations
class UserExtract(BaseModel):
    """Model for extracting user information from text."""
    name: str = Field(description="The user's full name")
    age: int = Field(description="The user's age in years")

# Extract structured data
try:
    user = client.create(
        model="provider-model-name",  # Use latest stable model version
        response_model=UserExtract,
        messages=[
            {"role": "system", "content": "Extract structured user information from the text."},
            {"role": "user", "content": "Extract jason is 25 years old"},
        ],
    )

    print(user.model_dump_json(indent=2))
    # Expected output:
    # {
    #   "name": "Jason",
    #   "age": 25
    # }
except Exception as e:
    print(f"Error: {e}")
```

## Async Example

For asynchronous use cases:

```python
# Standard library imports
import os
import asyncio
from typing import Optional

# Third-party imports
import instructor
from provider_sdk import AsyncClientClass
from pydantic import BaseModel, Field

# Set up environment (typically handled before script execution)
# os.environ["PROVIDER_API_KEY"] = "your-api-key"  # Uncomment and replace with your API key if not set

# Define your data structure with proper annotations
class UserExtract(BaseModel):
    """Model for extracting user information from text."""
    name: str = Field(description="The user's full name")
    age: int = Field(description="The user's age in years")

# Initialize the async client with explicit mode
client = instructor.from_provider(
    AsyncClientClass(
        api_key=os.environ.get("PROVIDER_API_KEY", "your_api_key_here"),
    ),
    mode=instructor.Mode.PROVIDER_SPECIFIC_MODE,
)

async def extract_data(text: str) -> UserExtract:
    """
    Asynchronously extract structured data from text.

    Args:
        text: The input text to extract from

    Returns:
        A structured UserExtract object
    """
    try:
        user = await client.create(
            model="provider-model-name",  # Use latest stable model version
            response_model=UserExtract,
            messages=[
                {"role": "system", "content": "Extract structured user information from the text."},
                {"role": "user", "content": text},
            ],
        )
        return user
    except Exception as e:
        print(f"Error during extraction: {e}")
        raise

# Example usage
async def main():
    result = await extract_data("Extract jason is 25 years old")
    print(result.model_dump_json(indent=2))

# Run the async function
if __name__ == "__main__":
    asyncio.run(main())

# Expected output:
# {
#   "name": "Jason",
#   "age": 25
# }
```

## Supported Modes

[Provider Name] supports the following instructor modes:

- `Mode.MODE_1` - Description of when to use this mode
- `Mode.MODE_2` - Description of when to use this mode
- [Additional modes as needed]

## Streaming Support

You can stream results with [Provider Name]:

```python
# Streaming partial results example code
```

## Provider-Specific Features

[Describe any special features or considerations specific to this provider]

## Models

[Provider Name] offers the following models:

- `model-1` - Description of capabilities
- `model-2` - Description of capabilities
- [More models as appropriate]