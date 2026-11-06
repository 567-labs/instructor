---
title: Audio Information Extraction with OpenAI
description: Learn how to extract structured information from audio files using OpenAI's audio capabilities and Instructor for type-safe data extraction.
---

# Audio Information Extraction with OpenAI

This example demonstrates how to use Instructor with OpenAI's audio capabilities to extract structured information from audio files. The example shows how to process audio input and extract specific fields into a Pydantic model.

## Prerequisites

- OpenAI API key with access to GPT-4 audio models
- An audio file in WAV format
- Instructor library installed with OpenAI support

## Code Example

```python
from pydantic import BaseModel
import instructor
from instructor.processing.multimodal import Audio
import base64

# Initialize the OpenAI client with Instructor
client = instructor.from_provider("openai/gpt-5-nano")


# Define the structure for extracted information
class Person(BaseModel):
    name: str
    age: int


# Read and encode the audio file
with open("./output.wav", "rb") as f:
    encoded_string = base64.b64encode(f.read()).decode("utf-8")

# Extract information from the audio
resp = client.create(
    model="gpt-4-audio-preview",
    response_model=Person,
    modalities=["text"],
    audio={"voice": "alloy", "format": "wav"},
    messages=[
        {
            "role": "user",
            "content": [
                "Extract the following information from the audio",
                Audio.from_path("./output.wav"),
            ],
        },
    ],
)

print(resp)
# Example output: Person(name='Jason', age=20)
```

## How It Works

1. First, we import the necessary libraries including the `Audio` class from `instructor.processing.multimodal`.

2. We define a Pydantic model `Person` that specifies the structure of the information we want to extract from the audio:
   - `name`: The person's name
   - `age`: The person's age

3. The audio file is read and encoded in base64 format.

4. We use OpenAI's audio-capable model to process the audio and extract the specified information:
   - The `model` parameter specifies the GPT-4 audio model
   - `response_model` tells Instructor to structure the output according to our `Person` model
   - `modalities` specifies that we want text output
   - The `audio` parameter configures audio-specific settings
   - In the message, we use `Audio.from_path()` to include the audio file

5. The response is automatically parsed into our Pydantic model, making the extracted information easily accessible in a structured format.

## Use Cases

This pattern is particularly useful for:

- Transcribing and extracting information from recorded interviews
- Processing voice messages or audio notes
- Automated form filling from voice input
- Voice-based data entry systems

## Tips

- Ensure your audio file is in a supported format (WAV in this example)
- The audio model works best with clear speech and minimal background noise
- Consider the length of the audio file, as there may be model-specific limitations
- Structure your Pydantic model to match the information you expect to extract

## Related Examples

- [Multi-Modal Data with Gemini](multi_modal_gemini.md)
- [Structured Outputs with OpenAI](../integrations/openai.md)