---
authors:
  - jxnl
categories:
  - OpenAI
  - Audio
comments: true
date: 2024-10-17
description: Explore the new audio capabilities in OpenAI's Chat Completions API using the gpt-4o-audio-preview model.
draft: false
tags:
  - OpenAI
  - Audio Processing
  - API
  - Machine Learning
---

# Audio Support in OpenAI's Chat Completions API

OpenAI has recently introduced audio support in their Chat Completions API, opening up exciting new possibilities for developers working with audio and text interactions. This feature is powered by the new `gpt-4o-audio-preview` model, which brings advanced voice capabilities to the familiar Chat Completions API interface.

<!-- more -->

## Key Features

The new audio support in the Chat Completions API offers several compelling features:

1. **Flexible Input Handling**: The API can now process any combination of text and audio inputs, allowing for more versatile applications.

2. **Natural, Steerable Voices**: Similar to the Realtime API, developers can use prompting to shape various aspects of the generated audio, including language, pronunciation, and emotional range.

3. **Tool Calling Integration**: The audio support seamlessly integrates with existing tool calling functionality, enabling complex workflows that combine audio, text, and external tools.

## Practical Example

To demonstrate how to use this new functionality, let's look at a simple example using the `instructor` library:

```python
from pydantic import BaseModel
import instructor
from instructor.processing.multimodal import Audio

client = instructor.from_provider("openai/gpt-5-nano")


class Person(BaseModel):
    name: str
    age: int


resp = client.create(
    model="gpt-4o-audio-preview",
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
# Expected output: Person(name='Jason', age=20)
```

In this example, we're using the `gpt-4o-audio-preview` model to extract information from an audio file. The API processes the audio input and returns structured data (a Person object with name and age) based on the content of the audio.

## Use Cases

The addition of audio support to the Chat Completions API enables a wide range of applications:

1. **Voice-based Personal Assistants**: Create more natural and context-aware voice interfaces for various applications.

2. **Audio Content Analysis**: Automatically extract information, sentiments, or key points from audio recordings or podcasts.

3. **Language Learning Tools**: Develop interactive language learning applications that can process and respond to spoken language.

4. **Accessibility Features**: Improve accessibility in applications by providing audio-based interactions and text-to-speech capabilities.

## Considerations

While this new feature is exciting, it's important to note that it's best suited for asynchronous use cases that don't require extremely low latencies. For more dynamic and real-time interactions, OpenAI recommends using their Realtime API.

As with any AI-powered feature, it's crucial to consider ethical implications and potential biases in audio processing and generation. Always test thoroughly and consider the diversity of your user base when implementing these features.

## Related Documentation
- [Multimodal Guide](../../concepts/multimodal.md) - Comprehensive multimodal reference
- [OpenAI Integration](../../integrations/openai.md) - Full OpenAI setup

## See Also
- [Gemini Multimodal](multimodal-gemini.md) - Alternative multimodal approach
- [Prompt Caching](anthropic-prompt-caching.md) - Cache large audio files
- [Monitoring with Logfire](logfire.md) - Track multimodal processing
