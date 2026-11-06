---
authors:
- jxnl
categories:
- Pydantic
comments: true
date: 2024-09-07
description: Explore how Pydantic enhances structured outputs in LLM applications,
  ensuring reliability and improved data management.
draft: false
slug: pydantic-is-still-all-you-need
tags:
- Pydantic
- Structured Outputs
- Data Validation
- LLM Techniques
- Performance Optimization
---

# Pydantic is Still All You Need: Reflections on a Year of Structured Outputs

A year ago, I gave a talk titled "Pydantic: All You Need" that kickstarted my Twitter career. Today, I'm back to reaffirm that message and share what I've learned in the past year about using structured outputs with language models.

[Watch the youtube video](https://www.youtube.com/watch?v=pZ4DIH2BVqg){ .md-button .md-button--primary }

<!-- more -->

## The Problem with Unstructured Outputs

Imagine hiring an intern to write an API that returns a string you have to JSON load into a dictionary and pray the data is still there. You'd probably fire them and replace them with GPT. Yet, many of us are content using LLMs in the same haphazard way.

By not using schemas and structured responses, we lose compatibility, composability, and reliability when building tools that interact with external systems. But there's a better way.

## The Power of Pydantic

Pydantic, combined with function calling, offers a superior alternative for structured outputs. It allows for:

- Nested objects and models for modular structures
- Validators to improve system reliability
- Cleaner, more maintainable code

For more details on how Pydantic enhances data validation, check out our [Data Validation with Pydantic](../../concepts/models.md) guide.

And here's the kicker: nothing's really changed in the past year. The core API is still just:

```python
from instructor import from_openai

client = from_openai(OpenAI())

response = client.create(model="gpt-5.4-mini", response_model=User, messages=[...])
```

## What's New in Pydantic?

Since last year:

- We've released version 1.0
- Launched in 5 languages (Python, TypeScript, Ruby, Go, Elixir)
- Built a version in Rust
- Seen 40% month-over-month growth in the Python library

We now support [Ollama](../../integrations/ollama.md), [llama-cpp-python](../../integrations/llama-cpp-python.md), [Anthropic](../../integrations/anthropic.md), [Cohere](../../integrations/cohere.md), [Google](../../integrations/google.md), [Vertex AI](../../integrations/vertex.md), and more. As long as language models support function calling capabilities, this API will remain standard.

## Key Features

1. **Streaming with Structure**: Get objects as they return, improving latency while maintaining structured output. Learn more about this in our [Streaming Support](../../concepts/partial.md) guide.

2. **Partials**: Validate entire objects, enabling real-time rendering for generative UI without complex JSON parsing. See our [Partial](../../concepts/partial.md) documentation for implementation details.

3. **Validators**: Add custom logic to ensure correct outputs, with the ability to retry on errors. Dive deeper into this topic in our [Reasking and Validation](../../concepts/reask_validation.md) guide.

## Real-World Applications

### Generation and Extraction

Structured outputs shine in tasks like:

- Generating follow-up questions in RAG applications
- Validating URLs in generated content
- Extracting structured data from transcripts or images

For a practical example, see our [Structured Data Extraction from Images](../../examples/image_to_ad_copy.md) case study.

### Search Queries

For complex search scenarios:

```python
class Search(BaseModel):
    query: str
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    limit: Optional[int]
    source: Literal["news", "social", "blog"]
```

This structure allows for more sophisticated search capabilities, handling queries like "What is the latest news from X?" that embeddings alone can't handle.

## Lessons Learned

1. Validation errors are crucial for improving system performance.
2. Not all language models support retry logic effectively yet.
3. Structured outputs benefit vision, text, RAG, and agent applications alike.

## The Future of Programming with LLMs

We're not changing the language of programming; we're relearning how to program with data structures. Structured outputs allow us to:

- Own the objects we define
- Control the functions we implement
- Manage the control flow
- Own the prompts

This approach makes Software 3.0 backwards compatible with existing software, demystifying language models and returning us to a more classical programming structure.

## Wrapping Up

Pydantic is still all you need for effective structured outputs with LLMs. It's not just about generating accurate responses; it's about doing so in a way that's compatible with our existing programming paradigms and tools.

As we continue to refine AI language models, keeping these principles in mind will lead to more robust, maintainable, and powerful applications. The future of AI isn't just about what the models can do, but how seamlessly we can integrate them into our existing software ecosystems.

For more advanced use cases and integrations, check out our [examples](../../examples/index.md) section, which covers various LLM providers and specialized implementations.

## Related Documentation
- [Instructor Philosophy](../../concepts/philosophy.md) - Why we chose Pydantic
- [Validation Guide](../../concepts/validation.md) - Practical validation techniques

## See Also
- [Validation Deep Dive](validation-part1.md) - Advanced validation patterns
- [Best Framework Comparison](best_framework.md) - Why Instructor stands out
- [Introduction to Instructor](introduction.md) - Getting started guide
