---
authors:
  - jxnl
categories:
  - Anthropic
  - LLM Techniques
  - Python
comments: true
date: 2024-09-26
description:
  Learn to implement Anthropic's Contextual Retrieval with async processing
  to enhance RAG systems and preserve crucial context efficiently.
draft: false
tags:
  - Contextual Retrieval
  - Async Processing
  - RAG Systems
  - Performance Optimization
  - Document Chunking
---

# Implementing Anthropic's Contextual Retrieval with Async Processing

Anthropic's [Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval) technique enhances RAG systems by preserving crucial context.

This post examines the method and demonstrates an efficient implementation using async processing. We'll explore how to optimize your RAG applications with this approach, building on concepts from our [async processing guide](./learn-async.md).

<!-- more -->

## Background: The Context Problem in RAG

Anthropic identifies a key issue in traditional RAG systems: loss of context when documents are split into chunks. They provide an example:

"Imagine you had a collection of financial information (say, U.S. SEC filings) embedded in your knowledge base, and you received the following question: 'What was the revenue growth for ACME Corp in Q2 2023?'

A relevant chunk might contain the text: 'The company's revenue grew by 3% over the previous quarter.' However, this chunk on its own doesn't specify which company it's referring to or the relevant time period."

## Anthropic's Solution: Contextual Retrieval

Contextual Retrieval solves this by adding chunk-specific explanatory context before embedding. Anthropic's example:

```
original_chunk = "The company's revenue grew by 3% over the previous quarter."

contextualized_chunk = "This chunk is from an SEC filing on ACME corp's performance in Q2 2023; the previous quarter's revenue was $314 million. The company's revenue grew by 3% over the previous quarter."
```

## Implementing Contextual Retrieval

Anthropic uses Claude to generate context. They provide this prompt:

```
<document>
{{WHOLE_DOCUMENT}}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{{CHUNK_CONTENT}}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
```

## Performance Improvements

Anthropic reports significant improvements:

- Contextual Embeddings reduced top-20-chunk retrieval failure rate by 35% (5.7% → 3.7%).
- Combining Contextual Embeddings and Contextual BM25 reduced failure rate by 49% (5.7% → 2.9%).
- Adding reranking further reduced failure rate by 67% (5.7% → 1.9%).

## Instructor implementation of Contextual Retrieval with Async Processing

We can implement Anthropic's technique using async processing for improved efficiency:

```python
from instructor import AsyncInstructor, Mode, patch
from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field
import asyncio
from typing import List, Dict


class SituatedContext(BaseModel):
    title: str = Field(..., description="The title of the document.")
    context: str = Field(
        ..., description="The context to situate the chunk within the document."
    )


client = AsyncInstructor(
    create=patch(
        create=AsyncAnthropic().beta.prompt_caching.messages.create,
        mode=Mode.TOOLS,
    ),
    mode=Mode.TOOLS,
)


async def situate_context(doc: str, chunk: str) -> str:
    response = await client.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "<document>{{doc}}</document>",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": "Here is the chunk we want to situate within the whole document\n<chunk>{{chunk}}</chunk>\nPlease give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.\nAnswer only with the succinct context and nothing else.",
                    },
                ],
            }
        ],
        response_model=SituatedContext,
        context={"doc": doc, "chunk": chunk},
    )
    return response.context


def chunking_function(doc: str) -> List[str]:
    chunk_size = 1000
    overlap = 200
    chunks = []
    start = 0
    while start < len(doc):
        end = start + chunk_size
        chunks.append(doc[start:end])
        start += chunk_size - overlap
    return chunks


async def process_chunk(doc: str, chunk: str) -> Dict[str, str]:
    context = await situate_context(doc, chunk)
    return {"chunk": chunk, "context": context}


async def process(doc: str) -> List[Dict[str, str]]:
    chunks = chunking_function(doc)
    tasks = [process_chunk(doc, chunk) for chunk in chunks]
    results = await asyncio.gather(*tasks)
    return results


# Example usage
async def main():
    document = "Your full document text here..."
    processed_chunks = await process(document)
    for i, item in enumerate(processed_chunks):
        print(f"Chunk {i + 1}:")
        print(f"Text: {item['chunk'][:50]}...")
        print(f"Context: {item['context']}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
```

## Key Features of This Implementation

1. Async Processing: Uses `asyncio` for concurrent chunk processing.
2. Structured Output: Uses Pydantic models for type-safe responses.
3. Prompt Caching: Utilizes Anthropic's prompt caching for efficiency.
4. Chunking: Implements a basic chunking strategy with overlap.
5. Jinja2 templating: Uses Jinja2 templating to inject variables into the prompt.

## Considerations from Anthropic's Article

Anthropic mentions several implementation considerations:

1. Chunk boundaries: Experiment with chunk size, boundary, and overlap.
2. Embedding model: They found Gemini and Voyage embeddings effective.
3. Custom contextualizer prompts: Consider domain-specific prompts.
4. Number of chunks: They found using 20 chunks most effective.
5. Evaluation: Always run evaluations on your specific use case.

## Further Enhancements

Based on Anthropic's suggestions:

1. Implement dynamic chunk sizing based on content complexity.
2. Integrate with vector databases for efficient storage and retrieval.
3. Add error handling and retry mechanisms.
4. Experiment with different embedding models and prompts.
5. Implement a reranking step for further performance improvements.

This implementation provides a starting point for leveraging Anthropic's Contextual Retrieval technique with the added efficiency of async processing.
