---
authors:
  - jxnl
categories:
  - LLM
  - Pydantic
comments: true
date: 2024-10-23
description: Learn how to use Instructor and Pydantic to create an LLM-based reranker for improving search results relevance.
draft: false
tags:
  - LLM
  - Pydantic
  - Instructor
  - Search Relevance
  - Reranking
---

# Building an LLM-based Reranker for your RAG pipeline

Are you struggling with irrelevant search results in your Retrieval-Augmented Generation (RAG) pipeline?

Imagine having a powerful tool that can intelligently reassess and reorder your search results, significantly improving their relevance to user queries.

In this blog post, we'll show you how to create an LLM-based reranker using Instructor and Pydantic. This approach will:

- Enhance the accuracy of your search results
- Leverage the power of large language models (LLMs)
- Utilize structured outputs for precise information retrieval

By the end of this tutorial, you'll be able to implement a llm reranker to label your synthetic data for fine-tuning a traditional reranker, or to build out an evaluation pipeline for your RAG system. Let's dive in!

<!-- more -->

## Setting Up the Environment

First, let's set up our environment with the necessary imports:

```python
import instructor

client = instructor.from_provider("openai/gpt-5-nano")
```

We're using the `instructor` library, which integrates seamlessly with OpenAI's API and Pydantic for structured outputs.

## Defining the Reranking Models

We'll use Pydantic to define our `Label` and `RerankedResults` models that structure the output of our LLM:

Notice that not only do I reference the chunk_id in the label class, I also asked a language model to use chain of thought. This is very useful for using models like 4o Mini or Claude, but not necessarily if we plan to use the `o1-mini` and `o1-preview` models.

```python
class Label(BaseModel):
    chunk_id: int = Field(description="The unique identifier of the text chunk")
    chain_of_thought: str = Field(
        description="The reasoning process used to evaluate the relevance"
    )
    relevancy: int = Field(
        description="Relevancy score from 0 to 10, where 10 is most relevant",
        ge=0,
        le=10,
    )


class RerankedResults(BaseModel):
    labels: list[Label] = Field(description="List of labeled and ranked chunks")

    @field_validator("labels")
    @classmethod
    def model_validate(cls, v: list[Label]) -> list[Label]:
        return sorted(v, key=lambda x: x.relevancy, reverse=True)
```

These models ensure that our LLM's output is structured and includes a list of labeled chunks with their relevancy scores. The `RerankedResults` model includes a validator that automatically sorts the labels by relevancy in descending order.

## Creating the Reranker Function

Next, we'll create a function that uses our LLM to rerank a list of text chunks based on their relevance to a query:

```python
def rerank_results(query: str, chunks: list[dict]) -> RerankedResults:
    return client.create(
        model="gpt-4o-mini",
        response_model=RerankedResults,
        messages=[
            {
                "role": "system",
                "content": """
                You are an expert search result ranker. Your task is to evaluate the relevance of each text chunk to the given query and assign a relevancy score.

                For each chunk:
                1. Analyze its content in relation to the query.
                2. Provide a chain of thought explaining your reasoning.
                3. Assign a relevancy score from 0 to 10, where 10 is most relevant.

                Be objective and consistent in your evaluations.
                """,
            },
            {
                "role": "user",
                "content": """
                <query>{{ query }}</query>

                <chunks_to_rank>
                {% for chunk in chunks %}
                <chunk id="{{ chunk.id }}">
                    {{ chunk.text }}
                </chunk>
                {% endfor %}
                </chunks_to_rank>

                Please provide a RerankedResults object with a Label for each chunk.
                """,
            },
        ],
        context={"query": query, "chunks": chunks},
    )
```

This function takes a query and a list of text chunks as input, sends them to the LLM with a predefined prompt, and returns a structured `RerankedResults` object. Thanks to instructor we can use jinja templating to inject the query and chunks into the prompt by passing in the `context` parameter.

## Testing the Reranker

To test our LLM-based reranker, we can create a sample query and a list of text chunks. Here's an example of how to use the reranker:

```python
def main():
    query = "What are the health benefits of regular exercise?"
    chunks = [
        {
            "id": 0,
            "text": "Regular exercise can improve cardiovascular health and reduce the risk of heart disease.",
        },
        {
            "id": 1,
            "text": "The price of gym memberships varies widely depending on location and facilities.",
        },
        {
            "id": 2,
            "text": "Exercise has been shown to boost mood and reduce symptoms of depression and anxiety.",
        },
        {
            "id": 3,
            "text": "Proper nutrition is essential for maintaining a healthy lifestyle.",
        },
        {
            "id": 4,
            "text": "Strength training can increase muscle mass and improve bone density, especially important as we age.",
        },
    ]

    results = rerank_results(query, chunks)

    print("Reranked results:")
    for label in results.labels:
        print(f"Chunk {label.chunk_id} (Relevancy: {label.relevancy}):")
        print(f"Text: {chunks[label.chunk_id]['text']}")
        print(f"Reasoning: {label.chain_of_thought}")
        print()


if __name__ == "__main__":
    main()
```

This test demonstrates how the reranker evaluates and sorts the chunks based on their relevance to the query. The full implementation can be found in the `examples/reranker/run.py` file.

If you want to extend this example, you could use the `rerank_results` function to label synthetic data for fine-tuning a traditional reranker, or to build out an evaluation pipeline for your RAG system.

Moreover, we could also add validators to the `Label.chunk_id` field to ensure that the chunk_id is present in the `chunks` list. This might be useful if labels are `uuids` or complex strings and we want to ensure that the chunk_id is a valid index for the chunks list.

heres an example

```python
class Label(BaseModel):
    chunk_id: int = Field(description="The unique identifier of the text chunk")
    ...

    @field_validator("chunk_id")
    @classmethod
    def validate_chunk_id(cls, v: int, info: ValidationInfo) -> int:
        context = info.context
        chunks = context["chunks"]
        if v not in [chunk["id"] for chunk in chunks]:
            raise ValueError(
                f"Chunk with id {v} not found, must be one of {[chunk['id'] for chunk in chunks]}"
            )
        return v
```

This will automatically check that the `chunk_id` is present in the `chunks` list and raise a `ValueError` if it is not, where `context` is the context dictionary that we passed into the `rerank_results` function.

## See Also
- [RAG and Beyond](rag-and-beyond.md) - Comprehensive RAG guide
- [Validation Fundamentals](validation-part1.md) - Validate ranking scores
- [Performance Monitoring](logfire.md) - Track reranking performance
