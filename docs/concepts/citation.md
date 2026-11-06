---
title: Citation Extraction with CitationMixin
description: Learn how to extract and validate citations from source text using CitationMixin to prevent hallucinations.
---

# Citation Extraction with CitationMixin

CitationMixin is a Pydantic mixin that helps extract and validate citations from source text. It ensures that quotes used in your extracted data actually exist in the source context, preventing hallucinations.

## What is CitationMixin?

CitationMixin adds citation validation to your Pydantic models. When you use it, your model gets a `substring_quotes` field that contains quotes from the source text. The mixin automatically validates that these quotes exist in the source and corrects them to match exact spans.

## Basic Usage

Inherit from CitationMixin to add citation support to your model:

```python
from pydantic import BaseModel, Field
from instructor import CitationMixin
import instructor


class User(CitationMixin, BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(description="The age of the person")
    role: str = Field(description="The role of the person")


client = instructor.from_provider("openai/gpt-4o-mini")

context = "Betty was a student. Jason was a student. Jason is 20 years old"

user = client.create(
    response_model=User,
    messages=[
        {
            "role": "user",
            "content": f"Extract information about Jason from: {context}",
        },
    ],
    context={"context": context},
)

# Verify quotes exist in context
for quote in user.substring_quotes:
    assert quote in context

print(user.model_dump())
# {
#     "name": "Jason",
#     "age": 20,
#     "role": "student",
#     "substring_quotes": [
#         "Jason was a student",
#         "Jason is 20 years old",
#     ]
# }
```

## How It Works

CitationMixin works in three steps:

1. **Extraction**: The LLM extracts data and provides quotes in the `substring_quotes` field
2. **Validation**: The mixin checks if each quote exists in the source context using fuzzy matching
3. **Correction**: Quotes are corrected to match exact spans in the source text

The validation happens automatically when you pass `context={"context": source_text}` to your `create()` call.

## Using with Validation Context

CitationMixin uses Pydantic's validation context to access the source text. Pass the source text in the `context` parameter:

```python
from pydantic import BaseModel, Field
from instructor import CitationMixin
import instructor


class Fact(CitationMixin, BaseModel):
    statement: str = Field(description="A factual statement")
    # substring_quotes is added automatically by CitationMixin


client = instructor.from_provider("openai/gpt-4o-mini")

source_text = """
The Eiffel Tower was completed in 1889 and stands 330 meters tall.
It was designed by Gustave Eiffel and is located in Paris, France.
"""

fact = client.create(
    response_model=Fact,
    messages=[
        {
            "role": "user",
            "content": f"Extract facts about the Eiffel Tower from: {source_text}",
        },
    ],
    context={"context": source_text},
)

# All quotes are validated and corrected to exact spans
for quote in fact.substring_quotes:
    print(f"Quote: {quote}")
    assert quote in source_text
```

## Fuzzy Matching

CitationMixin uses fuzzy matching to find quotes even if they don't match exactly. This handles minor differences like:
- Extra whitespace
- Slight wording variations
- Punctuation differences

The matching allows up to 5 character errors by default, which helps handle cases where the LLM paraphrases slightly.

## Advanced Example: Question Answering with Citations

Use CitationMixin to build question-answering systems that cite sources:

```python
from typing import List
from pydantic import BaseModel, Field
from instructor import CitationMixin
import instructor


class Fact(CitationMixin, BaseModel):
    statement: str = Field(description="A factual statement")


class Answer(CitationMixin, BaseModel):
    question: str
    facts: List[Fact] = Field(description="List of facts that answer the question")


client = instructor.from_provider("openai/gpt-4o-mini")

source_text = """
Jason Liu grew up in Toronto, Canada but was born in China.
He went to an arts high school but studied Computational Mathematics and Physics in university.
He worked at Stitchfix and Facebook as part of coop programs.
He started the Data Science club at the University of Waterloo and was president for 2 years.
"""

answer = client.create(
    response_model=Answer,
    messages=[
        {
            "role": "system",
            "content": "Answer questions with exact citations from the source text.",
        },
        {
            "role": "user",
            "content": f"Source: {source_text}\n\nQuestion: What did Jason do during college?",
        },
    ],
    context={"context": source_text},
)

# Verify all citations exist
for fact in answer.facts:
    for quote in fact.substring_quotes:
        assert quote in source_text
        print(f"Verified: {quote}")
```

## When to Use CitationMixin

Use CitationMixin when:

- You need to verify that extracted information comes from source text
- You're building RAG (Retrieval Augmented Generation) systems
- You want to prevent hallucinations by validating citations
- You need exact quote spans for highlighting or display

## Limitations

- Requires passing source text in `context={"context": ...}`
- Uses fuzzy matching which may not catch all paraphrasing
- Only validates quotes, not the accuracy of extracted facts themselves

## See Also

- [Validation](./validation.md) - Learn about validation in Instructor
- [Context-Based Validation](./reask_validation.md#using-context-for-dynamic-validation) - Using context for validation
- [Citation Examples](../examples/exact_citations.md) - More citation examples
- [RAG Patterns](../blog/posts/rag-and-beyond.md) - Building RAG systems with Instructor
