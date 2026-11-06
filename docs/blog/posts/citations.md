---
authors:
- jxnl
categories:
- Pydantic
comments: true
date: 2023-11-18
description: Explore how Pydantic enhances LLM citation verification, improving data
  accuracy and reliability in responses.
draft: false
slug: validate-citations
tags:
- Pydantic
- LLM
- Data Accuracy
- Citation Verification
- Python
---

# Verifying LLM Citations with Pydantic

Ensuring the accuracy of information is crucial. This blog post explores how Pydantic's powerful and flexible validators can enhance data accuracy through citation verification.

We'll start with using a simple substring check to verify citations. Then we'll use `instructor` itself to power an LLM to verify citations and align answers with the given citations. Finally, we'll explore how we can use these techniques to generate a dataset of accurate responses.

<!-- more -->

## Example 1: Simple Substring Check

In this example, we use the `Statements` class to verify if a given substring quote exists within a text chunk. If the substring is not found, an error is raised.

### Code Example:

```python
from typing import List
from pydantic import BaseModel, ValidationInfo, field_validator
import instructor

client = instructor.from_provider("openai/gpt-5-nano")


class Statements(BaseModel):
    body: str
    substring_quote: str

    @field_validator("substring_quote")
    @classmethod
    def substring_quote_exists(cls, v: str, info: ValidationInfo):
        context = info.context.get("text_chunks", None)

        for text_chunk in context.values():
            if v in text_chunk:  # (1)
                return v
        raise ValueError("Could not find substring_quote `{v}` in contexts")


class AnswerWithCitaton(BaseModel):
    question: str
    answer: List[Statements]
```

1. While we use a simple substring check in this example, we can use more complex techniques like regex or Levenshtein distance.

Once the class is defined, we can use it to validate the context and raise an error if the substring is not found.

```python
try:
    AnswerWithCitaton.model_validate(
        {
            "question": "What is the capital of France?",
            "answer": [
                {"body": "Paris", "substring_quote": "Paris is the capital of France"},
            ],
        },
        context={
            "text_chunks": {
                1: "Jason is a pirate",
                2: "Paris is not the capital of France",
                3: "Irrelevant data",
            }
        },
    )
except ValidationError as e:
    print(e)
```

### Error Message Example:

```
answer.0.substring_quote
  Value error, Could not find substring_quote `Paris is the capital of France` in contexts [type=value_error, input_value='Paris is the capital of France', input_type=str]
    For further information visit [https://errors.pydantic.dev/2.4/v/value_error](https://errors.pydantic.dev/2.4/v/value_error)
```

Pydantic raises a validation error when the `substring_quote` attribute does not exist in the context. This approach can be used to validate more complex data using techniques like regex or Levenshtein distance.

## Example 2: Using LLM for Verification

This approach leverages OpenAI's LLM to validate citations. If the citation does not exist in the context, the LLM returns an error message.

### Code Example:

```python
class Validation(BaseModel):
    is_valid: bool
    error_messages: Optional[str] = Field(None, description="Error messages if any")


class Statements(BaseModel):
    body: str
    substring_quote: str

    @model_validator(mode="after")
    def substring_quote_exists(self, info: ValidationInfo):
        context = info.context.get("text_chunks", None)

        resp: Validation = client.create(
            response_model=Validation,
            messages=[
                {
                    "role": "user",
                    "content": f"Does the following citation exist in the following context?\n\nCitation: {self.substring_quote}\n\nContext: {context}",
                }
            ],
            model="gpt-5.4-mini",
        )

        if resp.is_valid:
            return self

        raise ValueError(resp.error_messages)


class AnswerWithCitaton(BaseModel):
    question: str
    answer: List[Statements]
```

Now when we use a correct citation, the LLM returns a valid response.

```python
resp = AnswerWithCitaton.model_validate(
    {
        "question": "What is the capital of France?",
        "answer": [
            {"body": "Paris", "substring_quote": "Paris is the capital of France"},
        ],
    },
    context={
        "text_chunks": {
            1: "Jason is a pirate",
            2: "Paris is the capital of France",
            3: "Irrelevant data",
        }
    },
)
print(resp.model_dump_json(indent=2))
```

### Result:

```json
{
  "question": "What is the capital of France?",
  "answer": [
    {
      "body": "Paris",
      "substring_quote": "Paris is the capital of France"
    }
  ]
}
```

When we have citations that don't exist in the context, the LLM returns an error message.

```python
try:
    AnswerWithCitaton.model_validate(
        {
            "question": "What is the capital of France?",
            "answer": [
                {"body": "Paris", "substring_quote": "Paris is the capital of France"},
            ],
        },
        context={
            "text_chunks": {
                1: "Jason is a pirate",
                2: "Paris is not the capital of France",
                3: "Irrelevant data",
            }
        },
    )
except ValidationError as e:
    print(e)
```

### Error Message Example:

```
1 validation error for AnswerWithCitaton
answer.0
  Value error, Citation not found in context [type=value_error, input_value={'body': 'Paris', 'substr... the capital of France'}, input_type=dict]
    For further information visit [https://errors.pydantic.dev/2.4/v/value_error](https://errors.pydantic.dev/2.4/v/value_error)
```

## Example 3: Aligning Citations and Answers

In this example, we ensure that the provided answers are aligned with the given citations and context. The LLM is used to verify the alignment.

We use the same `Statements` model as above, but we add a new model for the answer that also verifies the alignment of citations.

### Code Example:

```python
class AnswerWithCitaton(BaseModel):
    question: str
    answer: List[Statements]

    @model_validator(mode="after")
    def validate_answer(self, info: ValidationInfo):
        context = info.context.get("text_chunks", None)

        resp: Validation = client.create(
            response_model=Validation,
            messages=[
                {
                    "role": "user",
                    "content": f"Does the following answers match the question and the context?\n\nQuestion: {self.question}\n\nAnswer: {self.answer}\n\nContext: {context}",
                }
            ],
            model="gpt-5.4-mini",
        )

        if resp.is_valid:
            return self

        raise ValueError(resp.error_messages)
```

When we have a mismatch between the answer and the citation, the LLM returns an error message.

```python
try:
    AnswerWithCitaton.model_validate(
        {
            "question": "What is the capital of France?",
            "answer": [
                {"body": "Texas", "substring_quote": "Paris is the capital of France"},
            ],
        },
        context={
            "text_chunks": {
                1: "Jason is a pirate",
                2: "Paris is the capital of France",
                3: "Irrelevant data",
            }
        },
    )
except ValidationError as e:
    print(e)
```

### Error Message Example:

```
1 validation error for AnswerWithCitaton
  Value error, The answer does not match the question and context [type=value_error, input_value={'question': 'What is the...he capital of France'}]}, input_type=dict]
    For further information visit [https://errors.pydantic.dev/2.4/v/value_error](https://errors.pydantic.dev/2.4/v/value_error)
```

## Related Documentation
- [Validation Guide](../../concepts/validation.md) - Validate citations

## See Also
- [RAG Techniques](rag-and-beyond.md) - Use citations in RAG
- [PDF Citations](generating-pdf-citations.md) - Extract from PDFs
- [Validation Basics](validation-part1.md) - Ensure citation quality

## Conclusion

These examples demonstrate the potential of using Pydantic and OpenAI to enhance data accuracy through citation verification. While the LLM-based approach may not be efficient for runtime operations, it has exciting implications for generating a dataset of accurate responses. By leveraging this method during data generation, we can fine-tune a model that excels in citation accuracy. Similar to our last post on [finetuning a better summarizer](https://jxnl.github.io/instructor/blog/2023/11/05/chain-of-density/).

If you like the content check out our [GitHub](https://github.com/jxnl/instructor) as give us a star and checkout the library.