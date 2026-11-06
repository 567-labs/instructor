# Using `llm_validator` with OpenAI's GPT-3.5 Turbo and Pydantic for Text Validation with Output Examples

## Overview

This document outlines how to use a custom text validation logic (`llm_validator`) with OpenAI's GPT-3.5 Turbo and Pydantic, including the outputs for each operation.

## Code Explanation

### Basic Setup

Import necessary modules and apply patches for compatibility.

```python
from typing_extensions import Annotated
from pydantic import (
    BaseModel,
    BeforeValidator,
)
from instructor import llm_validator, patch
import openai

patch()
```

### Defining Response Models

Define a basic Pydantic model named `QuestionAnswer`.

```python
class QuestionAnswer(BaseModel):
    question: str
    answer: str
```

### Generating a Response

Generate a response from GPT-3.5 Turbo.

```python
question = "What is the meaning of life?"
context = "The according to the devil is to live a life of sin and debauchery."

qa: QuestionAnswer = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    response_model=QuestionAnswer,
    messages=[
        {
            "role": "system",
            "content": "You are a system that answers questions based on the context. answer exactly what the question asks using the context.",
        },
        {
            "role": "user",
            "content": f"using the context: {context}\n\nAnswer the following question: {question}",
        },
    ],
)
```

#### Output

Before validation with `llm_validator`:

```json
{
  "question": "What is the meaning of life?",
  "answer": "The meaning of life, according to the context, is to live a life of sin and debauchery."
}
```

### Adding Custom Validation

Add custom validation using `llm_validator`.

```python
class QuestionAnswerNoEvil(BaseModel):
    question: str
    answer: Annotated[
        str,
        BeforeValidator(
            llm_validator("don't say objectionable things", allow_override=True)
        ),
    ]
```

#### Output

```text
1 validation error for QuestionAnswerNoEvil
answer
    Assertion failed, The statement promotes sin and debauchery, which is objectionable.
```

### Handling Validation Errors

Catch exceptions raised by the validation.

```python
try:
    qa: QuestionAnswerNoEvil = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        response_model=QuestionAnswerNoEvil,
        messages=[
            {
                "role": "system",
                "content": "You are a system that answers questions based on the context. answer exactly what the question asks using the context.",
            },
            {
                "role": "user",
                "content": f"using the context: {context}\n\nAnswer the following question: {question}",
            },
        ],
    )
except Exception as e:
    print(e)
```

### Retrying Validation

Allow for retries by setting `max_retries=2`.

```python
qa: QuestionAnswerNoEvil = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    response_model=QuestionAnswerNoEvil,
    max_retries=2,
    messages=[
        {
            "role": "system",
            "content": "You are a system that answers questions based on the context. answer exactly what the question asks using the context.",
        },
        {
            "role": "user",
            "content": f"using the context: {context}\n\nAnswer the following question: {question}",
        },
    ],
)
```

#### Output

After validation with `llm_validator` and `max_retries=2`:

```json
{
  "question": "What is the meaning of life?",
  "answer": "The meaning of life is subjective and can vary depending on individual beliefs and philosophies."
}
```

## Summary

This document described how to use `llm_validator` with OpenAI's GPT-3.5 Turbo and Pydantic, including example outputs. This approach allows for controlled and filtered responses.
