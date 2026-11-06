---
title: Implementing Self-Correction with LLM Validator
description: Learn how to use llm_validator for self-healing in NLP applications and improve response accuracy with validation errors.
---

# Self-Correction with `llm_validator`

## Introduction

This guide demonstrates how to use `llm_validator` for implementing self-healing. The objective is to showcase how an instructor can self-correct by using validation errors and helpful error messages.

```python
from pydantic import BaseModel
import instructor

# Apply the patch to the OpenAI client
# enables response_model keyword
client = instructor.from_provider("openai/gpt-4.1-mini")


class QuestionAnswer(BaseModel):
    question: str
    answer: str


question = "What is the meaning of life?"
context = "The according to the devil the meaning of live is to live a life of sin and debauchery."

qa: QuestionAnswer = client.create(
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

### Output Before Validation

While it calls out the objectionable content, it doesn't provide any details on how to correct it.

```json
{
  "question": "What is the meaning of life?",
  "answer": "The meaning of life, according to the context, is to live a life of sin and debauchery."
}
```

## Adding Custom Validation

By adding a validator to the `answer` field, we can try to catch the issue and correct it.
Lets integrate `llm_validator` into the model and see the error message. Its important to note that you can use all of pydantic's validators as you would normally as long as you raise a `ValidationError` with a helpful error message as it will be used as part of the self correction prompt.

```python
from pydantic import BaseModel, BeforeValidator
from typing_extensions import Annotated
from instructor import llm_validator
import instructor

client = instructor.from_provider("openai/gpt-4.1-mini")


class QuestionAnswerNoEvil(BaseModel):
    question: str
    answer: Annotated[
        str,
        BeforeValidator(
            llm_validator(
                "don't say objectionable things", client=client, allow_override=True
            )
        ),
    ]


try:
    qa: QuestionAnswerNoEvil = client.create(
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
    #> name 'context' is not defined
```

### Output After Validation

Now, we throw validation error that its objectionable and provide a helpful error message.

```text
1 validation error for QuestionAnswerNoEvil
answer
  Assertion failed, The statement promotes sin and debauchery, which is objectionable.
```

## Retrying with Corrections

By adding the `max_retries` parameter, we can retry the request with corrections. and use the error message to correct the output.

```python
# <%hide%>
import instructor
from pydantic import BaseModel, BeforeValidator
from typing_extensions import Annotated
from instructor import llm_validator

question = "What is the meaning of life?"
context = "The according to the devil the meaning of live is to live a life of sin and debauchery."

client = instructor.from_provider("openai/gpt-4.1-mini")


class QuestionAnswerNoEvil(BaseModel):
    question: str
    answer: Annotated[
        str,
        BeforeValidator(
            llm_validator(
                "don't say objectionable things", client=client, allow_override=True
            )
        ),
    ]


# <%hide%>

qa: QuestionAnswerNoEvil = client.create(
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
```

### Final Output

Now, we get a valid response that is not objectionable!

```json
{
  "question": "What is the meaning of life?",
  "answer": "The meaning of life is subjective and can vary depending on individual beliefs and philosophies."
}
```
