---
title: "Select Effective Examples"
description: "KNN can be leveraged to choose the most effective examples to use for a given query."
---

We can select effective in-context examples by choosing those that are semantically closer to the query using `KNN`.

In the below implementation using `instructor`, we follow these steps:

1. Embed the query examples
2. Embed the query that we want to answer
3. Find the _k_ query examples closest to the query
4. Use the chosen examples and their as the context for the LLM

```python
import instructor
from pydantic import BaseModel
from openai import OpenAI
import math
from textwrap import dedent


class Example(BaseModel):
    question: str
    answer: str


class Response(BaseModel):
    answer: str


oai = OpenAI()
client = instructor.from_provider("openai/gpt-4o")


def distance(a: list[float], b: list[float]):
    return 1 - sum(ai * bi for ai, bi in zip(a, b)) / (
        math.sqrt(sum(ai**2 for ai in a)) * math.sqrt(sum(bi**2 for bi in b))
    )


def embed_queries(queries: list[str]) -> list[tuple[list[float], str]]:
    return [
        (embedding_item.embedding, query)
        for embedding_item, query in zip(
            oai.embeddings.create(input=queries, model="text-embedding-3-large").data,
            queries,
        )
    ]


def knn(
    embedded_examples: list[tuple[list[float], str]],
    query_embedding: list[float],
    k: int,
):
    distances = [
        (distance(embedding, query_embedding), example)
        for embedding, example in embedded_examples
    ]
    distances.sort(key=lambda x: x[0])
    return distances[:k]


def generate_response(examples: list[str], query: str):
    formatted_examples = "\n".join(examples)
    return client.create(
        model="gpt-4o",
        response_model=Response,
        messages=[
            {
                "role": "user",
                "content": dedent(
                    f"""
                    Respond to the following query with the most accurate
                    and concise answer possible.
                    <examples>
                    {formatted_examples}
                    </examples>
                    <query>
                    {query}
                    </query>
                """
                ),
            }
        ],
    )


def generate_question_and_answer_pair(
    questions: list[str], question_and_answers: list[dict[str, str]]
) -> list[str]:
    question_to_answer = {}

    for question in question_and_answers:
        question_to_answer[question["question"]] = question["answer"]

    return [
        dedent(
            f"""
        <example>
        <question>{question}</question>
        <answer>{question_to_answer[question]}</answer>
        </example>
        """
        )
        for question in questions
    ]


if __name__ == "__main__":
    examples = [
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "Who wrote Romeo and Juliet", "answer": "Shakespeare"},
        {"question": "What is the capital of Germany?", "answer": "Berlin"},
    ]

    query = "What is the capital of Italy?"

    # Step 1 : Embed the Examples
    embeddings = embed_queries([example["question"] for example in examples] + [query])

    embedded_examples = embeddings[:-1]
    embedded_query = embeddings[-1]

    # # Step 3: Find the k closest examples to the query
    k_closest_examples = knn(embedded_examples, embedded_query[0], 2)

    for example in k_closest_examples:
        print(example)
        #> (0.4013468481736857, 'What is the capital of France?')
        #> (0.4471368596136872, 'What is the capital of Germany?')

    # Step 4: Use these examples as in-context examples
    formatted_examples = generate_question_and_answer_pair(
        [example[1] for example in k_closest_examples], examples
    )
    response = generate_response(formatted_examples, query)
    print(response.answer)
    #> Rome
```

### References

<sup id="ref-1">1</sup>: [What Makes Good In-Context Examples for GPT-3?](https://arxiv.org/abs/2101.06804)

<sup id="ref-asterisk">\*</sup>: [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608)
