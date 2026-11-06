---
title: "SimToM (Simulated Theory of Mind)"
description: "SimToM (Simulated Theory of Mind) is a two-step prompting technique that encourages a model to consider a specific perspective."
---

How can we encourage the model to focus on relevant information?

SimToM (Simulated Theory of Mind) is a two-step prompting technique that encourages a model to consider a specific perspective.

This can be useful for complex questions with multiple entities. For example, if the prompt contains information about two individuals, we can ask the model to answer our query from the perspective of one of the individuals.

This is implemented in two steps. Given an entity:

1. Identify and isolate information relevant to the entity
2. Ask the model to answer the query from the entity's perspective

!!! example "Sample Template"

    **Step 1**: Given the following context, list the facts that <*entity*> would know. Context: <*context*>

    **Step 2**: You are <*entity*>. Answer the following question based only on these facts you know: <*facts*>. Question: <*query*>

## Implementation

```python hl_lines="24-25"
import openai
import instructor
from pydantic import BaseModel, Field
from typing import Iterable

client = instructor.from_provider("openai/gpt-5-nano")


class KnownFact(BaseModel):
    fact: str = Field(description="A fact that the given entity would know")


class Response(BaseModel):
    location: str


def generate_known_facts(entity, context, query) -> Iterable[KnownFact]:
    return client.create(
        model="gpt-4o",
        response_model=Iterable[KnownFact],
        messages=[
            {
                "role": "user",
                "content": f"""Given the following context, list
                the facts that {entity} would know:

                Context:
                {context}
                {query}

                List only the facts relevant to {entity}.
                """,
            }
        ],
    )


def answer_question_based_on_facts(entity, query, known_facts) -> Response:
    return client.create(
        model="gpt-4o",
        response_model=Response,
        messages=[
            {
                "role": "system",
                "content": f"""You are {entity}. Answer the following question
                based only on these facts you know:
                {" ".join([str(fact) for fact in known_facts])}""",
            },
            {
                "role": "user",
                "content": f"Question: {query}",
            },
        ],
    )


if __name__ == "__main__":
    entity = "Alice"
    context = """Alice puts the book on the table.
        Alice leaves the room.
        Bob moves the book to the shelf.
        """
    query = f"Where does {entity} think the book is?"

    known_facts = generate_known_facts(entity, context, query)
    response = answer_question_based_on_facts(entity, query, known_facts)

    for fact in known_facts:
        print(fact)
        #> fact='Alice puts the book on the table.'
        #> fact='Alice leaves the room. Bob moves the book to the shelf.'
    print(response.location)
    #> On the table
```

## References

<sup id="ref-1">1</sup>: [Think Twice: Perspective-Taking Improves Large Language Models' Theory-of-Mind Capabilities](https://arxiv.org/abs/2311.10227)
