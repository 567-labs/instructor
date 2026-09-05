---
title: "Role Prompting"
description: "Role prompting, or persona prompting, assigns a role to the model."
---

How can we increase a model's performance on open-ended tasks?

Role prompting, or persona prompting, assigns a role to the model. Roles can be:

 - **specific to the query**: *You are a talented writer. Write me a poem.*
 - **general/social**: *You are a helpful AI assistant. Write me a poem.*

## Implementation

```python hl_lines="27"
import openai
import instructor
from pydantic import BaseModel

client = instructor.from_provider("openai/gpt-5-nano")


class Response(BaseModel):
    poem: str


def role_prompting(query, role):
    return client.create(
        model="gpt-4o",
        response_model=Response,
        messages=[
            {
                "role": "system",
                "content": f"{role} {query}",
            },
        ],
    )


if __name__ == "__main__":
    query = "Write me a short poem about coffee."
    role = "You are a renowned poet."

    response = role_prompting(query, role)
    print(response.poem)
    """
    In the morning's gentle light,
    A brew of warmth, dark and bright.
    Awakening dreams, so sweet,
    In every sip, the day we greet.

    Through the steam, stories spin,
    A liquid muse, caffeine within.
    Moments pause, thoughts unfold,
    In coffee's embrace, we find our gold.
    """
```

!!! info "More Role Prompting"
    To read about a systematic approach to choosing roles, check out [RoleLLM](https://arxiv.org/abs/2310.00746).

    For more examples of social roles, check out [this](https://arxiv.org/abs/2311.10054) evaluation of social roles in system prompts..

    To read about using more than one role, check out [Multi-Persona Self-Collaboration](https://arxiv.org/abs/2307.05300).

## References

<sup id="ref-1">1</sup>: [RoleLLM: Benchmarking, Eliciting, and Enhancing Role-Playing Abilities of Large Language Models](https://arxiv.org/abs/2310.00746)
<sup id="ref-2">2</sup>: [Is "A Helpful Assistant" the Best Role for Large Language Models? A Systematic Evaluation of Social Roles in System Prompts ](https://arxiv.org/abs/2311.10054)
<sup id="ref-4">3</sup>: [Unleashing the Emergent Cognitive Synergy in Large Language Models: A Task-Solving Agent through Multi-Persona Self-Collaboration ](https://arxiv.org/abs/2307.05300)
