---
title: "Style Prompting"
description: "To constrain a model's response to fit the boundaries of our task, we can specify a style."
---

How can we constrain model outputs through prompting alone?

To constrain a model's response to fit the boundaries of our task, we can specify a style.

Stylistic constraints can include:

 - **writing style**: write a *flowery* poem
 - **tone**: write a *dramatic* poem
 - **mood**: write a *happy* poem
 - **genre**: write a *mystery* poem

## Implementation

```python hl_lines="22"
import instructor
from pydantic import BaseModel
import openai


class Email(BaseModel):
    subject: str
    message: str


client = instructor.from_provider("openai/gpt-5-nano")


def generate_email(subject, to, sender, tone):
    return client.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": f"""
                Write an email about {subject} to {to} from {sender}.
                The email should be {tone}.
                """,
            }
        ],
        response_model=Email,
    )


if __name__ == "__main__":
    email = generate_email(
        subject="invitation to all-hands on Monday at 6pm",
        to="John Smith",
        sender="Jane Doe",
        tone="formal",
    )

    print(email.subject)
    #> Invitation to All-Hands Meeting
    print(email.message)
    """
    Dear Mr. Smith,

    I hope this message finds you well. I am writing to formally invite you to our upcoming all-hands meeting scheduled for Monday at 6:00 PM. This meeting is an important opportunity for us to come together, discuss key updates, and align on our strategic goals.

    Please confirm your availability at your earliest convenience. Your presence and contributions to the discussion would be greatly valued.

    Thank you and I look forward to your confirmation.

    Warm regards,

    Jane Doe
    """
```

## Stylistic Constraint Examples

| Constraint     | Possible Phrases                                                                  |
|----------------|-----------------------------------------------------------------------------------|
| Writing Style  | Functional, Flowery, Candid, Prosaic, Ornate, Poetic                              |
| Tone           | Dramatic, Humorous, Optimistic, Sad, Formal, Informal                             |
| Mood           | Angry, Fearful, Happy, Sad                                                        |
| Genre          | Historical Fiction, Literary Fiction, Science Fiction, Mystery, Dystopian, Horror |

!!! info "More Stylistic Constraints"

    To see even more examples of these stylistic constraints and additional constraints (**characterization**, **pacing**, and **plot**), check out [this](https://arxiv.org/abs/2302.09185) paper.

## References

<sup id="ref-1">1</sup>: [Bounding the Capabilities of Large Language Models in Open Text Generation with Prompt Constraints](https://arxiv.org/abs/2302.09185)

