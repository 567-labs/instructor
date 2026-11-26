---
authors:
  - jxnl
categories:
  - instructor
comments: true
date: 2026-11-26
description: Build a Python function that classifies support tickets as billing, access, or other using Pydantic, Instructor, and the OpenAI Decisions API.
draft: true
slug: decisions-api
tags:
  - Instructor
  - OpenAI
  - Pydantic
  - Classification
---

# Build a support ticket classifier with the OpenAI Decisions API

> I can't sign in to download my invoice.

Does this go to billing or access? I'd send it to access. The invoice is why the customer wants to sign in. The login is what they need help with.

Let's write a Python function that takes a message and returns `"billing"`, `"access"`, or `"other"`. We'll use Instructor to turn one Pydantic field into a question for the [OpenAI Decisions API](https://developers.openai.com/api/docs/guides/decisions).

<!-- more -->

## Define the labels and the rule

Use Python 3.10 or later and set `OPENAI_API_KEY` in your environment. The example requires Instructor 1.18.0 or later. Until it's on PyPI, install from GitHub in a virtual environment.

```bash
python -m pip install "instructor @ git+https://github.com/567-labs/instructor.git@main"
```

```python
from typing import Literal

import instructor
from pydantic import BaseModel, Field

Topic = Literal["billing", "access", "other"]


class Ticket(BaseModel):
    topic: Topic = Field(
        description=(
            "What does the customer need help fixing first? "
            "Choose access for sign-in or permission problems, "
            "billing for charges, invoices, or refunds, "
            "and other for everything else."
        )
    )


def classify(message: str) -> Topic:
    with instructor.from_provider(
        "openai/gpt-6-luna", mode=instructor.Mode.DECISIONS
    ) as client:
        ticket = client.create(response_model=Ticket, input=message)
        return ticket.topic


print(classify("I can't sign in to download my invoice."))
```

Instructor sends the field description as the question, the three values in `Literal` as the choices, and the message as input. It validates the answer and returns a `Ticket`. Your function returns the label. A missing answer or an unexpected label raises an error, so `"other"` stays an explicit choice.

## Change a few words and try again

I'd test the classifier with these three messages. Here's how I'd label them by hand.

| Message | Expected label |
| --- | --- |
| I can't sign in to download my invoice. | `access` |
| I signed in and downloaded my invoice. Why was I charged twice? | `billing` |
| Can I change the colors on my dashboard? | `other` |

The first two mention the same things. **“What does the customer need help fixing first?”** is the part of the field description that tells the model how to choose.

For more ways to define labels, see the [typed decisions guide](../../concepts/decisions.md#choices).

Before adding more categories, ask two people on your support team to label the same tickets. Where they disagree, write down the rule they settle on. That's what I'd put in `Field(description=...)`.
