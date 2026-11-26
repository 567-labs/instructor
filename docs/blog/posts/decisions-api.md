---
authors:
  - jxnl
categories:
  - instructor
comments: true
date: 2026-11-26
title: Classify tickets
description: Route tickets to billing, access, or other with the OpenAI Decisions API.
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

Billing or access? I'd pick access. The customer needs help signing in.

Let's write a Python function that takes a message and returns `"billing"`, `"access"`, or `"other"`. We'll use Instructor to turn one Pydantic field into a question for the [OpenAI Decisions API](https://developers.openai.com/api/docs/guides/decisions).

<!-- more -->

## Define the labels and the rule

Set `OPENAI_API_KEY` and use Python 3.10 or later. To try Instructor 1.18 before the PyPI release, install from GitHub.

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

Instructor sends the field description as the question and the [`Literal` values](../../concepts/decisions.md#choices) as the choices. `client.create` returns a validated `Ticket`. `classify` returns its label.

## Change a few words and try again

Here's how I'd label three test cases.

| Message | Expected label |
| --- | --- |
| I can't sign in to download my invoice. | `access` |
| I signed in and downloaded my invoice. Why was I charged twice? | `billing` |
| Can I change the colors on my dashboard? | `other` |

The first two tickets mention signing in and an invoice. In the second, the customer has already signed in. That's why the question asks **what they need help fixing first**.

Try this with a few tickets your team disagrees on. Once you settle on a routing rule, put it in `Field(description=...)` and keep those tickets as test cases.
