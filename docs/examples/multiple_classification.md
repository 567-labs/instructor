---
title: Multi-Label Classification - Support Ticket Categorization
description: Implement multi-label classification with Instructor for support tickets. Assign multiple categories like ACCOUNT, BILLING, and GENERAL_QUERY simultaneously.
---

For multi-label classification, we introduce a new enum class and a different Pydantic model to handle multiple labels.

```python
import instructor

from typing import List, Literal
from pydantic import BaseModel, Field

# Apply the patch to the OpenAI client
# enables response_model keyword
client = instructor.from_provider("openai/gpt-5-nano")

LABELS = Literal["ACCOUNT", "BILLING", "GENERAL_QUERY"]


class MultiClassPrediction(BaseModel):
    """
    A few-shot example of multi-label classification:
    Examples:
    - "My account is locked and I can't access my billing info.": ACCOUNT, BILLING
    - "I need help with my subscription.": ACCOUNT
    - "How do I change my payment method?": BILLING
    - "Can you tell me the status of my order?": BILLING
    - "I have a question about the product features.": GENERAL_QUERY
    """

    labels: List[LABELS] = Field(
        ...,
        description="Only select the labels that apply to the support ticket.",
    )


def multi_classify(data: str) -> MultiClassPrediction:
    return client.create(
        model="gpt-4o-mini",
        response_model=MultiClassPrediction,
        messages=[
            {
                "role": "system",
                "content": f"You are a support agent at a tech company. Only select the labels that apply to the support ticket.",
            },
            {
                "role": "user",
                "content": f"Classify the following support ticket: <text>{data}</text>",
            },
        ],
    )  # type: ignore


if __name__ == "__main__":
    ticket = "My account is locked and I can't access my billing info."
    prediction = multi_classify(ticket)
    assert {"ACCOUNT", "BILLING"} == {label for label in prediction.labels}
    print("input:", ticket)
    #> input: My account is locked and I can't access my billing info.
    print("labels:", LABELS)
    #> labels: typing.Literal['ACCOUNT', 'BILLING', 'GENERAL_QUERY']
    print("prediction:", prediction)
    #> prediction: labels=['ACCOUNT', 'BILLING']
```
