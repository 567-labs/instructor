---
title: Text Classification with OpenAI and Pydantic
description: Learn to implement single-label and multi-label text classification using OpenAI API and Pydantic models in Python.
---

# Text Classification using OpenAI and Pydantic

This tutorial showcases how to implement text classification tasks-specifically, single-label and multi-label classifications-using the OpenAI API and Pydantic models. For complete examples, check out our [single classification](./bulk_classification.md) and [multi-label classification](./bulk_classification.md) examples in the cookbook.

!!! tips "Motivation"

    Text classification is a common problem in many NLP applications, such as spam detection or support ticket categorization. The goal is to provide a systematic way to handle these cases using OpenAI's GPT models in combination with Python data structures.

## Single-Label Classification

### Defining the Structures

For single-label classification, we define a Pydantic model with a [Literal](../concepts/prompting.md#literals) field for the possible labels.

!!! note "Literals vs Enums"

    We prefer using `Literal` types over `enum` for classification labels. Literals provide better type checking and are more straightforward to use with Pydantic models.

!!! important "Few-Shot Examples"

    Including few-shot examples in the model's docstring is crucial for improving the model's classification accuracy. These examples guide the AI in understanding the task and expected outputs.

    If you want to learn more prompting tips check out our [prompting guide](../prompting/index.md)

!!! note "Chain of Thought"

    Using [Chain of Thought](../concepts/prompting.md#chain-of-thought) has been shown to improve the quality of the predictions by ~ 10%

```python
from pydantic import BaseModel, Field
from typing import Literal
import instructor

# Apply the patch to the OpenAI client
# enables response_model keyword
client = instructor.from_provider("openai/gpt-5-nano")


class ClassificationResponse(BaseModel):
    """
    A few-shot example of text classification:

    Examples:
    - "Buy cheap watches now!": SPAM
    - "Meeting at 3 PM in the conference room": NOT_SPAM
    - "You've won a free iPhone! Click here": SPAM
    - "Can you pick up some milk on your way home?": NOT_SPAM
    - "Increase your followers by 10000 overnight!": SPAM
    """

    chain_of_thought: str = Field(
        ...,
        description="The chain of thought that led to the prediction.",
    )
    label: Literal["SPAM", "NOT_SPAM"] = Field(
        ...,
        description="The predicted class label.",
    )
```

### Classifying Text

The function **`classify`** will perform the single-label classification.

```python
# <%hide%>
from pydantic import BaseModel, Field
from typing import Literal
import instructor


class ClassificationResponse(BaseModel):
    """
    A few-shot example of text classification:

    Examples:
    - "Buy cheap watches now!": SPAM
    - "Meeting at 3 PM in the conference room": NOT_SPAM
    - "You've won a free iPhone! Click here": SPAM
    - "Can you pick up some milk on your way home?": NOT_SPAM
    - "Increase your followers by 10000 overnight!": SPAM
    """

    chain_of_thought: str = Field(
        ...,
        description="The chain of thought that led to the prediction.",
    )
    label: Literal["SPAM", "NOT_SPAM"] = Field(
        ...,
        description="The predicted class label.",
    )


# Apply the patch to the OpenAI client
# enables response_model keyword
client = instructor.from_provider("openai/gpt-5-nano")


# <%hide%>
def classify(data: str) -> ClassificationResponse:
    """Perform single-label classification on the input text."""
    return client.create(
        model="gpt-4o-mini",
        response_model=ClassificationResponse,
        messages=[
            {
                "role": "user",
                "content": f"Classify the following text: <text>{data}</text>",
            },
        ],
    )
```

### Testing and Evaluation

Let's run examples to see if it correctly identifies spam and non-spam messages.

```python
# <%hide%>
from pydantic import BaseModel, Field
from typing import Literal
import instructor

client = instructor.from_provider("openai/gpt-5-nano")


class ClassificationResponse(BaseModel):
    """
    A few-shot example of text classification:

    Examples:
    - "Buy cheap watches now!": SPAM
    - "Meeting at 3 PM in the conference room": NOT_SPAM
    - "You've won a free iPhone! Click here": SPAM
    - "Can you pick up some milk on your way home?": NOT_SPAM
    - "Increase your followers by 10000 overnight!": SPAM
    """

    chain_of_thought: str = Field(
        ...,
        description="The chain of thought that led to the prediction.",
    )
    label: Literal["SPAM", "NOT_SPAM"] = Field(
        ...,
        description="The predicted class label.",
    )


def classify(data: str) -> ClassificationResponse:
    """Perform single-label classification on the input text."""
    return client.create(
        model="gpt-4o-mini",
        response_model=ClassificationResponse,
        messages=[
            {
                "role": "user",
                "content": f"Classify the following text: <text>{data}</text>",
            },
        ],
    )


# <%hide%>
if __name__ == "__main__":
    for text, label in [
        ("Hey Jason! You're awesome", "NOT_SPAM"),
        ("I am a nigerian prince and I need your help.", "SPAM"),
    ]:
        prediction = classify(text)
        assert prediction.label == label
        print(f"Text: {text}, Predicted Label: {prediction.label}")
        #> Text: Hey Jason! You're awesome, Predicted Label: NOT_SPAM
        #> Text: I am a nigerian prince and I need your help., Predicted Label: SPAM
```

## Multi-Label Classification

### Defining the Structures

For multi-label classification, we'll update our approach to use Literals instead of enums, and include few-shot examples in the model's docstring.

```python
from typing import List
from pydantic import BaseModel, Field
from typing import Literal


class MultiClassPrediction(BaseModel):
    """
    Class for a multi-class label prediction.

    Examples:
    - "My account is locked": ["TECH_ISSUE"]
    - "I can't access my billing info": ["TECH_ISSUE", "BILLING"]
    - "When do you close for holidays?": ["GENERAL_QUERY"]
    - "My payment didn't go through and now I can't log in": ["BILLING", "TECH_ISSUE"]
    """

    chain_of_thought: str = Field(
        ...,
        description="The chain of thought that led to the prediction.",
    )

    class_labels: List[Literal["TECH_ISSUE", "BILLING", "GENERAL_QUERY"]] = Field(
        ...,
        description="The predicted class labels for the support ticket.",
    )
```

### Classifying Text

The function **`multi_classify`** is responsible for multi-label classification.

```python
# <%hide%>
from typing import List
from pydantic import BaseModel, Field
from typing import Literal


class MultiClassPrediction(BaseModel):
    """
    Class for a multi-class label prediction.

    Examples:
    - "My account is locked": ["TECH_ISSUE"]
    - "I can't access my billing info": ["TECH_ISSUE", "BILLING"]
    - "When do you close for holidays?": ["GENERAL_QUERY"]
    - "My payment didn't go through and now I can't log in": ["BILLING", "TECH_ISSUE"]
    """

    chain_of_thought: str = Field(
        ...,
        description="The chain of thought that led to the prediction.",
    )

    class_labels: List[Literal["TECH_ISSUE", "BILLING", "GENERAL_QUERY"]] = Field(
        ...,
        description="The predicted class labels for the support ticket.",
    )


# <%hide%>
import instructor

client = instructor.from_provider("openai/gpt-5-nano")


def multi_classify(data: str) -> MultiClassPrediction:
    """Perform multi-label classification on the input text."""
    return client.create(
        model="gpt-4o-mini",
        response_model=MultiClassPrediction,
        messages=[
            {
                "role": "user",
                "content": f"Classify the following support ticket: <ticket>{data}</ticket>",
            },
        ],
    )
```

### Testing and Evaluation

Finally, we test the multi-label classification function using a sample support ticket.

```python
# <%hide%>
from typing import List
from pydantic import BaseModel, Field
from typing import Literal
import instructor


class MultiClassPrediction(BaseModel):
    """
    Class for a multi-class label prediction.

    Examples:
    - "My account is locked": ["TECH_ISSUE"]
    - "I can't access my billing info": ["TECH_ISSUE", "BILLING"]
    - "When do you close for holidays?": ["GENERAL_QUERY"]
    - "My payment didn't go through and now I can't log in": ["BILLING", "TECH_ISSUE"]
    """

    chain_of_thought: str = Field(
        ...,
        description="The chain of thought that led to the prediction.",
    )

    class_labels: List[Literal["TECH_ISSUE", "BILLING", "GENERAL_QUERY"]] = Field(
        ...,
        description="The predicted class labels for the support ticket.",
    )


client = instructor.from_provider("openai/gpt-5-nano")


def multi_classify(data: str) -> MultiClassPrediction:
    """Perform multi-label classification on the input text."""
    return client.create(
        model="gpt-4o-mini",
        response_model=MultiClassPrediction,
        messages=[
            {
                "role": "user",
                "content": f"Classify the following support ticket: <ticket>{data}</ticket>",
            },
        ],
    )


# <%hide%>
# Test multi-label classification
ticket = "My account is locked and I can't access my billing info."
prediction = multi_classify(ticket)
assert "TECH_ISSUE" in prediction.class_labels
assert "BILLING" in prediction.class_labels
print(f"Ticket: {ticket}")
#> Ticket: My account is locked and I can't access my billing info.
print(f"Predicted Labels: {prediction.class_labels}")
#> Predicted Labels: ['TECH_ISSUE', 'BILLING']
```

By using Literals and including few-shot examples, we've improved both the single-label and multi-label classification implementations. These changes enhance type safety and provide better guidance for the AI model, potentially leading to more accurate classifications.
