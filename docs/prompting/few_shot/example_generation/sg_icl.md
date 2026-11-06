---
title: "Generate In-Context Examples"
description: ""
---

How can we generate examples for our prompt?

Self-Generated In-Context Learning (SG-ICL) is a technique which uses an LLM to generate examples to be used during the task. This allows for in-context learning, where examples of the task are provided in the prompt.

We can implement SG-ICL using `instructor` as seen below.

```python
import instructor
from pydantic import BaseModel
from typing import Literal
n = 4  # num examples to generate per class


class GeneratedReview(BaseModel):
    review: str
    sentiment: Literal["positive", "negative"]


class SentimentPrediction(BaseModel):
    sentiment: Literal["positive", "negative"]


client = instructor.from_provider("openai/gpt-5-nano")


def generate_sample(input_review, sentiment):
    return client.create(
        model="gpt-4o",
        response_model=GeneratedReview,
        messages=[
            {
                "role": "user",
                "content": f"""
                           Generate a '{sentiment}' review similar to: {input_review}
                           Generated review:
                           """,
            }
        ],
    )


def predict_sentiment(input_review, in_context_samples):
    return client.create(
        model="gpt-4o",
        response_model=SentimentPrediction,
        messages=[
            {
                "role": "user",
                "content": "".join(
                    [
                        f"Review: {sample.review}\nSentiment: {sample.sentiment}\n\n"
                        for sample in in_context_samples
                    ]
                )
                + f"Review: {input_review}\nSentiment:",
            }
        ],
    ).sentiment


if __name__ == "__main__":
    input_review = (
        "This movie was a rollercoaster of emotions, keeping me engaged throughout."
    )

    # Generate in-context samples
    samples = [
        generate_sample(input_review, sentiment)
        for sentiment in ('positive', 'negative')
        for _ in range(n)
    ]
    for sample in samples:
        print(sample)
        """
        review='This film was an enthralling experience from start to finish, leaving me captivated every moment.' sentiment='positive'
        """
        """
        review='This film was an emotional journey that captivated me from start to finish.' sentiment='positive'
        """
        """
        review='The film took me on an unforgettable journey, capturing my attention at every moment.' sentiment='positive'
        """
        """
        review='This book was a riveting journey, capturing my attention from start to finish.' sentiment='positive'
        """
        """
        review='The movie was a total letdown, failing to hold my interest from start to finish.' sentiment='negative'
        """
        """
        review='This movie was a disjointed mess of emotions, leaving me confused throughout.' sentiment='negative'
        """
        """
        review='The movie was an emotional rollercoaster, but it left me feeling more confused than engaged.' sentiment='negative'
        """
        """
        review='This movie was a monotonous ride, failing to engage me at any point.' sentiment='negative'
        """
        """
        review='This film was an emotional journey, captivating me from start to finish.' sentiment='positive'
        """
        """
        review='This film captivated me from start to finish with its thrilling plot and emotional depth.' sentiment='positive'
        """
        """
        review='This movie was a breathtaking journey, capturing my attention from start to finish.' sentiment='positive'
        """
        """
        review='This movie was a chaotic mess of emotions, losing me at every turn.' sentiment='negative'
        """
        """
        review='This movie was a confusing mess, leaving me disengaged throughout.' sentiment='negative'
        """
        """
        review='This movie was a chore to sit through, leaving me bored most of the time.' sentiment='negative'
        """
        """
        review='This movie was a mishmash of confusing scenes, leaving me frustrated throughout.' sentiment='negative'
        """

    # Predict sentiment
    print(predict_sentiment(input_review, samples))
    #> positive
```

### References

<sup id="ref-1">1</sup>: [Self-Generated In-Context Learning: Leveraging Auto-regressive Language Models as a Demonstration Generator](https://arxiv.org/abs/2206.08082)

<sup id="ref-asterisk">\*</sup>: [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608)
