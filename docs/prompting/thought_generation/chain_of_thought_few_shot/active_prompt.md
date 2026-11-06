---
description: "Active prompting is a method used to identify the most effective examples for human annotation. "
---

When we have a large pool of unlabeled examples that could be used in a prompt, how should we decide which examples to manually label?

Active prompting is a method used to identify the most effective examples for human annotation. The process involves four key steps:

1. **Uncertainty Estimation**: Assess the uncertainty of the LLM's predictions on each possible example
2. **Selection**: Choose the most uncertain examples for human annotation
3. **Annotation**: Have humans label the selected examples
4. **Inference**: Use the newly labeled data to improve the LLM's performance

## Uncertainty Estimation

In this step, we define an unsupervised method to measure the uncertainty of an LLM in answering a given example.

!!! example "Uncertainty Estimation Example"

    Let's say we ask an LLM the following query:
    >query = "Classify the sentiment of this sentence as positive or negative: I am very excited today."

    and the LLM returns:
    >response = "positive"

    The goal of uncertainty estimation is to answer: **How sure is the LLM in this response?**

In order to do this, we query the LLM with the same example _k_ times. Then, we use the _k_ responses to determine how dissimmilar these responses are. Three possible metrics<sup><a href="https://arxiv.org/abs/2302.12246">1</a></sup> are:

1. **Disagreement**: Ratio of unique responses to total responses.
2. **Entropy**: Measurement based on frequency of each response.
3. **Variance**: Calculation of the spread of numerical responses.

Below is an example of uncertainty estimation for a single input example using the disagreement uncertainty metric.

```python
import instructor
from pydantic import BaseModel

class Response(BaseModel):
    height: int


client = instructor.from_provider("openai/gpt-5-nano")


def query_llm():
    return client.create(
        model="gpt-4o",
        response_model=Response,
        messages=[
            {
                "role": "user",
                "content": "How tall is the Empire State Building in meters?",
            }
        ],
    )


def calculate_disagreement(responses):
    unique_responses = set(responses)
    h = len(unique_responses)
    return h / k


if __name__ == "__main__":
    k = 5  # (1)!
    responses = [query_llm() for _ in range(k)]  # Query the LLM k times
    for response in responses:
        print(response)
        #> height=443
        #> height=443
        #> height=443
        #> height=443
        #> height=381

    print(
        calculate_disagreement([response.height for response in responses])
    )  # Calculate the uncertainty metric
    #> 0.4
```

1. _k_ is the number of times to query the LLM with a single unlabeled example

This process will then be repeated for all unlabeled examples.

## Selection & Annotation

Once we have a set of examples and their uncertainties, we can select _n_ of them to be annotated by humans. Here, we choose the examples with the highest uncertainties.

## Inference

Now, each time the LLM is prompted, we can include the newly-annotated examples.

## References

<sup id="ref-1">1</sup>: [Active Prompting with Chain-of-Thought for Large Language Models](https://arxiv.org/abs/2302.12246)

<sup id="ref-asterisk">\*</sup>: [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608)
