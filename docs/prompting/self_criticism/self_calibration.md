---
description: "Self Calibration aims to get language models to determine what they know and do not know"
---

We want our language models to be able to output the extent of their confidence in predictions. To do so, we can get language models to evaluate their responses to a given prompt using a technique called Self Calibration <sup><a href="https://arxiv.org/pdf/2207.05221">1</a></sup>

> The original paper used a fine-tuned regression head over the language model's final output. However, since we don't have access to the model's final hidden states, we can substitute it for a function call instead to achieve a similar result.

We can ask language models to evaluate their outputs by using the following template

We can implement this using `instructor` as seen below

```python hl_lines="23-27"
import instructor
from pydantic import BaseModel, Field
client = instructor.from_provider("openai/gpt-5-nano")


class SelfCalibration(BaseModel):
    chain_of_thought: str
    is_valid_answer: bool = Field(description="Whether the answer is correct or not")


def evaluate_model_output(original_prompt: str, model_response: str):
    return client.create(
        messages=[
            {
                "role": "user",
                "content": f"""
                Question: {original_prompt}

                {model_response}

                Is this a valid answer to the question?
                Make sure to examine the question
                thoroughly and generate a complete
                reasoning for why the answer is correct
                or not before responding.
                """,
            }
        ],
        response_model=SelfCalibration,
        model="gpt-4o",
    )


if __name__ == "__main__":
    original_prompt = """
    Question: Who was the third president of the
    United States?
    """
    model_response = """
    Here are some brainstormed ideas: James Monroe
    Thomas Jefferson
    Jefferson
    Thomas Jefferson
    George Washington
    """
    response = evaluate_model_output(original_prompt, model_response)
    print(response.model_dump_json(indent=2))
    """
    {
      "chain_of_thought": "Let's examine the question
      carefully: 'Who was the third president of the
      United States?'\n\nThe brainstormed ideas are:
      \n1. James Monroe\n2. Thomas Jefferson\n3.
      Jefferson\n4. Thomas Jefferson\n5. George
      Washington.\n\nTo determine the validity of these
      answers, I'll cross-check with historical
      records.\n\n1. James Monroe was not the third
      president; he was the fifth president.\n2. Thomas
      Jefferson was indeed the third president of the
      United States.\n3. 'Jefferson' is a correct but
      incomplete answer; it lacks the first name, though
      it is commonly understood.\n4. 'Thomas Jefferson'
      is the full name and correct answer.\n5. George
      Washington was the first president, not the
      third.\n\nTherefore, the correct, valid answer to
      the question 'Who was the third president of the
      United States?' is 'Thomas Jefferson,' and this
      answer is correct.",
      "is_valid_answer": true
    }
    """
```

### References

<sup id="ref-1">1</sup>: [Language Models (Mostly) Know What They Know](https://arxiv.org/pdf/2207.05221)
