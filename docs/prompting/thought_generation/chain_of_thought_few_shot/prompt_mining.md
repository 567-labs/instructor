---
description: "We get a LLM to generate prompts"
---

Large Language Models are sensitive to the way that they are prompted. When prompted incorrectly, they might perform much worse despite having the information or capability to respond to the prompt. Prompt Mining aims to help us discover better formats that occur more frequently in the corpus.

Here are some examples of mined completions that were provided in the paper.

| Manual Prompts                      | Mined Prompts           |
| ----------------------------------- | ----------------------- |
| x is affiliated with the y religion | x who converted to y    |
| The headquarter of x is in y        | x is based in y         |
| x died in y                         | x died at his home in y |
| x is represented by music label y   | x recorded for y        |
| x is a subclass of y                | x is a type of y        |

> The original paper uses a large wikipedia corpus to automatically extract prompt templates by looking at middle words of the prompts and parsing the dependencies within the sentence. We present a more lightweight approach to help achieve a similar result with `instructor`.

We can implement Prompt Mining using `instructor` as seen below.

```python hl_lines="29-33"
from pydantic import BaseModel, Field
import instructor

class PromptTemplate(BaseModel):
    prompt_template: str = Field(
        description=(
            """
            A template that has the subject and object that we
            want to extract from the prompt replaced with a
            single placeholder of {subject} and {object}.
            Rephrase the prompt if necessary to make it more
            concise and easier to understand
            """
        ),
    )


client = instructor.from_provider("openai/gpt-5-nano")


def generate_prompt_templates(prompt: str):
    return client.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert prompt miner that excels at "
                    "generating prompt templates which are more "
                    "concise and easier to understand\n\nYou are "
                    "about to be passed a prompt to extract 3 new "
                    "prompt templates for"
                ),
            },
            {"role": "system", "content": prompt},
        ],
        response_model=list[PromptTemplate],
        temperature=0,
        max_retries=3,
        model="gpt-4o",
    )


if __name__ == "__main__":
    prompt = "France is the capital of Paris"
    prompt_template = generate_prompt_templates(prompt)
    for prompt in prompt_template:
        print(prompt)
        #> prompt_template='{subject} is the capital of {object}'
        #> prompt_template='The capital of {object} is {subject}'
        #> prompt_template="{object}'s capital is {subject}"
```

### References

<sup id="ref-1">1</sup>: [How Can We Know What Language Models Know? ](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00324/96460/How-Can-We-Know-What-Language-Models-Know)
