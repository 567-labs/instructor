---
description: "We can improve model performance by deliberating including incorrect examples of reasoning for our model to see"
---

We can get better performance from our model when using chain-of-thought by including examples of incorrect reasoning. This helps our language model to learn what mistakes to avoid when generating a response. This is known as Contrastive Chain Of Thought<sup><a href="https://arxiv.org/pdf/2311.09277">1</a></sup> and can be done using the following template.

!!! example "Contrastive Chain Of Thought template"

    <context>sample question</context>
    <question>sample question</question>

    <Explanations>
        <Explanation>correct reasoning</Explanation>
        <WrongExplanation>incorrect reasoning example</WrongExplanation>
    <Explanations>

    <context>sample question</context>
    <question>sample question</question>

We can implement Contrastive Chain Of Thought using `instructor` as seen below.

```python hl_lines="35-40"
import instructor
from pydantic import BaseModel, Field
from textwrap import dedent
client = instructor.from_provider("openai/gpt-5-nano")


class ChainOfThought(BaseModel):
    chain_of_thought: str = Field(description="Incorrect reasoning for the answer")
    correct_answer: str


def contrastive_chain_of_thought(
    query: str,
    context: str,
    example_prompt: str,
    correct_examples: list[str],
    incorrect_examples: list[str],
):
    correct_example_prompt = "\n".join(
        [f"<Explanation>{example}</Explanation>" for example in correct_examples]
    )
    incorrect_example_prompt = "\n".join(
        [
            f"<WrongExplanation>{example}</WrongExplanation>"
            for example in incorrect_examples
        ]
    )
    ""
    return client.create(
        model="gpt-4o",
        response_model=ChainOfThought,
        messages=[
            {
                "role": "system",
                "content": dedent(
                    f"""
            <prompt>
                <role>system</role>
                <context>
                You are an expert question answering AI System.

                You are about to be given some examples of incorrect
                and correct reasoning for a question. You will then
                be asked to correctly reason through another question
                to generate a valid response.
                </context>

                <question>{example_prompt}</question>

                <Explanations>
                    {correct_example_prompt}
                    {incorrect_example_prompt}
                </Explanations>
                <context>{context}</context>
                <question>{query}</question>

            </prompt>
            """
                ),
            }
        ],
    )


if __name__ == "__main__":
    context = """
    James writes a 3-page letter to 2
    different friends twice a week.
    """
    query = "How many pages does James write in a year?"

    sample_question = """
    James has 30 teeth. His dentist drills 4
    of them and caps 7 more teeth than he drills.

    What percentage of James' teeth does the dentist fix?
    """

    incorrect_examples = [
        """James has 30 teeth. The dentist drills and caps some
        teeth. Since drills are normally used on cars and not
        teeth, it's safe to say none of the teeth were actually
        fixed.""",
        """The dentist drills 4 teeth and caps 11 of them, which
        means that he fixes 15 teeth. So we take 15 and multiply
        it by the number of petals on a daisy, and the result is
        30%, which is the percentage of teeth he fixes.""",
    ]

    correct_examples = [
        """The dentist drills 4 teeth, so there are 30 - 4 = 26
        teeth left. The dentist caps 7 more teeth than he drills,
        so he caps 4 + 7 = 11 teeth. Therefore, the dentist fixes
        a total of 4 + 11 = 15 teeth. To find the percentage of
        teeth the dentist fixes, we divide the number of teeth
        fixed by the total number of teeth and multiply by 100:
        15/30 x 100 = 50%"""
    ]

    response = contrastive_chain_of_thought(
        query=query,
        context=context,
        example_prompt=sample_question,
        correct_examples=correct_examples,
        incorrect_examples=incorrect_examples,
    )

    print(response.model_dump_json(indent=2))
    """
    {
      "chain_of_thought": "First, let's determine how many pages James writes per week.
      He writes a 3-page letter to 2 different friends, so for one writing session, he
      writes 3 pages x 2 friends = 6 pages. He does this twice a week, so the total number
       of pages written per week is 6 pages/session x 2 sessions/week = 12 pages/week. \n\n
       Next, we need to find out how many weeks are in a year. There are 52 weeks in a year,
       so we multiply the number of pages James writes per week by the number of weeks in a year:
       12 pages/week x 52 weeks/year = 624 pages/year.\n\nTherefore, James writes 624 pages in a year.",
      "correct_answer": "624"
    }
    """
```

### References

<sup id="ref-1">1</sup>: [Contrastive Chain-of-Thought Prompting](https://arxiv.org/pdf/2311.09277)
