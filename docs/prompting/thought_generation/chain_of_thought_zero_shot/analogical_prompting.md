---
description: "Analogical Prompting aims to help improve model accuracy by getting a model to generate relevant exemplars before solving the problem"
---

Analogical Prompting<sup><a href="https://arxiv.org/pdf/2310.01714">1</a></sup> is a method that aims to get LLMs to generate examples that are relevant to the problem before starting to address the user's query.

This takes advantage of the various forms of knowledge that the LLM has acquired during training and explicitly prompts them to recall the relevant problems and solutions. We can use Analogical Prompting using the following template

![](../../../img/analogical_prompting.png)

!!! example "Analogical Prompting Prompt Template"

    Problem: [User Prompt]

    Relevant Problems: Recall three relevant and distinct problems. For each problem, describe it and explain the solution

    Solve the problem

We can implement this using `instructor` as seen below with some slight modifications.

```python hl_lines="33-36"
from pydantic import BaseModel, Field
import instructor
from textwrap import dedent
client = instructor.from_provider("openai/gpt-5-nano")


class RelevantProblem(BaseModel):
    problem_explanation: str
    solution: str


class Response(BaseModel):
    relevant_problems: list[RelevantProblem] = Field(
        max_length=3,
        min_length=3,
    )
    answer: RelevantProblem


def analogical_prompting(query: str):
    return client.create(
        messages=[
            {
                "role": "user",
                "content": dedent(
                    f"""
                <problem>
                {query}
                </problem>

                Relevant Problems: Recall three relevant and
                distinct problems. For each problem, describe
                it and explain the solution before solving
                the problem
                """
                ),
            }
        ],
        model="gpt-4o",
        response_model=Response,
    )


if __name__ == "__main__":
    query = (
        "What is the area of the square with the four "
        "vertices at (-2, 2), (2, -2), (-2, -6), and "
        "(-6, -2)?"
    )
    response = analogical_prompting(query)
    for problem in response.relevant_problems:
        print(problem.model_dump_json(indent=2))
        """
        {
          "problem_explanation": "Determine the distance
          between two points in a coordinate plane.",
          "solution": "To find the distance between two
          points, use the distance formula: \\(d =
          \\sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}\\). This
          formula calculates the Euclidean distance between
          points (x_1, y_1) and (x_2, y_2)."
        }
        """
        """
        {
          "problem_explanation": "Calculate the area of a
          square given its side length.",
          "solution": "The area of a square can be found
          using the formula: \\(A = s^2\\), where \\(s\\) is
          the length of one side of the square."
        }
        """
        """
        {
          "problem_explanation": "Identify vertices and
          properties of a geometry shape such as
          parallelogram.",
          "solution": "For any quadrilateral, verify that
          all sides are equal and angles are right angles to
          confirm it is a square. Use properties of
          quadrilaterals and distance formula."
        }
        """

    print(response.answer.model_dump_json(indent=2))
    """
    {
      "problem_explanation": "Calculate the area of a
      square given its vertices.",
      "solution": "First, confirm the shape is a square by
      checking the distance between consecutive vertices
      and ensuring all sides are of equal length using the
      distance formula. For vertices (-2,2), (2,-2),
      (-2,-6), and (-6,-2), calculate distances between
      consecutive points. If distances are equal, use the
      side length to compute area using \\(A = s^2\\)."
    }
    """
```

### References

<sup id="ref-1">1</sup>: [Large Language Models As Analogical Reasoners](https://arxiv.org/pdf/2310.01714)
