---
description: "Reverse Chain Of Thought is a method to help identify logical inconsistencies in the reasoning steps of a large language model's response"
---

We can use a method called Reverse Chain Of Thought<sup><a href="https://arxiv.org/pdf/2305.11499">1</a></sup> to reverse engineer a problem given a solution. This helps us to find specific inconsistencies in the reasoning steps taken by our model and to give targeted feedback which can improve the quality of the solution.

This is done through a 3 step process

1. **Reconstruct The Question** : We first attempt to reconstruct the original problem given the solution and reasoning steps generated
2. **Identify Inconsistencies** : Identify the inconsistencies between the original problem and the reconstructed problem
3. **Generate Feedback** : Give fine-grained fedback to guide the LLM in revising its solution

We can implement this using `instructor` as seen below.

```python hl_lines="54-59 76-83 98-107 127-140 155-167"
import instructor
from pydantic import BaseModel, Field
client = instructor.from_provider("openai/gpt-5-nano")


class ReconstructedPrompt(BaseModel):
    chain_of_thought: str
    reconstructed_prompt: str = Field(
        description="""Reconstruction of a potential prompt
        that could have been used to generate the reasoning
        and final solution provided by the user"""
    )


class ConditionList(BaseModel):
    conditions: list[str] = Field(
        description="""Key information and conditions present
        in the reasoning steps which are relevant to answering
        the question"""
    )


class ModelFeedback(BaseModel):
    detected_inconsistencies: list[str] = Field(
        description="""Inconsistencies that were detected between
        the original condition list and the reconstructed condition
        list"""
    )
    feedback: str = Field(
        description="""Feedback on how to fix the inconsistencies
        detected in the original condition list and the reconstructed
        condition list"""
    )
    is_equal: bool


class ModelResponse(BaseModel):
    chain_of_thought: str = Field(
        description="""Logical Steps that were taken to derive
        the final concluding statement"""
    )
    correct_answer: str


def generate_response(query: str):
    return client.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """
                You are a helpful AI Question Answerer. You are
                about to be passed a query by a User.

                Make sure to generate a series of logical steps
                and reason about the problem before generating
                a solution.
                """,
            },
            {"role": "user", "content": query},
        ],
        response_model=ModelResponse,
    )


def reconstruct_prompt(model_response: ModelResponse):
    return client.create(
        model="gpt-4o",
        response_model=ReconstructedPrompt,
        messages=[
            {
                "role": "system",
                "content": f"""
                    Give the concrete prompt (problem) that can
                    generate this answer. The problem should
                    contain all basic and necessary information
                    and correspond to the answer. The problem
                    can only ask for one result

                    Reasoning: {model_response.chain_of_thought}
                    Response: {model_response.correct_answer}
                    """,
            }
        ],
    )


def deconstruct_prompt_into_condition_list(prompt: str):
    return client.create(
        model="gpt-4o",
        response_model=ConditionList,
        messages=[
            {
                "role": "system",
                "content": """
                You are an expert AI system that excels at
                analyzing and decomposing questions into their
                constituent parts.

                Please list the conditions of the problem given
                below. There might be multiple conditions in the
                problem so make sure to navigate through the
                prompt incrementally, identifying and extracting
                the conditions necessary to answer the question
                in your final response.
                """,
            },
            {"role": "user", "content": prompt},
        ],
    )


def generate_feedback(
    original_condition_list: list[str], final_condition_list: list[str]
):
    formatted_original_conditions = "\n- ".join(original_condition_list)
    formatted_final_conditions = "\n- ".join(final_condition_list)
    return client.create(
        model="gpt-4o",
        response_model=ModelFeedback,
        messages=[
            {
                "role": "system",
                "content": f"""
                You are an expert AI system that excels at
                analyzing and comparing two lists of conditions.

                Original Condition List:
                {formatted_original_conditions}

                Reconstructed Condition List:
                {formatted_final_conditions}

                Determine if the two condition lists are roughly
                equivalent. If they are not, give targeted
                feedback on what is missing from the reconstructed
                condition list as compared to the original condition
                list and how it can be fixed.
                """,
            }
        ],
    )


def revise_response(response: ModelResponse, feedback: ModelFeedback):
    formatted_inconsistencies = "\n- ".join(feedback.detected_inconsistencies)
    return client.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"""
                Here are the mistakes and reasons in your answer
                to the prompt

                Original Response: {response.correct_answer}
                You have overlooked some real conditions:
                {formatted_inconsistencies}

                Here are detailed reasons:
                {feedback.feedback}

                Generate a revised response that takes into account
                the detailed feedback and includes the ignored
                conditions
                """,
            }
        ],
        response_model=ModelResponse,
    )


if __name__ == "__main__":
    query = """
    Mary is an avid gardener. Yesterday, she received 18 new
    potted plants from her favorite plant nursery. She already
    has 2 potted plants on each of the 40 window ledges of her
    large backyard. How many potted plants will Mary remain
    with?
    """
    response = generate_response(query)
    reconstructed_prompt = reconstruct_prompt(response)
    print(reconstructed_prompt.reconstructed_prompt)
    """
    Mary received 18 new potted plants. She already has 2 potted plants on each
    of the 40 window ledges in her backyard. How many potted plants does she have now?
    """

    original_condition_list = deconstruct_prompt_into_condition_list(query)
    new_condition_list = deconstruct_prompt_into_condition_list(
        reconstructed_prompt.reconstructed_prompt
    )
    print(original_condition_list.model_dump_json(indent=2))
    """
    {
      "conditions": [
        "Mary received 18 new potted plants.",
        "Mary has 2 potted plants on each of the 40 window ledges in her backyard.",
        "We are required to find the total number of potted plants Mary will have."
      ]
    }
    """
    print(new_condition_list.model_dump_json(indent=2))
    """
    {
      "conditions": [
        "Mary received 18 new potted plants.",
        "She already has 2 potted plants on each of the 40 window ledges in her backyard."
      ]
    }
    """

    feedback = generate_feedback(
        original_condition_list.conditions, new_condition_list.conditions
    )
    print(feedback.model_dump_json(indent=2))
    """
    {
      "detected_inconsistencies": [
        "The reconstructed list is missing the requirement
        to find the total number of potted plants Mary will
        have."
      ],
      "feedback": "Add the requirement of finding the total
      number of potted plants Mary will have to the
      reconstructed condition list to match the original
      condition list.",
      "is_equal": false
    }
    """

    if not feedback.is_equal:
        response = revise_response(response, feedback)

    print(response.model_dump_json(indent=2))
    """
    {
      "chain_of_thought": "First, we note that Mary starts
      with 18 potted plants. According to the problem, she
      bought 2 packs of 40 new potted plants. So, to find
      the total number of plants she will have, we add the
      number of plants she initially has to the number she
      bought. This gives us 18 (initial) + 2 * 40 (new) =
      18 + 80 = 98 potted plants.",
      "correct_answer": "98 potted plants"
    }
    """
```

### References

<sup id="ref-1">1</sup>: [RCoT: Detecting And Rectifying Factual Inconsistency In Reasoning By Reversing Chain-Ofthought](https://arxiv.org/pdf/2305.11499)
