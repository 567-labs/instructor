---
description: "DECOMP involves using a LLM to break down a complicated task into sub tasks that it has been provided with"
---

Decomposed Prompting<sup><a href="https://arxiv.org/pdf/2210.02406">1</a></sup> leverages a Language Model (LLM) to deconstruct a complex task into a series of manageable sub-tasks. Each sub-task is then processed by specific functions, enabling the LLM to handle intricate problems more effectively and systematically.

In the code snippet below, we define a series of data models and functions to implement this approach.

The `derive_action_plan` function generates an action plan using the LLM, which is then executed step-by-step. Each action can be

1. InitialInput: Which represents the chunk of the original prompt we need to process
2. Split : An operation to split strings using a given separator
3. StrPos: An operation to help extract a string given an index
4. Merge: An operation to join a list of strings together using a given character

We can implement this using `instructor` as seen below.

```python hl_lines="57-58"
import instructor
from pydantic import BaseModel, Field
from typing import Union
client = instructor.from_provider("openai/gpt-5-nano")


class Split(BaseModel):
    split_char: str = Field(
        description="""This is the character to split
        the string with"""
    )

    def split_chars(self, s: str, c: str):
        return s.split(c)


class StrPos(BaseModel):
    index: int = Field(
        description="""This is the index of the character
        we wish to return"""
    )

    def get_char(self, s: list[str], i: int):
        return [c[i] for c in s]


class Merge(BaseModel):
    merge_char: str = Field(
        description="""This is the character to merge the
        inputs we plan to pass to this function with"""
    )

    def merge_string(self, s: list[str]):
        return self.merge_char.join(s)


class Action(BaseModel):
    id: int = Field(
        description="""Unique Incremental id to identify
        this action with"""
    )
    action: Union[Split, StrPos, Merge]


class ActionPlan(BaseModel):
    initial_data: str
    plan: list[Action]


def derive_action_plan(task_description: str) -> ActionPlan:
    return client.create(
        messages=[
            {
                "role": "system",
                "content": """Generate an action plan to help you complete
                the task outlined by the user""",
            },
            {"role": "user", "content": task_description},
        ],
        response_model=ActionPlan,
        max_retries=3,
        model="gpt-4o",
    )


if __name__ == "__main__":
    task = """Concatenate the second letter of every word in Jack
    Ryan together"""
    plan = derive_action_plan(task)
    print(plan.model_dump_json(indent=2))
    """
    {
      "initial_data": "Jack Ryan",
      "plan": [
        {
          "id": 1,
          "action": {
            "split_char": " "
          }
        },
        {
          "id": 2,
          "action": {
            "index": 1
          }
        },
        {
          "id": 3,
          "action": {
            "merge_char": ""
          }
        }
      ]
    }
    """

    curr = plan.initial_data
    cache = {}

    for action in plan.plan:
        if isinstance(action.action, Split) and isinstance(curr, str):
            curr = action.action.split_chars(curr, action.action.split_char)
        elif isinstance(action.action, StrPos) and isinstance(curr, list):
            curr = action.action.get_char(curr, action.action.index)
        elif isinstance(action.action, Merge) and isinstance(curr, list):
            curr = action.action.merge_string(curr)
        else:
            raise ValueError("Unsupported Operation")

        print(action, curr)
        #> id=1 action=Split(split_char=' ') ['Jack', 'Ryan']
        #> id=2 action=StrPos(index=1) ['a', 'y']
        #> id=3 action=Merge(merge_char='') ay

    print(curr)
    #> ay
```

### References

<sup id="ref-1">1</sup>: [Decomposed Prompting: A Modular Approach for Solving Complex Tasks](https://arxiv.org/pdf/2210.02406)
