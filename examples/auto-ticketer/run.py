import instructor
from openai import OpenAI

from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum

client = instructor.from_openai(OpenAI())


class PriorityEnum(str, Enum):
    high = "High"
    medium = "Medium"
    low = "Low"


class Subtask(BaseModel):
    """
    Correctly resolved subtask from the given transcript
    """

    id: int = Field(..., description="Unique identifier for the subtask")
    name: str = Field(..., description="Informative title of the subtask")


class Ticket(BaseModel):
    """
    Correctly resolved ticket from the given transcript
    """

    id: int = Field(..., description="Unique identifier for the ticket")
    name: str = Field(..., description="Title of the task")
    description: str = Field(..., description="Detailed description of the task")
    priority: PriorityEnum = Field(..., description="Priority level")
    assignees: list[str] = Field(..., description="List of users assigned to the task")
    subtasks: Optional[list[Subtask]] = Field(
        None, description="List of subtasks associated with the main task"
    )
    dependencies: Optional[list[int]] = Field(
        None, description="List of ticket IDs that this ticket depends on"
    )


class ActionItems(BaseModel):
    """
    Correctly resolved set of action items from the given transcript
    """

    items: list[Ticket]


def generate(data: str):
    return client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=ActionItems,
        messages=[
            {
                "role": "system",
                "content": "The following is a transcript of a meeting between a manager and their team. The manager is assigning tasks to their team members and creating action items for them to complete.",
            },
            {
                "role": "user",
                "content": f"Create the action items for the following transcript: {data}",
            },
        ],
    )


prediction = generate(
    """
Alice: Hey team, we have several critical tasks we need to tackle for the upcoming release. First, we need to work on improving the authentication system. It's a top priority.

Bob: Got it, Alice. I can take the lead on the authentication improvements. Are there any specific areas you want me to focus on?

Alice: Good question, Bob. We need both a front-end revamp and back-end optimization. So basically, two sub-tasks.

Carol: I can help with the front-end part of the authentication system.

Bob: Great, Carol. I'll handle the back-end optimization then.

Alice: Perfect. Now, after the authentication system is improved, we have to integrate it with our new billing system. That's a medium priority task.

Carol: Is the new billing system already in place?

Alice: No, it's actually another task. So it's a dependency for the integration task. Bob, can you also handle the billing system?

Bob: Sure, but I'll need to complete the back-end optimization of the authentication system first, so it's dependent on that.

Alice: Understood. Lastly, we also need to update our user documentation to reflect all these changes. It's a low-priority task but still important.

Carol: I can take that on once the front-end changes for the authentication system are done. So, it would be dependent on that.

Alice: Sounds like a plan. Let's get these tasks modeled out and get started."""
)

print(prediction.model_dump_json(indent=2))
"""
{
  "items": [
    {
      "id": 1,
      "name": "Improve Authentication System",
      "description": "Revamp the front-end and optimize the back-end of the authentication system",
      "priority": "High",
      "assignees": [
        "Bob",
        "Carol"
      ],
      "subtasks": [
        {
          "id": 2,
          "name": "Front-end Revamp"
        },
        {
          "id": 3,
          "name": "Back-end Optimization"
        }
      ],
      "dependencies": []
    },
    {
      "id": 4,
      "name": "Integrate Authentication System with Billing System",
      "description": "Integrate the improved authentication system with the new billing system",
      "priority": "Medium",
      "assignees": [
        "Bob"
      ],
      "subtasks": [],
      "dependencies": [
        1
      ]
    },
    {
      "id": 5,
      "name": "Update User Documentation",
      "description": "Update the user documentation to reflect the changes in the authentication system",
      "priority": "Low",
      "assignees": [
        "Carol"
      ],
      "subtasks": [],
      "dependencies": [
        2
      ]
    }
  ]
}
"""
