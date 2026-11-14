---
title: Automating Action Item Extraction from Meeting Transcripts
description: Learn to extract actionable items from meeting transcripts using OpenAI's API and Pydantic for efficient project management.
---

# Extracting Action Items from Meeting Transcripts

In this guide, we'll walk through how to extract action items from meeting transcripts using OpenAI's API and Pydantic. This use case is essential for automating project management tasks, such as task assignment and priority setting.

For multi-label classification, we introduce a new enum class and a different Pydantic model to handle multiple labels.

!!! tips "Motivation"

    Significant amount of time is dedicated to meetings, where action items are generated as the actionable outcomes of these discussions. Automating the extraction of action items can save time and guarantee that no critical tasks are overlooked.

## Defining the Structures

We'll model a meeting transcript as a collection of **`Ticket`** objects, each representing an action item. Every **`Ticket`** can have multiple **`Subtask`** objects, representing smaller, manageable pieces of the main task.

## Extracting Action Items

To extract action items from a meeting transcript, we use the **`generate`** function. It calls OpenAI's API, processes the text, and returns a set of action items modeled as **`ActionItems`**.

## Evaluation and Testing

To test the **`generate`** function, we provide it with a sample transcript, and then print the JSON representation of the extracted action items.

```python
import instructor
from typing import Iterable, List, Optional
from enum import Enum
from pydantic import BaseModel


class PriorityEnum(str, Enum):
    high = "High"
    medium = "Medium"
    low = "Low"


class Subtask(BaseModel):
    """Correctly resolved subtask from the given transcript"""

    id: int
    name: str


class Ticket(BaseModel):
    """Correctly resolved ticket from the given transcript"""

    id: int
    name: str
    description: str
    priority: PriorityEnum
    assignees: List[str]
    subtasks: Optional[List[Subtask]]
    dependencies: Optional[List[int]]


# Apply the patch to the OpenAI client
# enables response_model keyword
client = instructor.from_provider("openai/gpt-6-luna")


def generate(data: str) -> Iterable[Ticket]:
    return client.create(
        model="gpt-6-luna",
        response_model=Iterable[Ticket],
        messages=[
            {
                "role": "system",
                "content": "The following is a transcript of a meeting...",
            },
            {
                "role": "user",
                "content": f"Create the action items for the following transcript: {data}",
            },
        ],
    )


prediction = generate("""
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
Alice: Sounds like a plan. Let's get these tasks modeled out and get started.""")
```

## Visualizing the tasks

In order to quickly visualize the data we used code interpreter to create a graphviz export of the json version of the ActionItems array.

![Action items visualization showing extracted tasks with priorities and dependencies](../img/action_items.png)

```json
[
  {
    "id": 1,
    "name": "Improve Authentication System",
    "description": "Revamp the front-end and optimize the back-end of the authentication system",
    "priority": "High",
    "assignees": ["Bob", "Carol"],
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
    "assignees": ["Bob"],
    "subtasks": [],
    "dependencies": [1]
  },
  {
    "id": 5,
    "name": "Update User Documentation",
    "description": "Update the user documentation to reflect the changes in the authentication system",
    "priority": "Low",
    "assignees": ["Carol"],
    "subtasks": [],
    "dependencies": [2]
  }
]
```

In this example, the **`generate`** function successfully identifies and segments the action items, assigning them priorities, assignees, subtasks, and dependencies as discussed in the meeting.

By automating this process, you can ensure that important tasks and details are not lost in the sea of meeting minutes, making project management more efficient and effective.
