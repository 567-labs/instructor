import instructor
import enum

from typing import Optional
from pydantic import BaseModel, Field
from openai import OpenAI

client = instructor.from_openai(OpenAI())


class Action(enum.Enum):
    CREATE = "create_task"
    DELETE = "close_task"
    UPDATE = "update_task"


class Projects(enum.Enum):
    FRONTLINE_QA_AI = "frontline_qa_ai"
    FUTURE_OF_PROGRAMMING = "future_of_programming"
    PERSONAL_SITE = "personal_site"
    NORDIC_HAMSTRING_CURLS = "nordic_hamstring_curls"


class Buckets(enum.Enum):
    FINANCE = "finance"
    PURVIEW_OPERATIONS = "purview_operations"
    TASKBOT = "taskbot"
    CHECKBOT = "checkbot"
    NIGHT_HACKING = "night_hacking"
    TICKLER = "tickler"


class TaskAction(BaseModel):
    id: int
    method: Action = Field(
        description="Method of creating and closing a task: to close a task, only an ID is required"
    )
    waiting_on: Optional[list[int]] = Field(
        None, description="IDs of tasks that this task is waiting on"
    )
    name: Optional[str] = Field(None, description="Name of the task")
    notes: Optional[str] = Field(None, description="Notes about the task")
    bucket: Optional[Buckets] = Field(
        None, description="Bucket of the task, to set, or update"
    )
    project: Optional[Projects] = Field(
        None, description="Project of the task, to set, or update"
    )


class Response(BaseModel):
    text: str = Field(description="The text of the response")
    task_action: Optional[list[TaskAction]] = Field(
        description="The action to take on the task"
    )


initial_messages = [
    {
        "role": "system",
        "content": "You are an AI assistant. have the ability to create, update, and close tasks.",
    },
    {
        "role": "assistant",
        "content": """
        The task is below. When assisting the user, reference the details from this task.

        [BEGIN TASK]
            id: 23
            Name: Create 10 new GIFs
            Description: Create 10 new GIFs for the Taskbot page on the user's personal site. They should be similar to the existing GIFs, but with different use cases.
            Projects: Personal site
            Buckets: Taskbot
            Updates:
        [BEGIN UPDATE]
            **User Update - September 01, 2023 03:58:00 PM EDT**
            The user plans to create the GIFs in the background as they work through their daily tasks. They aim to produce about one to two GIFs per day. If this plan doesn't work, they will reconsider their strategy.
        [END UPDATE]
        [END TASK]
    """,
    },
    {"role": "assistant", "content": "What's up with this task?"},
    {
        "role": "user",
        "content": "Change it to 20, then make a new task for when its done make 20 more that moves.",
    },
]

response: Response = client.chat.completions.create(
    messages=initial_messages, response_model=Response, model="gpt-4"
)  # type: ignore

print(response.model_dump_json(indent=2))
{
    "text": "Updating task to create 20 GIFs and creating a new task to create an additional 20 animated GIFs after the initial task is done.",
    "task_action": [
        {
            "id": 23,
            "method": "update_task",
            "waiting_on": None,
            "name": "Create 20 new GIFs",
            "notes": "The user increased the number of GIFs from 10 to 20. They plan to create these as they work through their daily tasks, creating about one to two GIFs per day. If this plan doesn't work, they will reconsider their strategy.",
            "bucket": "taskbot",
            "project": "personal_site",
        },
        {
            "id": 24,
            "method": "create_task",
            "waiting_on": [23],
            "name": "Create 20 new animated GIFs",
            "notes": "The task will be initiated once the task with id 23 is completed.",
            "bucket": "taskbot",
            "project": "personal_site",
        },
    ],
}
