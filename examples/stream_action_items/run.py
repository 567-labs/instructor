import instructor

from pydantic import BaseModel, Field
from typing import Optional
from collections.abc import Iterable
from openai import OpenAI
from rich.console import Console


client = instructor.from_openai(OpenAI())


class ActionItem(BaseModel):
    slug: str = Field(..., description="compact short slug")
    title: str = Field(description="The title of the action item")
    chain_of_thought: str = Field(
        description="Short chain of thought that led to this action item, specifically think about whether or not a task should be marked as completed"
    )
    is_completed: Optional[bool] = Field(
        False, description="Whether the action item is completed"
    )


class ActionItemResponse(BaseModel):
    action_items: Optional[list[ActionItem]] = Field(
        ..., title="The list of action items"
    )

    def patch(self, action_item: ActionItem):
        current_items = {item.slug: item for item in self.action_items}
        current_items[action_item.slug] = action_item
        new_response = ActionItemResponse(action_items=list(current_items.values()))
        print(f"BEFORE\n{self}\n\nAFTER\n{new_response}")
        return new_response

    def __repr__(self):
        completed_str = "DONE -"
        pending_str = "TODO -"

        def format_item(item):
            return f"{completed_str if item.is_completed else pending_str} {item.title}"

        return "\n\n".join([format_item(item) for item in self.action_items])

    def __str__(self) -> str:
        return self.__repr__()


console = Console()


def yield_action_items(transcript: str, state: ActionItemResponse):
    action_items = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        temperature=0,
        seed=42,
        response_model=Iterable[ActionItem],
        stream=True,
        messages=[
            {
                "role": "system",
                "content": f"""
                You're a world-class note taker. 
                You are given the current state of the notes and an additional piece of the transcript. 
                Use this to update the action.
                
                If you return an action item with the same ID as something in the set, It will be overwritten.
                Use this to update the complete status or change the title if there's more context. 

                - If they are distinct items, do not repeat the slug.
                - Only repeat a slug if we need to update the title or completion status.
                - If the completion status is not mentioned, it should be assumed to be incomplete.
                - For each task describe the success / completion criteria as well.
                - If something is explicitly mentioned as being done, mark it as done. 

                {state.model_dump_json(indent=2)}
                """,
            },
            {
                "role": "user",
                "content": f"Take the following transcript to return a set of transactions from the transcript\n\n{transcript}",
            },
        ],
    )

    for action_item in action_items:
        state = state.patch(action_item)
        yield state


transcript = """
Bob: Great, Carol. I'll handle the back-end optimization then.

Alice: Perfect. Now, after the authentication system is improved, we have to integrate it with our new billing system. That's a medium priority task.

Bob: Sure, but I'll need to complete the back-end optimization of the authentication system first, so it's dependent on that.

Jason: The backend optimization was finished last week actually.

Alice: Understood. Lastly, we also need to update our user documentation to reflect all these changes. It's a low-priority task but still important.
""".strip().split("\n\n")


def text_to_speech(chunk):
    """
    Uses a subprocess to convert text to speech via the `say` command on macOS.
    """
    import subprocess

    subprocess.run(["say", chunk], check=True)


def process_transcript(transcript: list[str]):
    state = ActionItemResponse(action_items=[])
    for chunk in transcript:
        console.print(f"update: {chunk}")
        for new_state in yield_action_items(chunk, state):
            state = new_state
            console.clear()
            console.print("# Action Items")
            console.print(str(state))
            console.print("\n")


if __name__ == "__main__":
    process_transcript(transcript)
