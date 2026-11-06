from pydantic import BaseModel, Field
from typing import Union
import instructor
from openai import OpenAI


class Search(BaseModel):
    """Search action class with a 'query' field and a process method."""

    query: str = Field(description="The search query")

    def process(self):
        """Process the search action."""
        return f"Search method called for query: {self.query}"


class Lookup(BaseModel):
    """Lookup action class with a 'keyword' field and a process method."""

    keyword: str = Field(description="The lookup keyword")

    def process(self):
        """Process the lookup action."""
        return f"Lookup method called for keyword: {self.keyword}"


class Finish(BaseModel):
    """Finish action class with an 'answer' field and a process method."""

    answer: str = Field(description="The answer for finishing the process")

    def process(self):
        """Process the finish action."""
        return f"Finish method called with answer: {self.answer}"


# Union of Search, Lookup, and Finish
class TakeAction(BaseModel):
    action: Union[Search, Lookup, Finish]

    def process(self):
        """Process the action."""
        return self.action.process()


try:
    # Enables `response_model`
    client = instructor.from_openai(OpenAI())
    action = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=TakeAction,
        messages=[
            {"role": "user", "content": "Please choose one action"},
        ],
    )
    assert isinstance(action, TakeAction), "The action is not TakeAction"
    print(action.process())
except Exception as e:
    print(f"An error occurred: {e}")
