import os
import instructor
from pydantic import BaseModel
import pytest


class User(BaseModel):
    name: str
    age: int


class UserList(BaseModel):
    items: list[User]


MODEL = os.getenv("GOOGLE_GENAI_MODEL", "google/gemini-pro")


@pytest.mark.asyncio
async def test_list_of_strings():
    client = instructor.from_provider(
        MODEL,
        mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS,
        async_client=True,
    )

    users = [
        {
            "name": "Jason",
            "age": 25,
        },
        {
            "name": "Elizabeth",
            "age": 12,
        },
        {
            "name": "Chris",
            "age": 27,
        },
    ]

    prompt = """
    Extract a list of users from the following text:

    {% for user in users %}
    - Name: {{ user.name }}, Age: {{ user.age }}
    {% endfor %}
    """

    result = await client.chat.completions.create(
        response_model=UserList,
        messages=[
            {"role": "user", "content": prompt},
        ],
        context={"users": users},
    )

    assert isinstance(result, UserList), "Result should be an instance of UserList"
    assert isinstance(result.items, list), "items should be a list"
    assert len(result.items) == 3, "List should contain 3 items"

    names = [item.name.upper() for item in result.items]
    assert "JASON" in names, "'JASON' should be in the list"
    assert "ELIZABETH" in names, "'ELIZABETH' should be in the list"
    assert "CHRIS" in names, "'CHRIS' should be in the list"
