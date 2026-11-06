import os

import pytest
import instructor
from pydantic import BaseModel, Field
from .public_client_models import User

cohere = pytest.importorskip("cohere")

pytestmark = pytest.mark.skipif(
    not os.getenv("COHERE_API_KEY"), reason="COHERE_API_KEY is unavailable"
)


@pytest.mark.skip(reason="Skipping if Cohere API is not available")
def test_client_cohere_response():
    client = cohere.ClientV2()
    instructor_client = instructor.from_cohere(
        client,
        max_tokens=1000,
        model="command-a-03-2025",
    )

    user = instructor_client.messages.create(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    assert user.name == "Jason"
    assert user.age == 10


@pytest.mark.skip(reason="Skipping if Cohere API is not available")
def test_client_cohere_response_with_nested_classes():
    client = cohere.ClientV2()
    instructor_client = instructor.from_cohere(
        client,
        max_tokens=1000,
        model="command-a-03-2025",
    )

    class Person(BaseModel):
        name: str = Field(description="name of the person")
        country_of_origin: str = Field(description="country of origin of the person")

    class Group(BaseModel):
        group_name: str = Field(description="name of the group")
        members: list[Person] = Field(description="list of members in the group")

    task = """\
    Given the following text, create a Group object for 'The Beatles' band

    Text:
    The Beatles were an English rock band formed in Liverpool in 1960. With a line-up comprising John Lennon, Paul McCartney, George Harrison and Ringo Starr, they are regarded as the most influential band of all time. The group were integral to the development of 1960s counterculture and popular music's recognition as an art form.
    """
    group = instructor_client.messages.create(
        response_model=Group,
        messages=[{"role": "user", "content": task}],
        temperature=0,
    )
    assert group.group_name == "The Beatles"
    assert len(group.members) == 4
    assert group.members[0].name == "John Lennon"
    assert group.members[1].name == "Paul McCartney"
    assert group.members[2].name == "George Harrison"
    assert group.members[3].name == "Ringo Starr"


@pytest.mark.skip(reason="Skipping if Cohere API is not available")
@pytest.mark.asyncio
async def test_client_cohere_async():
    client = cohere.AsyncClientV2()
    instructor_client = instructor.from_cohere(
        client,
        max_tokens=1000,
        model="command-a-03-2025",
    )

    class Person(BaseModel):
        name: str = Field(description="name of the person")
        country_of_origin: str = Field(description="country of origin of the person")

    class Group(BaseModel):
        group_name: str = Field(description="name of the group")
        members: list[Person] = Field(description="list of members in the group")

    task = """\
    Given the following text, create a Group object for 'The Beatles' band

    Text:
    The Beatles were an English rock band formed in Liverpool in 1960. With a line-up comprising John Lennon, Paul McCartney, George Harrison and Ringo Starr, they are regarded as the most influential band of all time. The group were integral to the development of 1960s counterculture and popular music's recognition as an art form.
    """
    group = await instructor_client.messages.create(
        response_model=Group,
        messages=[{"role": "user", "content": task}],
        temperature=0,
    )
    assert group.group_name == "The Beatles"
    assert len(group.members) == 4
    assert group.members[0].name == "John Lennon"
    assert group.members[1].name == "Paul McCartney"
    assert group.members[2].name == "George Harrison"
    assert group.members[3].name == "Ringo Starr"
