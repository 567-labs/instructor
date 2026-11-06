import cohere
import instructor
from pydantic import BaseModel, Field


# Patching the Cohere client with the instructor for enhanced capabilities
client = instructor.from_cohere(
    cohere.ClientV2(),
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
group = client.messages.create(
    response_model=Group,
    messages=[{"role": "user", "content": task}],
    temperature=0,
)

print(group.model_dump_json(indent=2))
"""
{
  "group_name": "The Beatles",
  "members": [
    {
      "name": "John Lennon",
      "country_of_origin": "England"
    },
    {
      "name": "Paul McCartney",
      "country_of_origin": "England"
    },
    {
      "name": "George Harrison",
      "country_of_origin": "England"
    },
    {
      "name": "Ringo Starr",
      "country_of_origin": "England"
    }
  ]
}
"""
