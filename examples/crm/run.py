from enum import Enum
from pydantic import BaseModel, Field
import instructor
from openai import OpenAI

client = instructor.from_openai(OpenAI())


class CRMSource(Enum):
    personal = "personal"
    business = "business"
    work_contacts = "work_contacts"
    all = "all"


class CRMSearch(BaseModel):
    """A CRM search query

    The search description is a natural language description of the search query
    the backend will use semantic search so use a range of phrases to describe the search
    """

    source: CRMSource
    city_location: str = Field(
        ..., description="City location used to match the desired customer profile"
    )
    search_description: str = Field(
        ..., description="Search query used to match the desired customer profile"
    )


class CRMSearchQuery(BaseModel):
    """
    A set of CRM queries to be executed against a CRM system,
    for large locations decompose into multiple queries of smaller locations
    """

    queries: list[CRMSearch]


def query_crm(query: str) -> CRMSearchQuery:
    queries = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=CRMSearchQuery,
        messages=[
            {
                "role": "system",
                "content": """
            You are a world class CRM search career generator. 
            You will take the user query and decompose it into a set of CRM queries.
            """,
            },
            {"role": "user", "content": query},
        ],
    )
    return queries


if __name__ == "__main__":
    query = "find me all the pottery businesses in San Francisco and my friends in the east coast big cities"
    print(query_crm(query).model_dump_json(indent=2))
    """
    {
    "queries": [
        {
            "source": "business",
            "city_location": "San Francisco",
            "search_description": "pottery businesses"
        },
        {
            "source": "personal",
            "city_location": "New York",
            "search_description": "friends in New York"
        },
        {
            "source": "personal",
            "city_location": "Boston",
            "search_description": "friends in Boston"
        },
        {
            "source": "personal",
            "city_location": "Philadelphia",
            "search_description": "friends in Philadelphia"
        }
    ]
    }
    """
