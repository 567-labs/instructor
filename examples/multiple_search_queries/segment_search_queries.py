import enum
import instructor

from openai import OpenAI
from pydantic import Field, BaseModel

client = instructor.from_openai(OpenAI())


class SearchType(str, enum.Enum):
    """Enumeration representing the types of searches that can be performed."""

    VIDEO = "video"
    EMAIL = "email"


class Search(BaseModel):
    """
    Class representing a single search query which contains title, query and the search type
    """

    search_title: str = Field(..., description="Title of the request")
    query: str = Field(..., description="Query to search for relevant content")
    type: SearchType = Field(..., description="Type of search")

    async def execute(self):
        import asyncio

        await asyncio.sleep(1)
        print(
            f"Searching for `{self.search_title}` with query `{self.query}` using `{self.type}`"
        )


class MultiSearch(BaseModel):
    """
    Class representing multiple search queries.
    Make sure they contain all the required attributes

    Args:
        searches (List[Search]): The list of searches to perform.
    """

    searches: list[Search] = Field(..., description="List of searches")

    def execute(self):
        import asyncio

        loop = asyncio.get_event_loop()

        tasks = asyncio.gather(*[search.execute() for search in self.searches])
        return loop.run_until_complete(tasks)


def segment(data: str) -> MultiSearch:
    """
    Convert a string into multiple search queries using OpenAI's GPT-3 model.

    Args:
        data (str): The string to convert into search queries.

    Returns:
        MultiSearch: An object representing the multiple search queries.
    """

    completion = client.chat.completions.create(
        model="gpt-4-0613",
        temperature=0.1,
        response_model=MultiSearch,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": f"Consider the data below:\n{data} and segment it into multiple search queries",
            },
        ],
        max_tokens=1000,
    )
    return MultiSearch.from_response(completion)


if __name__ == "__main__":
    queries = segment(
        "Please send me the video from last week about the investment case study and also documents about your GPDR policy?"
    )

    queries.execute()
    # >>> Searching for `Video` with query `investment case study` using `SearchType.VIDEO`
    # >>> Searching for `Documents` with query `GPDR policy` using `SearchType.EMAIL`
