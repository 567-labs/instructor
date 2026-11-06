from instructor import ResponseSchema, dsl
from pydantic import Field
import json


class SearchQuery(ResponseSchema):
    query: str = Field(
        ...,
        description="Detailed, comprehensive, and specific query to be used for semantic search",
    )


SearchResponse = dsl.MultiTask(
    subtask_class=SearchQuery,
    description="Correctly segmented set of search queries",
)


task = (
    dsl.ChatCompletion(name="Segmenting Search requests example")
    | dsl.SystemTask(task="Segment search results")
    | dsl.TaggedMessage(
        content="can you send me the data about the video investment and the one about spot the dog?",
        tag="query",
    )
    | dsl.TipsMessage(
        tips=[
            "Expand query to contain multiple forms of the same word (SSO -> Single Sign On)",
            "Use the title to explain what the query should return, but use the query to complete the search",
            "The query should be detailed, specific, and cast a wide net when possible",
        ]
    )
    | SearchResponse
)


print(json.dumps(task.kwargs, indent=1))
"""
{
  "tasks": [
    {
      "query": "data about video investment"
    },
    {
      "query": "data about spot the dog"
    }
  ]
}
"""
