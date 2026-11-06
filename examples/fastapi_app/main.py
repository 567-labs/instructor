from fastapi import FastAPI
from instructor import ResponseSchema
import instructor.dsl as dsl
from pydantic import BaseModel, Field

app = FastAPI(title="Example Application using instructor")


class SearchRequest(BaseModel):
    body: str


class SearchQuery(ResponseSchema):
    title: str = Field(..., description="Question that the query answers")
    query: str = Field(
        ...,
        description="Detailed, comprehensive, and specific query to be used for semantic search",
    )


SearchResponse = dsl.MultiTask(
    subtask_class=SearchQuery,
    description="Correctly segmented set of search queries",
)


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    task = (
        dsl.ChatCompletion(name="Segmenting Search requests example")
        | dsl.SystemTask(task="Segment search results")
        | dsl.TaggedMessage(content=request.body, tag="query")
        | dsl.TipsMessage(
            tips=[
                "Expand query to contain multiple forms of the same word (SSO -> Single Sign On)",
                "Use the title to explain what the query should return, but use the query to complete the search",
                "The query should be detailed, specific, and cast a wide net when possible",
            ]
        )
        | SearchRequest
    )
    return await task.acreate()
