from pydantic import BaseModel
from fastapi import FastAPI
from openai import AsyncOpenAI
import instructor
import logfire
import asyncio
from collections.abc import Iterable
from fastapi.responses import StreamingResponse


class UserData(BaseModel):
    query: str


class MultipleUserData(BaseModel):
    queries: list[str]


class UserDetail(BaseModel):
    name: str
    age: int


app = FastAPI()
openai_client = AsyncOpenAI()
logfire.configure()
logfire.instrument_pydantic(record="all")
logfire.instrument_fastapi(app)
logfire.instrument_openai(openai_client)
client = instructor.from_openai(openai_client)


@app.post("/user", response_model=UserDetail)
async def endpoint_function(data: UserData) -> UserDetail:
    user_detail = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=UserDetail,
        messages=[
            {"role": "user", "content": f"Extract: `{data.query}`"},
        ],
    )
    logfire.info("/User returning", value=user_detail)
    return user_detail


@app.post("/many-users", response_model=list[UserDetail])
async def extract_many_users(data: MultipleUserData):
    async def extract_user(query: str):
        user_detail = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            response_model=UserDetail,
            messages=[
                {"role": "user", "content": f"Extract: `{query}`"},
            ],
        )
        logfire.info("/User returning", value=user_detail)
        return user_detail

    coros = [extract_user(query) for query in data.queries]
    return await asyncio.gather(*coros)


@app.post("/extract", response_class=StreamingResponse)
async def extract(data: UserData):
    supressed_client = AsyncOpenAI()
    logfire.instrument_openai(supressed_client, suppress_other_instrumentation=False)
    client = instructor.from_openai(supressed_client)
    users = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=Iterable[UserDetail],
        stream=True,
        messages=[
            {"role": "user", "content": data.query},
        ],
    )

    async def generate():
        with logfire.span("Generating User Response Objects"):
            async for user in users:
                resp_json = user.model_dump_json()
                logfire.info("Returning user object", value=resp_json)

                yield resp_json

    return StreamingResponse(generate(), media_type="text/event-stream")
