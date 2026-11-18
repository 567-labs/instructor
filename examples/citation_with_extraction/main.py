import json
from collections.abc import AsyncIterator
from fastapi import FastAPI, Request, HTTPException
from fastapi.params import Depends
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

import os
import instructor
from openai import AsyncOpenAI

# FastAPI app
app = FastAPI(
    title="Citation with Extraction",
)


class Fact(BaseModel):
    """
    Class representing single statement.
    Each fact has a body and a list of sources.
    If there are multiple facts make sure to break them apart such that each one only uses a set of sources that are relevant to it.
    """

    fact: str = Field(
        ...,
        description="Body of the sentences, as part of a response, it should read like a sentence that answers the question",
    )
    substring_quotes: list[str] = Field(
        ...,
        description="Each source should be a direct quote from the context, as a substring of the original content",
    )

    def _get_span(self, quote, context):
        import regex

        if not quote.strip():
            return

        start = context.find(quote)
        if start >= 0:
            yield start, start + len(quote)
            return

        max_edits = min(3, len(quote) // 20)
        if max_edits:
            pattern = f"(?:{regex.escape(quote)}){{e<={max_edits}}}"
            try:
                match = regex.search(pattern, context, regex.BESTMATCH, timeout=0.05)
            except TimeoutError:
                return
            if match is not None:
                yield match.span()

    def get_spans(self, context):
        if self.substring_quotes:
            for quote in self.substring_quotes:
                yield from self._get_span(quote, context)


class Question(BaseModel):
    context: str = Field(..., description="Context to extract answers from")
    query: str = Field(..., description="Question to answer")


def stream_extract(
    question: Question, client: instructor.AsyncInstructor
) -> AsyncIterator[Fact]:
    return client.create_iterable(
        model="gpt-6-luna",
        temperature=0,
        response_model=Fact,
        messages=[
            {
                "role": "system",
                "content": "You are a world class algorithm to answer questions with correct and exact citations. ",
            },
            {"role": "user", "content": "Answer question using the following context"},
            {"role": "user", "content": f"{question.context}"},
            {"role": "user", "content": f"Question: {question.query}"},
            {
                "role": "user",
                "content": "Tips: Make sure to cite your sources, and use the exact words from the context.",
            },
        ],
        max_completion_tokens=2000,
    )


def get_api_key(request: Request):
    """
    This just gets the API key from the request headers.
    but tries to read from the environment variable OPENAI_API_KEY first.
    """
    if "OPENAI_API_KEY" in os.environ:
        return os.environ["OPENAI_API_KEY"]

    auth = request.headers.get("Authorization")
    if auth is None:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    if auth.startswith("Bearer "):
        return auth.replace("Bearer ", "")

    raise HTTPException(status_code=401, detail="Expected a bearer token")


# Route to handle SSE events and return users
@app.post("/extract", response_class=StreamingResponse)
async def extract(question: Question, openai_key: str = Depends(get_api_key)):
    async def generate():
        async with AsyncOpenAI(api_key=openai_key) as openai_client:
            client = instructor.from_openai(openai_client)
            facts = stream_extract(question, client)
            async for fact in facts:
                spans = list(fact.get_spans(question.context))
                resp = {
                    "body": fact.fact,
                    "spans": spans,
                    "citation": [question.context[a:b] for (a, b) in spans],
                }
                yield f"data: {json.dumps(resp)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
