from __future__ import annotations

from typing import Any
from typing_extensions import assert_type

import openai
from pydantic import BaseModel

from instructor import (
    AsyncInstructor,
    Instructor,
    Maybe,
    Mode,
    Partial,
    from_anyscale,
    from_databricks,
    from_deepseek,
    from_openai,
    from_provider,
    from_together,
    from_vertexai,
    response_schema,
)
from instructor.decisions.client import AsyncDecisionsClient, DecisionsClient
from instructor.v2.dsl.maybe import MaybeBase


class User(BaseModel):
    name: str


def check_public_factories(
    sync_client: openai.OpenAI,
    async_client: openai.AsyncOpenAI,
    vertex_model: Any,
) -> None:
    assert_type(from_vertexai(vertex_model), Instructor)
    assert_type(from_vertexai(vertex_model, use_async=True), AsyncInstructor)
    assert_type(from_vertexai(vertex_model, _async=True), AsyncInstructor)
    assert_type(from_openai(sync_client), Instructor)
    assert_type(from_openai(async_client), AsyncInstructor)
    assert_type(from_openai(sync_client, mode=Mode.DECISIONS), DecisionsClient)
    assert_type(from_openai(async_client, mode=Mode.DECISIONS), AsyncDecisionsClient)
    assert_type(from_anyscale("model"), Instructor)
    assert_type(from_anyscale("model", async_client=True), AsyncInstructor)
    assert_type(from_together("model"), Instructor)
    assert_type(from_together("model", async_client=True), AsyncInstructor)
    assert_type(from_databricks("model"), Instructor)
    assert_type(from_databricks("model", async_client=True), AsyncInstructor)
    assert_type(from_deepseek("model"), Instructor)
    assert_type(from_deepseek("model", async_client=True), AsyncInstructor)
    assert_type(from_provider("openai/gpt-4o-mini"), Instructor)
    assert_type(from_provider("openai/gpt-4o-mini", async_client=True), AsyncInstructor)


def check_public_model_helpers() -> None:
    assert_type(response_schema(User), type[User])
    assert_type(Maybe(User), type[MaybeBase[User]])
    assert_type(Partial[User], type[User])


async def check_responses_on_installed_clients(
    sync_client: Instructor, async_client: AsyncInstructor
) -> None:
    assert_type(sync_client.responses.create(input="one", response_model=User), User)
    item, _ = await async_client.responses.create_with_completion(
        input="one", response_model=User
    )
    assert_type(item, User)
