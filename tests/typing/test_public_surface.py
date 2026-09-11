from __future__ import annotations

# mypy: disable-error-code=unused-coroutine

from collections.abc import AsyncGenerator, Generator
from types import CoroutineType
from typing import TYPE_CHECKING, Any, Protocol

from typing_extensions import assert_type

import openai
from pydantic import BaseModel

from instructor.v2.auto_client import from_provider
from instructor.v2.core.client import (
    AsyncInstructor,
    AsyncResponse,
    Instructor,
    Response,
)
from instructor.v2.core.function_calls import response_schema
from instructor.v2.core.mode import Mode
from instructor.v2.dsl.maybe import Maybe, MaybeBase
from instructor.v2.dsl.parallel import ParallelBase
from instructor.v2.dsl.partial import Partial
from instructor.v2.providers.openai.client import (
    from_anyscale,
    from_databricks,
    from_deepseek,
    from_openai,
    from_together,
)

if TYPE_CHECKING:
    import anthropic
    import cohere
    import google.generativeai as legacy_genai  # ty: ignore[unresolved-import]
    import groq
    import vertexai.generative_models as gm
    from botocore.client import BaseClient
    from cerebras.cloud.sdk import AsyncCerebras, Cerebras
    from fireworks.client import Fireworks
    from writerai import AsyncWriter, Writer
    from xai_sdk.sync.client import Client as SyncXAIClient

    from instructor.v2.providers.anthropic.client import from_anthropic
    from instructor.v2.providers.bedrock.client import from_bedrock
    from instructor.v2.providers.cerebras.client import from_cerebras
    from instructor.v2.providers.cohere.client import from_cohere
    from instructor.v2.providers.fireworks.client import from_fireworks
    from instructor.v2.providers.genai.client import from_genai
    from instructor.v2.providers.gemini.client import from_gemini
    from instructor.v2.providers.groq.client import from_groq
    from instructor.v2.providers.litellm.client import from_litellm
    from instructor.v2.providers.mistral.client import from_mistral
    from instructor.v2.providers.openrouter.client import from_openrouter
    from instructor.v2.providers.requesty.client import from_requesty
    from instructor.v2.providers.perplexity.client import from_perplexity
    from instructor.v2.providers.vertexai.client import from_vertexai
    from instructor.v2.providers.writer.client import from_writer
    from instructor.v2.providers.xai.client import from_xai


class User(BaseModel):
    name: str


class MistralClient(Protocol):
    chat: Any


def check_response_helpers(
    sync_response: Response, async_response: AsyncResponse
) -> None:
    assert_type(sync_response.create(response_model=User), User)
    assert_type(sync_response.create(response_model=User, token_budget=1_000), User)
    assert_type(sync_response.create(response_model=None), Any)
    assert_type(
        sync_response.create_with_completion(response_model=User),
        tuple[User, Any],
    )
    assert_type(
        sync_response.create_with_completion(response_model=None), tuple[Any, Any]
    )
    assert_type(
        sync_response.create_iterable(response_model=User),
        Generator[User, None, None],
    )
    assert_type(
        sync_response.create_iterable(response_model=None), Generator[Any, None, None]
    )
    assert_type(
        sync_response.create_partial(response_model=User),
        Generator[User, None, None],
    )
    assert_type(
        sync_response.create_partial(response_model=None), Generator[Any, None, None]
    )

    create_coro = async_response.create(response_model=User, token_budget=1_000)
    create_any_coro = async_response.create(response_model=None)
    completion_coro = async_response.create_with_completion(response_model=User)
    completion_any_coro = async_response.create_with_completion(response_model=None)
    iterable_coro = async_response.create_iterable(response_model=User)
    iterable_any_coro = async_response.create_iterable(response_model=None)
    partial_stream = async_response.create_partial(response_model=User)
    partial_any_stream = async_response.create_partial(response_model=None)

    assert_type(create_coro, CoroutineType[Any, Any, User])
    assert_type(
        create_any_coro,
        CoroutineType[Any, Any, Any],
    )
    assert_type(
        completion_coro,
        CoroutineType[Any, Any, tuple[User, Any]],
    )
    assert_type(
        completion_any_coro,
        CoroutineType[Any, Any, tuple[Any, Any]],
    )
    assert_type(
        iterable_coro,
        CoroutineType[Any, Any, AsyncGenerator[User, None]],
    )
    assert_type(
        iterable_any_coro,
        CoroutineType[Any, Any, AsyncGenerator[Any, None]],
    )
    assert_type(partial_stream, AsyncGenerator[User, None])
    assert_type(partial_any_stream, AsyncGenerator[Any, None])


def check_client_stream_helpers(
    sync_client: Instructor, async_client: AsyncInstructor
) -> None:
    assert_type(
        sync_client.create_iterable(response_model=User, messages=[]),
        Generator[User, None, None],
    )
    assert_type(
        sync_client.create_partial(response_model=User, messages=[]),
        Generator[User, None, None],
    )
    assert_type(
        async_client.create_iterable(response_model=User, messages=[]),
        AsyncGenerator[User, None],
    )
    assert_type(
        async_client.create_partial(response_model=User, messages=[]),
        AsyncGenerator[User, None],
    )


def check_openai_compatible_factories(
    sync_client: openai.OpenAI,
    async_client: openai.AsyncOpenAI,
) -> None:
    assert_type(from_anyscale("model"), Instructor)
    assert_type(from_anyscale("model", async_client=True), AsyncInstructor)
    assert_type(from_anyscale(sync_client), Instructor)
    assert_type(from_anyscale(async_client), AsyncInstructor)

    assert_type(from_together("model"), Instructor)
    assert_type(from_together("model", async_client=True), AsyncInstructor)
    assert_type(from_together(sync_client), Instructor)
    assert_type(from_together(async_client), AsyncInstructor)

    assert_type(from_databricks("model"), Instructor)
    assert_type(from_databricks("model", async_client=True), AsyncInstructor)
    assert_type(from_databricks(sync_client), Instructor)
    assert_type(from_databricks(async_client), AsyncInstructor)

    assert_type(from_deepseek("model"), Instructor)
    assert_type(from_deepseek("model", async_client=True), AsyncInstructor)
    assert_type(from_deepseek(sync_client), Instructor)
    assert_type(from_deepseek(async_client), AsyncInstructor)
    assert_type(from_openai(sync_client), Instructor)
    assert_type(from_openai(async_client), AsyncInstructor)


def check_auto_client_factory() -> None:
    assert_type(from_provider("openai/gpt-4o-mini"), Instructor)
    assert_type(from_provider("openai/gpt-4o-mini", async_client=True), AsyncInstructor)


def check_genai_factory(client: Any) -> None:
    assert_type(from_genai(client), Instructor)
    assert_type(from_genai(client, use_async=True), AsyncInstructor)


def _sync_completion(*_args: Any, **_kwargs: Any) -> object:
    return None


async def _async_completion(*_args: Any, **_kwargs: Any) -> object:
    return None


def check_litellm_factory() -> None:
    assert_type(from_litellm(_sync_completion), Instructor)
    assert_type(from_litellm(_async_completion, async_client=True), AsyncInstructor)


def check_bool_selected_factories(
    gemini_model: legacy_genai.GenerativeModel,
    vertex_model: gm.GenerativeModel,
    mistral_client: MistralClient,
) -> None:
    assert_type(from_gemini(gemini_model), Instructor)
    assert_type(from_gemini(gemini_model, use_async=True), AsyncInstructor)
    assert_type(from_vertexai(vertex_model), Instructor)
    assert_type(from_vertexai(vertex_model, use_async=True), AsyncInstructor)
    assert_type(from_mistral(mistral_client), Instructor)
    assert_type(from_mistral(mistral_client, use_async=True), AsyncInstructor)


def check_provider_factories(
    anthropic_sync: anthropic.Anthropic,
    anthropic_async: anthropic.AsyncAnthropic,
    bedrock_client: BaseClient,
    cerebras_sync: Cerebras,
    cerebras_async: AsyncCerebras,
    cohere_sync: cohere.Client,
    cohere_sync_v2: cohere.ClientV2,
    cohere_async: cohere.AsyncClient,
    cohere_async_v2: cohere.AsyncClientV2,
    fireworks_sync: Fireworks,
    groq_sync: groq.Groq,
    groq_async: groq.AsyncGroq,
    writer_sync: Writer,
    writer_async: AsyncWriter,
    xai_sync: SyncXAIClient,
    openai_sync: openai.OpenAI,
    openai_async: openai.AsyncOpenAI,
) -> None:
    assert_type(from_anthropic(anthropic_sync), Instructor)
    assert_type(from_anthropic(anthropic_async), AsyncInstructor)
    assert_type(from_bedrock(bedrock_client), Instructor)
    assert_type(from_bedrock(bedrock_client, async_client=True), AsyncInstructor)
    assert_type(from_cerebras(cerebras_sync), Instructor)
    assert_type(from_cerebras(cerebras_async), AsyncInstructor)
    assert_type(from_cohere(cohere_sync), Instructor)
    assert_type(from_cohere(cohere_sync_v2), Instructor)
    assert_type(from_cohere(cohere_async), AsyncInstructor)
    assert_type(from_cohere(cohere_async_v2), AsyncInstructor)
    assert_type(from_fireworks(fireworks_sync), Instructor)
    assert_type(from_groq(groq_sync), Instructor)
    assert_type(from_groq(groq_async), AsyncInstructor)
    assert_type(from_openrouter(openai_sync), Instructor)
    assert_type(from_openrouter(openai_async), AsyncInstructor)
    assert_type(from_requesty(openai_sync), Instructor)
    assert_type(from_requesty(openai_async), AsyncInstructor)
    assert_type(from_perplexity(openai_sync), Instructor)
    assert_type(from_perplexity(openai_async), AsyncInstructor)
    assert_type(from_writer(writer_sync), Instructor)
    assert_type(from_writer(writer_async), AsyncInstructor)
    assert_type(from_xai(xai_sync), Instructor)


def check_base_model_helpers() -> None:
    assert_type(response_schema(User), type[User])
    assert_type(Maybe(User), type[MaybeBase[User]])
    assert_type(Partial[User], type[User])


def check_parallel_wrapper(parallel: ParallelBase[User]) -> None:
    assert_type(
        parallel.from_response(response=None, mode=Mode.TOOLS),
        Generator[User, None, None],
    )
