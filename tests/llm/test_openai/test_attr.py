import instructor
import openai
import pytest


def test_has_embedding():
    oai = openai.OpenAI()
    client = instructor.from_openai(oai)

    embedding = client.embeddings.create(
        input="Hello world", model="text-embedding-3-small"
    )
    assert embedding is not None, "The 'embeddings' attribute is None."


@pytest.mark.asyncio
async def test_has_embedding_async():
    oai = openai.AsyncOpenAI()
    client = instructor.from_openai(oai)

    # Check if the 'embeddings' attribute can be accessed through the client
    embedding = await client.embeddings.create(
        input="Hello world", model="text-embedding-3-small"
    )
    assert embedding is not None, "The 'embeddings' attribute is None."
