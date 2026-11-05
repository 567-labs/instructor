"""Azure resource routing and authentication using the installed OpenAI SDK."""

import json

import instructor
import openai
import pytest
from pydantic import BaseModel

from instructor.v2.core.errors import ConfigurationError
from instructor.v2.core.providers import Provider
from tests.openai_http import AsyncClient, Client, MockTransport, Request, Response


class Contact(BaseModel):
    name: str


def completion(request: Request) -> Response:
    assert json.loads(request.content)["model"] == "contact-deployment"
    return Response(
        200,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "contact-deployment",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": '{"name":"Avery"}'},
                }
            ],
        },
    )


def test_azure_deployment_uses_its_resource_key():
    def handler(request: Request) -> Response:
        assert (
            request.url.path
            == "/openai/deployments/contact-deployment/chat/completions"
        )
        assert request.url.params["api-version"] == "2024-10-21"
        assert request.headers["api-key"] == "unit-azure-key"
        return completion(request)

    with openai.AzureOpenAI(
        api_key="unit-azure-key",
        azure_endpoint="https://resource.openai.azure.com",
        api_version="2024-10-21",
        http_client=Client(transport=MockTransport(handler)),
    ) as sdk:
        client = instructor.from_azure(
            sdk, mode=instructor.Mode.JSON, model="contact-deployment"
        )
        assert client.provider is Provider.AZURE_OPENAI
        assert (
            client.create(
                response_model=Contact, messages=[{"role": "user", "content": "Avery"}]
            ).name
            == "Avery"
        )


@pytest.mark.asyncio
async def test_foundry_v1_accepts_the_async_openai_sdk():
    def handler(request: Request) -> Response:
        assert request.url.path == "/openai/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer unit-foundry-key"
        return completion(request)

    async with openai.AsyncOpenAI(
        base_url="https://resource.services.ai.azure.com/openai/v1/",
        api_key="unit-foundry-key",
        http_client=AsyncClient(transport=MockTransport(handler)),
    ) as sdk:
        client = instructor.from_azure(
            sdk, mode=instructor.Mode.JSON, model="contact-deployment"
        )
        assert client.provider is Provider.AZURE_OPENAI
        assert (
            await client.create(
                response_model=Contact, messages=[{"role": "user", "content": "Avery"}]
            )
        ).name == "Avery"


@pytest.mark.parametrize("prefix", ["azure", "azure_openai"])
def test_azure_aliases_and_entra_credentials(prefix, monkeypatch):
    for key in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_AD_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "unit-unrelated-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://resource.openai.azure.com")
    with pytest.raises(ConfigurationError, match="AZURE_OPENAI_API_KEY"):
        instructor.from_provider(f"{prefix}/contact-deployment")

    def handler(request: Request) -> Response:
        assert request.url.path == "/openai/responses"
        assert request.headers["Authorization"] == "Bearer unit-entra-token"
        assert "api-key" not in request.headers
        return Response(
            200,
            json={
                "id": "resp-local",
                "created_at": 1,
                "object": "response",
                "model": "contact-deployment",
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call-local",
                        "name": "Contact",
                        "arguments": '{"name":"Avery"}',
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            },
        )

    with Client(transport=MockTransport(handler)) as transport:
        client = instructor.from_provider(
            f"{prefix}/contact-deployment",
            azure_ad_token_provider=lambda: "unit-entra-token",
            http_client=transport,
            mode=instructor.Mode.RESPONSES_TOOLS,
        )
        assert client.provider is Provider.AZURE_OPENAI
        assert isinstance(client.client, openai.AzureOpenAI)
        result = client.create(
            response_model=Contact, messages=[{"role": "user", "content": "Avery"}]
        )
        assert result.name == "Avery"


@pytest.mark.parametrize(
    "old,new",
    [
        (instructor.Mode.FUNCTIONS, instructor.Mode.TOOLS),
        (instructor.Mode.TOOLS_STRICT, instructor.Mode.TOOLS),
        (instructor.Mode.JSON_O1, instructor.Mode.JSON_SCHEMA),
        (instructor.Mode.AZURE_TOOLS, instructor.Mode.TOOLS),
        (instructor.Mode.AZURE_JSON, instructor.Mode.JSON),
        (
            instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
            instructor.Mode.RESPONSES_TOOLS,
        ),
    ],
)
def test_azure_normalizes_legacy_modes(old, new):
    with openai.AzureOpenAI(
        api_key="unit-azure-key",
        azure_endpoint="https://resource.openai.azure.com",
        api_version="2024-10-21",
    ) as sdk:
        assert instructor.from_azure(sdk, mode=old).mode is new
