"""Offline regressions for text blocks in OpenAI-compatible responses."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import httpx
import pytest
from openai.types.chat import ChatCompletionMessage
from pydantic import BaseModel, ValidationError

import instructor
from instructor import Mode, Provider
from instructor.v2.core.registry import mode_registry
from tests.coverage._openai import chat_completion


class User(BaseModel):
    name: str


@pytest.mark.parametrize("mode", [Mode.JSON, Mode.JSON_SCHEMA, Mode.MD_JSON])
@pytest.mark.parametrize(
    "content",
    [
        pytest.param('{"name":"Ada"}', id="string"),
        pytest.param(
            [{"type": "text", "text": '{"name":"Ada"}', "thoughtSignature": "sig"}],
            id="text-block-with-metadata",
        ),
        pytest.param(
            [
                {"type": "text", "text": '{"name":"A'},
                {"type": "text", "text": 'da"}'},
            ],
            id="multiple-text-blocks",
        ),
        pytest.param(
            [
                {"type": "reasoning", "text": '{"name":"Wrong"}'},
                {"type": "text", "text": '{"name":"Ada"}'},
            ],
            id="ignore-non-text-blocks",
        ),
    ],
)
def test_json_response_content_blocks(mode: Mode, content: Any) -> None:
    response = chat_completion()
    # The SDK's permissive response construction retains provider-specific lists.
    response.choices[0].message = ChatCompletionMessage.model_construct(
        role="assistant", content=content
    )
    original = deepcopy(content)

    result = mode_registry.get_handlers(Provider.DATABRICKS, mode).response_parser(
        response, User, strict=True
    )

    assert result == User(name="Ada")
    assert getattr(result, "_raw_response", None) is response
    assert response.choices[0].message.content == original


@pytest.mark.parametrize("mode", [Mode.JSON, Mode.JSON_SCHEMA, Mode.MD_JSON])
@pytest.mark.parametrize(
    "content",
    [
        None,
        "",
        [],
        [{"type": "reasoning", "text": '{"name":"Ada"}'}],
        [None, {"type": "text"}, {"type": "text", "text": 1}],
    ],
    ids=["null", "empty-string", "empty-list", "no-text-blocks", "invalid-blocks"],
)
def test_json_response_without_text_still_fails_validation(
    mode: Mode, content: Any
) -> None:
    response = chat_completion()
    response.choices[0].message = ChatCompletionMessage.model_construct(
        role="assistant", content=content
    )

    with pytest.raises(ValidationError):
        mode_registry.get_handlers(Provider.DATABRICKS, mode).response_parser(
            response, User
        )


def test_databricks_md_json_text_blocks_through_sdk() -> None:
    content = [
        {
            "type": "text",
            "text": '```json\n{"tasks":[{"name":"Ada"}]}\n```',
            "thoughtSignature": "sig",
        }
    ]
    payload = chat_completion().model_dump()
    payload["choices"][0]["message"]["content"] = content

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/serving-endpoints/chat/completions"
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        client = instructor.from_provider(
            "databricks/databricks-gemini-3-5-flash",
            mode=Mode.MD_JSON,
            api_key="test-key",
            base_url="https://databricks.example",
            http_client=http_client,
        )
        result = client.create(
            response_model=list[User],
            messages=[{"role": "user", "content": "Extract Ada"}],
            max_retries=1,
        )

    assert result == [User(name="Ada")]
