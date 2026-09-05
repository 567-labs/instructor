from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from instructor.mode import Mode
from instructor.utils.providers import Provider
from instructor.v2.core.errors import InstructorRetryException

try:
    from instructor.v2 import from_genai
    from instructor.v2.core import mode_registry
    from google.genai import types
except ModuleNotFoundError:
    # fmt: off
    pytest.skip("google-genai package is not installed", allow_module_level=True)  # ty: ignore[too-many-positional-arguments]
    # fmt: on


class DummyModels:
    def __init__(self):
        self.called = False
        self.stream_called = False

    def generate_content(self, *_args, **_kwargs):
        self.called = True
        return types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(role="model", parts=[]))]
        )

    def generate_content_stream(self, *_args, **_kwargs):
        self.stream_called = True
        yield types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(role="model", parts=[]))]
        )


class DummyAsyncModels:
    def __init__(self):
        self.called = False

    async def generate_content(self, *_args, **_kwargs):
        self.called = True
        return types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(role="model", parts=[]))]
        )

    async def generate_content_stream(self, *_args, **_kwargs):
        self.called = True

        async def _gen():
            yield types.GenerateContentResponse(
                candidates=[
                    types.Candidate(content=types.Content(role="model", parts=[]))
                ]
            )

        return _gen()


class DummyClient:
    def __init__(self):
        self.models = DummyModels()
        self.aio = SimpleNamespace(models=DummyAsyncModels())


class WireResult(BaseModel):
    item_id: str


def _wire_response(item_id: str | None, signature: bytes):
    thought = types.Part.from_text(text="checking")
    thought.thought = True
    function_call = types.Part.from_function_call(
        name="WireResult", args={"item_id": item_id}
    )
    function_call.thought_signature = signature
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[thought, function_call],
                )
            )
        ]
    )


class RetryModels:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.requests: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any):
        self.requests.append(kwargs)
        return self.responses[len(self.requests) - 1]


class RetryClient:
    responses: list[Any] = []

    def __init__(self) -> None:
        self.models = RetryModels(self.responses)


def _assert_retry_history(
    request: dict[str, Any], expected_contents: list[Any]
) -> None:
    contents = request["contents"]
    assert len(contents) == 1 + 2 * len(expected_contents)
    for index, expected_content in enumerate(expected_contents):
        model_content = contents[1 + index * 2]
        function_response_content = contents[2 + index * 2]
        assert model_content == expected_content
        assert model_content.parts == expected_content.parts
        assert (
            model_content.parts[1].thought_signature
            == expected_content.parts[1].thought_signature
        )
        assert function_response_content.role == "user"
        assert function_response_content.role != "tool"
        function_response = function_response_content.parts[0].function_response
        assert function_response.name == "WireResult"
        assert "Validation Error found" in function_response.response["error"]


def test_genai_invalid_then_valid_preserves_signed_wire_history(monkeypatch):
    invalid = _wire_response(None, b"invalid-signature")
    valid = _wire_response("ok", b"valid-signature")
    RetryClient.responses = [invalid, valid]
    monkeypatch.setattr("instructor.v2.providers.genai.client.Client", RetryClient)
    client = RetryClient()
    instructor = from_genai(client, mode=Mode.TOOLS, use_async=False)

    result = instructor.chat.completions.create(
        model="gemini-test",
        messages=[{"role": "user", "content": "pick"}],
        response_model=WireResult,
        max_retries=2,
    )

    assert result.item_id == "ok"
    assert len(client.models.requests) == 2
    _assert_retry_history(client.models.requests[-1], [invalid.candidates[0].content])


def test_genai_three_invalid_attempts_preserve_bound(monkeypatch):
    invalid = [
        _wire_response(None, f"invalid-signature-{attempt}".encode())
        for attempt in range(1, 4)
    ]
    RetryClient.responses = invalid
    monkeypatch.setattr("instructor.v2.providers.genai.client.Client", RetryClient)
    client = RetryClient()
    instructor = from_genai(client, mode=Mode.TOOLS, use_async=False)

    with pytest.raises(InstructorRetryException) as exc_info:
        instructor.chat.completions.create(
            model="gemini-test",
            messages=[{"role": "user", "content": "pick"}],
            response_model=WireResult,
            max_retries=2,
        )

    assert exc_info.value.n_attempts == 3
    assert len(client.models.requests) == 3
    _assert_retry_history(
        client.models.requests[-1],
        [response.candidates[0].content for response in invalid[:2]],
    )


def test_mode_registry_has_genai_handlers():
    # Test generic modes
    assert mode_registry.is_registered(Provider.GENAI, Mode.TOOLS)
    assert mode_registry.is_registered(Provider.GENAI, Mode.JSON)
    # Legacy modes remain accepted through the v2 normalization shim.
    assert mode_registry.is_registered(Provider.GENAI, Mode.GENAI_TOOLS)
    assert mode_registry.is_registered(Provider.GENAI, Mode.GENAI_JSON)
    assert mode_registry.is_registered(Provider.GENAI, Mode.GENAI_STRUCTURED_OUTPUTS)


def test_from_genai_sync_generic_mode(monkeypatch):
    """Test using generic Mode.TOOLS."""
    monkeypatch.setattr(
        "instructor.v2.providers.genai.client.Client",
        DummyClient,
    )

    client = DummyClient()
    instructor = from_genai(client, mode=Mode.TOOLS, use_async=False)
    instructor.chat.completions.create(
        messages=[{"role": "user", "content": "Ping"}],
        response_model=None,
    )

    assert client.models.called


def test_from_genai_sync_legacy_mode_normalized(monkeypatch):
    """Legacy Mode.GENAI_TOOLS remains accepted in v2."""
    monkeypatch.setattr(
        "instructor.v2.providers.genai.client.Client",
        DummyClient,
    )

    client = DummyClient()
    instructor = from_genai(client, mode=Mode.GENAI_TOOLS, use_async=False)
    instructor.chat.completions.create(
        messages=[{"role": "user", "content": "Ping"}],
        response_model=None,
    )
    assert client.models.called


@pytest.mark.asyncio
async def test_from_genai_async_generic_mode(monkeypatch):
    """Test using generic Mode.TOOLS with async."""
    monkeypatch.setattr(
        "instructor.v2.providers.genai.client.Client",
        DummyClient,
    )
    client = DummyClient()
    instructor = from_genai(client, mode=Mode.TOOLS, use_async=True)
    await instructor.chat.completions.create(
        messages=[{"role": "user", "content": "Ping"}],
        response_model=None,
    )
    assert client.aio.models.called


@pytest.mark.asyncio
async def test_from_genai_async_legacy_mode_normalized(monkeypatch):
    """Legacy Mode.GENAI_TOOLS remains accepted in async v2."""
    monkeypatch.setattr(
        "instructor.v2.providers.genai.client.Client",
        DummyClient,
    )
    client = DummyClient()
    instructor = from_genai(client, mode=Mode.GENAI_TOOLS, use_async=True)
    await instructor.chat.completions.create(
        messages=[{"role": "user", "content": "Ping"}],
        response_model=None,
    )
    assert client.aio.models.called


def test_from_genai_json_mode(monkeypatch):
    """Test using generic Mode.JSON."""
    monkeypatch.setattr(
        "instructor.v2.providers.genai.client.Client",
        DummyClient,
    )

    client = DummyClient()
    instructor = from_genai(client, mode=Mode.JSON, use_async=False)
    instructor.chat.completions.create(
        messages=[{"role": "user", "content": "Ping"}],
        response_model=None,
    )

    assert client.models.called


def test_from_genai_json_legacy_mode_normalized(monkeypatch):
    """Legacy structured outputs mode remains accepted in v2."""
    monkeypatch.setattr(
        "instructor.v2.providers.genai.client.Client",
        DummyClient,
    )

    client = DummyClient()
    instructor = from_genai(
        client,
        mode=Mode.GENAI_STRUCTURED_OUTPUTS,
        use_async=False,
    )
    instructor.chat.completions.create(
        messages=[{"role": "user", "content": "Ping"}],
        response_model=None,
    )
    assert client.models.called
