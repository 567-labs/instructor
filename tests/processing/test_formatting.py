from __future__ import annotations

import pytest
from jinja2.exceptions import SecurityError
from instructor.templating import handle_templating
from instructor import Mode


def test_handle_insecure_template():
    with pytest.raises(SecurityError):
        kwargs = {
            "messages": [
                {
                    "role": "user",
                    "content": "{{ self.__init__.__globals__.__builtins__.__import__('os').system('ls') }} {{ variable }}",
                }
            ]
        }
        context = {"variable": "test"}
        handle_templating(kwargs, Mode.TOOLS, context)


def test_handle_templating_with_context():
    kwargs = {"messages": [{"role": "user", "content": "Hello {{ name }}!"}]}
    context = {"name": "Alice"}

    result = handle_templating(kwargs, Mode.TOOLS, context)

    assert result == {"messages": [{"role": "user", "content": "Hello Alice!"}]}


def test_handle_templating_without_context():
    kwargs = {"messages": [{"role": "user", "content": "Hello {{ name }}!"}]}

    result = handle_templating(kwargs, Mode.TOOLS)

    assert result == kwargs


def test_handle_templating_with_anthropic_format():
    kwargs = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Hello {{ name }}!"}]}
        ]
    }
    context = {"name": "Bob"}

    result = handle_templating(kwargs, Mode.TOOLS, context)

    assert result == {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Hello Bob!"}]}
        ]
    }


def test_handle_templating_with_mixed_content():
    kwargs = {
        "messages": [
            {"role": "user", "content": "Hello {{ name }}!"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Nice to meet you, {{ name }}!"}],
            },
        ]
    }
    context = {"name": "Charlie"}

    result = handle_templating(kwargs, Mode.TOOLS, context)

    assert result == {
        "messages": [
            {"role": "user", "content": "Hello Charlie!"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Nice to meet you, Charlie!"}],
            },
        ]
    }


def test_handle_templating_with_secret_context():
    from pydantic import BaseModel, SecretStr

    class UserContext(BaseModel):
        name: str
        address: SecretStr

    kwargs = {
        "messages": [
            {
                "role": "user",
                "content": "{{ user.name }}'s address is '{{ user.address.get_secret_value() }}'",
            }
        ]
    }
    context = {
        "user": UserContext(
            name="Jason", address=SecretStr("123 Secret St, Hidden City")
        )
    }

    result = handle_templating(kwargs, Mode.TOOLS, context)

    assert result == {
        "messages": [
            {
                "role": "user",
                "content": "Jason's address is '123 Secret St, Hidden City'",
            }
        ]
    }

    # Ensure the original SecretStr is not exposed when rendered
    assert str(context["user"].address) == "**********"


def test_handle_templating_with_cohere_format():
    kwargs = {
        "message": "Hello {{ name }}!",
        "chat_history": [{"message": "Previous message to {{ name }}"}],
    }
    context = {"name": "David"}

    result = handle_templating(kwargs, Mode.TOOLS, context)

    assert result == {
        "message": "Hello David!",
        "chat_history": [{"message": "Previous message to David"}],
    }


def test_handle_templating_with_gemini_format():
    kwargs = {
        "contents": [
            {"role": "user", "parts": ["Hello {{ name }}!", "How are you {{ name }}?"]}
        ]
    }
    context = {"name": "Eve"}

    result = handle_templating(kwargs, Mode.TOOLS, context)

    assert result == {
        "contents": [{"role": "user", "parts": ["Hello Eve!", "How are you Eve?"]}]
    }


def test_handle_templating_with_genai_multimodal_parts():
    """Non-text Parts (images/URIs) have text=None and must pass through untouched.

    Regression test for #2253: apply_template was being called with None,
    crashing Jinja2 with "Can't compile non template nodes".
    """
    from google.genai import types

    text_part = types.Part.from_text(text="Describe {{ subject }}")
    image_part = types.Part.from_uri(
        file_uri="gs://example/cat.png", mime_type="image/png"
    )
    content = types.Content(role="user", parts=[text_part, image_part])

    kwargs = {"contents": [content]}
    context = {"subject": "this image"}

    result = handle_templating(kwargs, Mode.GENAI_TOOLS, context)

    processed = result["contents"][0]
    assert processed.parts[0].text == "Describe this image"
    assert processed.parts[1].text is None
    assert processed.parts[1].file_data.file_uri == "gs://example/cat.png"


@pytest.mark.parametrize("thought", [True, False, None])
@pytest.mark.parametrize("text", ["The previous response.", ""])
def test_handle_templating_preserves_genai_text_part_metadata(
    thought: bool | None, text: str
) -> None:
    from google.genai import types

    history = types.Content(
        role="model",
        parts=[
            types.Part(
                text=text,
                thought=thought,
                thought_signature=b"previous-response-signature",
            )
        ],
    )
    user = types.Content(
        role="user",
        parts=[types.Part.from_text(text="Describe {{ subject }}")],
    )
    original_history = history.model_dump()
    original_user = user.model_dump()

    result = handle_templating(
        {"contents": [history, user]},
        Mode.GENAI_TOOLS,
        {"subject": "this image"},
    )

    assert result["contents"][0].model_dump() == original_history
    assert result["contents"][1].parts[0].text == "Describe this image"
    assert history.model_dump() == original_history
    assert user.model_dump() == original_user
