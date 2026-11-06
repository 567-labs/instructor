import pytest
import vertexai.generative_models as gm
from instructor.v2.providers.vertexai.handlers import vertexai_message_parser


def test_vertexai_message_parser_string_content():
    message = {"role": "user", "content": "Hello, world!"}
    result = vertexai_message_parser(message)

    assert isinstance(result, gm.Content)
    assert result.role == "user"
    assert len(result.parts) == 1
    assert isinstance(result.parts[0], gm.Part)
    assert result.parts[0].text == "Hello, world!"


def test_vertexai_message_parser_list_content():
    message = {
        "role": "user",
        "content": [
            "Hello, ",
            gm.Part.from_text("world!"),
            gm.Part.from_text(" How are you?"),
        ],
    }
    result = vertexai_message_parser(message)

    assert isinstance(result, gm.Content)
    assert result.role == "user"
    assert len(result.parts) == 3
    assert isinstance(result.parts[0], gm.Part)
    assert isinstance(result.parts[1], gm.Part)
    assert isinstance(result.parts[2], gm.Part)
    assert result.parts[0].text == "Hello, "
    assert result.parts[1].text == "world!"
    assert result.parts[2].text == " How are you?"


def test_vertexai_message_parser_invalid_content():
    message = {"role": "user", "content": 123}  # Invalid content type

    with pytest.raises(ValueError, match="Unsupported message content type"):
        vertexai_message_parser(message)


def test_vertexai_message_parser_invalid_list_item():
    message = {"role": "user", "content": ["Hello", 123, gm.Part.from_text("world!")]}

    with pytest.raises(ValueError, match="Unsupported content type in list"):
        vertexai_message_parser(message)
