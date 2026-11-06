from unittest.mock import Mock

import openai

from instructor import Mode
from instructor.v2.providers.openai.client import from_openai


def test_responses_mode_uses_responses_api() -> None:
    client = Mock(spec=openai.OpenAI)
    client.chat = Mock()
    client.chat.completions = Mock()
    client.chat.completions.create = Mock()
    client.responses = Mock()
    client.responses.create = Mock()

    instructor_client = from_openai(client, mode=Mode.RESPONSES_TOOLS)

    instructor_client.responses.create(
        messages=[{"role": "user", "content": "hello"}],
        response_model=None,
    )

    client.responses.create.assert_called_once()
    client.chat.completions.create.assert_not_called()
