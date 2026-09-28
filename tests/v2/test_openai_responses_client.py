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


def test_responses_tools_preserves_caller_tools_and_sets_tool_choice_auto() -> None:
    from pydantic import BaseModel
    from instructor.v2.core.registry import mode_registry
    from instructor.v2.core.providers import Provider

    class Answer(BaseModel):
        value: int

    prepare = mode_registry.get_handler(Provider.OPENAI, Mode.RESPONSES_TOOLS)
    caller_tool = {"type": "code_interpreter", "container": {"type": "auto"}}
    _, request = prepare(
        Answer,
        {
            "messages": [{"role": "user", "content": "Sum the first 2000 primes."}],
            "tools": [caller_tool],
        },
    )

    # Caller tool is preserved alongside the function schema tool
    assert caller_tool in request["tools"]
    assert any(
        t.get("type") == "function" and t.get("name") == "Answer"
        for t in request["tools"]
    )
    assert len(request["tools"]) == 2
    # tool_choice is auto so model can execute built-in tools before answering
    assert request["tool_choice"] == "auto"


def test_responses_tools_forces_tool_choice_without_caller_tools() -> None:
    from pydantic import BaseModel
    from instructor.v2.core.registry import mode_registry
    from instructor.v2.core.providers import Provider

    class Answer(BaseModel):
        value: int

    prepare = mode_registry.get_handler(Provider.OPENAI, Mode.RESPONSES_TOOLS)
    _, request = prepare(
        Answer,
        {
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert len(request["tools"]) == 1
    assert request["tools"][0]["name"] == "Answer"
    assert request["tool_choice"] == {"type": "function", "name": "Answer"}


def test_responses_tools_parse_response_with_builtin_and_intermediate_tools() -> None:
    from types import SimpleNamespace
    from pydantic import BaseModel
    from instructor.v2.providers.openai.handlers import OpenAIResponsesToolsHandler

    class Answer(BaseModel):
        value: int

    handler = OpenAIResponsesToolsHandler()

    item_code = SimpleNamespace(type="code_interpreter_call")
    item_search = SimpleNamespace(
        type="function_call", name="search", arguments='{"query": "primes"}'
    )
    item_answer = SimpleNamespace(
        type="function_call", name="Answer", arguments='{"value": 16274627}'
    )

    response = SimpleNamespace(output=[item_code, item_search, item_answer])
    result = handler.parse_response(response, Answer)
    assert isinstance(result, Answer)
    assert result.value == 16274627
