import pytest


pytest.importorskip("google.genai")


from google.genai import types

from instructor.v2.providers.gemini.utils import reask_genai_tools


def _response_with_content(content: types.Content) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(candidates=[types.Candidate(content=content)])


def test_reask_genai_tools_preserves_thought_signature():
    function_call_part = types.Part.from_function_call(
        name="test_fn", args={"value": 1}
    )
    function_call_part.thought_signature = b"sig"
    model_content = types.Content(role="model", parts=[function_call_part])

    original_kwargs = {"contents": []}
    result = reask_genai_tools(
        kwargs=original_kwargs,
        response=_response_with_content(model_content),
        exception=Exception("boom"),
    )

    assert original_kwargs["contents"] == []
    assert result["contents"][-2] is model_content
    assert result["contents"][-2].parts[0].thought_signature == b"sig"

    tool_content = result["contents"][-1]
    assert tool_content.role == "user"
    assert tool_content.parts[0].function_response.name == "test_fn"


def test_reask_genai_tools_finds_function_call_part_when_not_first():
    function_call_part = types.Part.from_function_call(
        name="test_fn", args={"value": 1}
    )
    model_content = types.Content(
        role="model",
        parts=[
            types.Part.from_text(text="some preface"),
            function_call_part,
        ],
    )

    result = reask_genai_tools(
        kwargs={"contents": []},
        response=_response_with_content(model_content),
        exception=Exception("boom"),
    )

    assert result["contents"][-2] is model_content
    assert result["contents"][-1].role == "user"


def test_reask_genai_tools_handles_none_response():
    result = reask_genai_tools(
        kwargs={},
        response=None,
        exception=Exception("boom"),
    )

    assert result["contents"][-1].role == "user"


def test_reask_genai_tools_falls_back_when_no_function_call():
    model_content = types.Content(
        role="model",
        parts=[types.Part.from_text(text="not a function call")],
    )

    result = reask_genai_tools(
        kwargs={"contents": []},
        response=_response_with_content(model_content),
        exception=Exception("boom"),
    )

    assert result["contents"][0] is model_content
    assert result["contents"][1].role == "user"
