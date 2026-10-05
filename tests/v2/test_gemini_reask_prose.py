"""reask_gemini_tools crashed with an unrelated AttributeError when the model
answered in prose instead of calling the tool, so the retry loop aborted on
attempt 1 with ``'NoneType' object has no attribute 'name'``. The sibling
GenAI/OpenAI/mistral reask handlers already fall back to a plain user
correction; this one now mirrors them."""

from types import SimpleNamespace

from instructor.v2.providers.gemini.handlers import reask_gemini_tools


def _prose_response() -> SimpleNamespace:
    part = SimpleNamespace(
        text="Sure! The user is Alice, 30 years old.", function_call=None
    )
    return SimpleNamespace(
        text="Sure! The user is Alice, 30 years old.",
        candidates=[
            SimpleNamespace(
                finish_reason=1,
                content=SimpleNamespace(parts=[part], role="model"),
            )
        ],
        parts=[part],
    )


def test_reask_gemini_tools_falls_back_to_user_correction_when_no_function_call() -> (
    None
):
    kwargs = {"contents": []}
    out = reask_gemini_tools(kwargs, _prose_response(), ValueError("bad schema"))
    assert out["contents"][-1]["role"] == "user"
    assert "bad schema" in out["contents"][-1]["parts"][0]


def test_reask_gemini_tools_keeps_function_call_path() -> None:
    fc = SimpleNamespace(name="Order", args={"drug": "x"})
    part = SimpleNamespace(text=None, function_call=fc)
    response = SimpleNamespace(text=None, candidates=None, parts=[part])
    kwargs = {"contents": []}

    out = reask_gemini_tools(kwargs, response, ValueError("bad schema"))

    assert out["contents"][0]["role"] == "model"
