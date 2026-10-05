"""reask_gemini_json interpolated `response.text` unguarded; the SDK's `.text`
raises ValueError on blocked/empty responses, so the reask handler itself blew
up instead of producing the retry prompt. Mirrors parse_gemini_json's guard."""

from types import SimpleNamespace

from instructor.v2.providers.gemini.handlers import reask_gemini_json


class _BlockedResponse(SimpleNamespace):
    @property
    def text(self) -> str:
        raise ValueError("response.text is not available for blocked responses")


def test_reask_gemini_json_surfaces_validation_error_instead_of_valueerror() -> None:
    kwargs = {"contents": []}
    out = reask_gemini_json(kwargs, _BlockedResponse(), ValueError("bad json"))
    text = out["contents"][-1]["parts"][0]
    assert "bad json" in text
    assert "<no readable text" in text


def test_reask_gemini_json_keeps_normal_path() -> None:
    kwargs = {"contents": []}
    response = SimpleNamespace(text='{"name": "x"')
    out = reask_gemini_json(kwargs, response, ValueError("bad json"))
    text = out["contents"][-1]["parts"][0]
    assert '{"name": "x"' in text
