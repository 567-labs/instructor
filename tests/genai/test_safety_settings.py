from instructor.v2.providers.gemini.utils import update_genai_kwargs


def test_update_genai_kwargs_safety_settings_excludes_image_categories():
    """IMAGE_* harm categories must never be sent to the standard Gemini API.

    They are only supported by Vertex AI and cause 400 INVALID_ARGUMENT errors
    on the standard Gemini API.
    See: https://github.com/567-labs/instructor/issues/2146
    """
    from google.genai import types

    kwargs = {
        "contents": [
            types.Content(
                role="user",
                parts=[types.Part.from_bytes(data=b"123", mime_type="image/png")],
            )
        ]
    }
    base_config = {}

    result = update_genai_kwargs(kwargs, base_config)

    assert "safety_settings" in result
    assert isinstance(result["safety_settings"], list)
    # No IMAGE_* categories should be present
    for setting in result["safety_settings"]:
        assert not setting["category"].name.startswith("HARM_CATEGORY_IMAGE_"), (
            f"IMAGE_ category {setting['category'].name} must not be sent to the "
            "standard Gemini API"
        )


def test_update_genai_kwargs_text_only_categories_for_image_content():
    """Even with image content, only text harm categories should be used."""
    from google.genai import types
    from google.genai.types import HarmBlockThreshold, HarmCategory

    custom_safety = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
    }

    kwargs = {
        "contents": [
            types.Content(
                role="user",
                parts=[types.Part.from_bytes(data=b"123", mime_type="image/png")],
            )
        ],
        "safety_settings": custom_safety,
    }
    base_config = {}

    result = update_genai_kwargs(kwargs, base_config)

    # Custom threshold should be preserved for text category
    found_hate_speech = False
    for setting in result["safety_settings"]:
        assert not setting["category"].name.startswith("HARM_CATEGORY_IMAGE_")
        if setting["category"] == HarmCategory.HARM_CATEGORY_HATE_SPEECH:
            assert setting["threshold"] == HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
            found_hate_speech = True
    assert found_hate_speech, "HARM_CATEGORY_HATE_SPEECH should be in safety_settings"


def test_handle_genai_tools_autodetect_images_excludes_image_categories():
    """Autodetected image content should NOT use IMAGE_* harm categories."""
    from pydantic import BaseModel

    from instructor.v2.providers.gemini.utils import handle_genai_tools

    class SimpleModel(BaseModel):
        text: str

    data_uri = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO6q0S8AAAAASUVORK5CYII="
    )

    kwargs = {
        "messages": [
            {
                "role": "user",
                "content": ["What is in this image?", data_uri],
            }
        ]
    }

    _, out = handle_genai_tools(SimpleModel, kwargs, autodetect_images=True)

    assert "config" in out
    assert out["config"].safety_settings is not None
    # No IMAGE_* categories should be present
    assert not any(
        s.category.name.startswith("HARM_CATEGORY_IMAGE_")
        for s in out["config"].safety_settings
    ), "IMAGE_ categories must not be sent to the standard Gemini API"
