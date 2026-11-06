from unittest.mock import Mock

import pytest

from instructor.v2.core.remote import MAX_IMAGE_BYTES, RemoteContent
from instructor.v2.providers.bedrock import handlers


@pytest.mark.parametrize(
    ("text_type", "remote"), [("text", False), ("input_text", True)]
)
def test_bedrock_converts_openai_text_and_images(
    text_type, remote, tiny_png_data_url, tiny_png_bytes, monkeypatch
):
    image_url = "https://images.example.com/sample.png"
    fetch = Mock(return_value=RemoteContent(tiny_png_bytes, "image/png", image_url))
    monkeypatch.setattr(handlers, "fetch_remote_content", fetch)
    content = [
        {"type": text_type, "text": "Describe the image."},
        {
            "type": "image_url",
            "image_url": {"url": image_url if remote else tiny_png_data_url},
        },
    ]

    assert handlers._to_bedrock_content_items(content) == [
        {"text": "Describe the image."},
        {"image": {"format": "png", "source": {"bytes": tiny_png_bytes}}},
    ]
    if remote:
        fetch.assert_called_once_with(image_url, max_bytes=MAX_IMAGE_BYTES, timeout=30)
    else:
        fetch.assert_not_called()
