from __future__ import annotations
import pytest
from instructor.v2.providers.bedrock.handlers import _normalize_bedrock_image_format


@pytest.mark.parametrize(
    "inp,expected",
    [
        ("image/jpeg", "jpeg"),
        ("image/jpg", "jpeg"),
        ("jpg", "jpeg"),
        ("jpeg", "jpeg"),
        ("image/pjpeg", "jpeg"),
        ("image/png", "png"),
        ("png", "png"),
        ("image/gif", "gif"),
        ("gif", "gif"),
        ("image/webp", "webp"),
        ("webp", "webp"),
        ("", "jpeg"),
        (None, "jpeg"),
        ("image/whatever", "jpeg"),
    ],
)
def test_normalize_bedrock_image_format(inp, expected):
    assert _normalize_bedrock_image_format(inp) == expected
