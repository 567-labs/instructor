"""Data URIs with optional parameters (RFC 2397) must still be autodetected."""

import pytest

from instructor.processing.multimodal import (
    PDF,
    Audio,
    Image,
    autodetect_media,
)

BASE64_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    # noqa: E501
)


def test_image_autodetect_accepts_data_uri_with_parameters() -> None:
    uri = "data:image/png;charset=utf-8;base64," + BASE64_PNG
    assert Image.is_base64(uri)
    assert isinstance(autodetect_media(uri), Image)


def test_audio_autodetect_accepts_data_uri_with_parameters() -> None:
    uri = "data:audio/wav;bitrate=8;base64,UklGRiQAAABXQVZF"
    assert Audio.is_base64(uri)
    assert isinstance(autodetect_media(uri), Audio)


def test_pdf_autodetect_accepts_data_uri_with_parameters() -> None:
    uri = "data:application/pdf;charset=utf-8;base64,JVBERi0xLjQK"
    assert PDF.is_base64(uri)
    assert isinstance(autodetect_media(uri), PDF)


def test_autodetect_media_does_not_degrade_parameterized_data_uri() -> None:
    uri = "data:image/png;charset=utf-8;base64," + BASE64_PNG
    assert not isinstance(autodetect_media(uri), str)


def test_non_base64_data_uri_is_still_rejected() -> None:
    uri = "data:image/png;charset=utf-8,not-a-base64-payload"
    assert not Image.is_base64(uri)
    assert isinstance(autodetect_media(uri), str)
