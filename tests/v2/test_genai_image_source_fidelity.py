"""Image encodings preserve downloaded GCS bytes across supported SDK versions."""

import base64

import pytest

from instructor.processing.multimodal import Image

pytest.importorskip("google.genai")

# A complete synthetic 1x1 PNG; no remote storage or provider request is needed.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aXioAAAAASUVORK5CYII="
)


@pytest.mark.parametrize("source", ["gs://example/image.png", "image.png"])
def test_genai_image_decodes_base64_payload(source: str) -> None:
    # This is the public Image state returned after from_gs_url downloads an image.
    image = Image(
        source=source,
        media_type="image/png",
        data=base64.b64encode(PNG).decode("ascii"),
    )

    part = image.to_genai()

    assert part.inline_data is not None
    assert part.inline_data.mime_type == "image/png"
    assert part.inline_data.data == PNG
