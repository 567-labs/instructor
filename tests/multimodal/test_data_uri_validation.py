import pytest

from instructor.v2.core.errors import MultimodalError
from instructor.v2.core.multimodal import Audio, Image, PDF


@pytest.mark.parametrize("media_class, content_type", [(Image, "image"), (Audio, "audio"), (PDF, "pdf")])
@pytest.mark.parametrize("bad_uri", ["data:image/png;base64", "image/png;base64,AAAA", ",AAAA", "data:,AAAA"])
def test_a_malformed_data_uri_is_named(media_class, content_type, bad_uri):
    with pytest.raises(MultimodalError, match="Invalid data URI") as error:
        media_class.from_base64(bad_uri)

    assert error.value.content_type == content_type


def test_a_valid_data_uri_still_splits():
    image = Image.from_base64("data:image/png;base64,AAAA")

    assert image.media_type == "image/png"
    assert image.data == "AAAA"
