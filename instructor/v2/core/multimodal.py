from __future__ import annotations
import base64
import re
from collections.abc import Mapping, Hashable
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Union,
    TypedDict,
    TypeVar,
    cast,
)
from pathlib import Path
from urllib.parse import urlparse
import mimetypes

import requests
from pydantic import BaseModel, Field

from instructor.v2.core.errors import MultimodalError
from instructor.v2.core.mode import Mode
from instructor.v2.core.remote import (
    MAX_AUDIO_BYTES,
    MAX_IMAGE_BYTES,
    MAX_PDF_BYTES,
    fetch_remote_content,
    probe_remote_content_type,
)

mimetypes.add_type("image/webp", ".webp")

F = TypeVar("F", bound=Callable[..., Any])
K = TypeVar("K", bound=Hashable)
V = TypeVar("V")

# OpenAI source: https://platform.openai.com/docs/guides/vision/what-type-of-files-can-i-upload
# Anthropic source: https://docs.anthropic.com/en/docs/build-with-claude/vision#ensuring-image-quality
VALID_MIME_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"]
VALID_AUDIO_MIME_TYPES = [
    "audio/aac",
    "audio/flac",
    "audio/mp3",
    "audio/m4a",
    "audio/mpeg",
    "audio/mpga",
    "audio/mp4",
    "audio/opus",
    "audio/pcm",
    "audio/wav",
    "audio/webm",
]
VALID_PDF_MIME_TYPES = ["application/pdf"]
CacheControlType = Mapping[str, str]
OptionalCacheControlType = Optional[CacheControlType]


def _normalize_media_type(media_type: str | None) -> str | None:
    """Return the lowercase media type without optional HTTP parameters."""
    if media_type is None:
        return None
    return media_type.split(";", 1)[0].strip().lower()


def _normalize_audio_media_type(media_type: str | None) -> str | None:
    """Use the same canonical audio MIME types for every source."""
    if media_type is None:
        return None
    return {
        "audio/x-wav": "audio/wav",
        "audio/vnd.dlna.adts": "audio/aac",
    }.get(media_type, media_type)


class ImageParamsBase(TypedDict):
    type: Literal["image"]
    source: str


class ImageParams(ImageParamsBase, total=False):
    cache_control: CacheControlType


class Image(BaseModel):
    """Image content loaded from a URL, path, base64 data, or bytes."""

    source: Union[str, Path, bytes] = Field(  # noqa: UP007
        description="URL, file path, base64 data, or raw bytes of the image"
    )
    media_type: str = Field(description="MIME type of the image")
    data: Union[str, None] = Field(  # noqa: UP007
        None, description="Base64 encoded image data", repr=False
    )

    @classmethod
    def autodetect(cls, source: str | Path | bytes) -> Image:
        """Attempt to autodetect an image from a source string, Path, or bytes."""
        if isinstance(source, str):
            if cls.is_base64(source):
                return cls.from_base64(source)
            if source.startswith(("http://", "https://")):
                return cls.from_url(source)
            if source.startswith("gs://"):
                return cls.from_gs_url(source)
            # Since detecting the max length of a file universally cross-platform is difficult,
            # we'll just try/catch the Path conversion and file check
            try:
                path = Path(source)
                if path.is_file():
                    return cls.from_path(path)
            except OSError:
                pass  # Fall through to raw base64 attempt

            return cls.from_raw_base64(source)

        if isinstance(source, Path):
            return cls.from_path(source)

        if isinstance(source, bytes):
            encoded = base64.b64encode(source).decode("utf-8")
            return cls.from_raw_base64(encoded)

        raise ValueError(f"Unsupported image source type: {type(source).__name__}")

    @classmethod
    def autodetect_safely(cls, source: Union[str, Path, bytes]) -> Union[Image, str]:  # noqa: UP007
        """Safely attempt to autodetect an image from a source string or path.

        Args:
            source (Union[str,path]): The source string or path.
        Returns:
            An Image if the source is detected to be a valid image, otherwise
            the source itself as a string.
        """
        try:
            return cls.autodetect(source)
        except ValueError:
            return str(source)

    @classmethod
    def is_base64(cls, s: str) -> bool:
        return bool(re.match(r"^data:image/[a-zA-Z]+;base64,", s))

    @classmethod  # Caching likely unnecessary
    def from_base64(cls, data_uri: str) -> Image:
        header, encoded = data_uri.split(",", 1)
        media_type = header.split(":")[1].split(";")[0]
        if media_type not in VALID_MIME_TYPES:
            raise MultimodalError(
                f"Unsupported image format: {media_type}. Supported formats: {', '.join(VALID_MIME_TYPES)}",
                content_type="image",
            )
        return cls(
            source=data_uri,
            media_type=media_type,
            data=encoded,
        )

    @classmethod
    def from_gs_url(cls, data_uri: str, timeout: int = 30) -> Image:
        """
        Create an Image instance from a Google Cloud Storage URL.

        Args:
            data_uri: GCS URL starting with gs://
            timeout: Request timeout in seconds (default: 30)
        """
        if not data_uri.startswith("gs://"):
            raise ValueError("URL must start with gs://")

        public_url = f"https://storage.googleapis.com/{data_uri[5:]}"

        try:
            response = fetch_remote_content(
                public_url,
                max_bytes=MAX_IMAGE_BYTES,
                timeout=timeout,
            )
            media_type = response.content_type
            if media_type not in VALID_MIME_TYPES:
                raise ValueError(f"Unsupported image format: {media_type}")

            data = base64.b64encode(response.content).decode("utf-8")

            return cls(source=data_uri, media_type=media_type, data=data)
        except requests.RequestException as e:
            raise ValueError(
                "Failed to access GCS image (must be publicly readable)"
            ) from e

    @classmethod  # Caching likely unnecessary
    def from_raw_base64(cls, data: str) -> Image:
        try:
            decoded = base64.b64decode(data)

            # Detect image type from file signature (magic bytes)
            # This replaces imghdr which was removed in Python 3.13
            img_type = None
            if decoded.startswith(b"\xff\xd8\xff"):
                img_type = "jpeg"
            elif decoded.startswith(b"\x89PNG\r\n\x1a\n"):
                img_type = "png"
            elif decoded.startswith(b"GIF87a") or decoded.startswith(b"GIF89a"):
                img_type = "gif"
            elif decoded.startswith(b"RIFF") and decoded[8:12] == b"WEBP":
                img_type = "webp"

            if img_type:
                return cls(
                    source=data,
                    media_type=f"image/{img_type}",
                    data=data,
                )
            raise ValueError(f"Unsupported image type: {img_type}")
        except Exception as e:
            raise ValueError(f"Invalid or unsupported base64 image data") from e

    @classmethod
    def from_url(cls, url: str) -> Image:
        if url.startswith("gs://"):
            return cls.from_gs_url(url)
        if cls.is_base64(url):
            return cls.from_base64(url)

        parsed_url = urlparse(url)
        media_type, _ = mimetypes.guess_type(parsed_url.path)

        if not media_type:
            try:
                media_type = probe_remote_content_type(url, timeout=30)
            except requests.RequestException as e:
                raise ValueError(f"Failed to fetch image from URL") from e

        if media_type not in VALID_MIME_TYPES:
            raise ValueError(f"Unsupported image format: {media_type}")
        return cls(source=url, media_type=media_type, data=None)

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> Image:  # noqa: UP007
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")

        if path.stat().st_size == 0:
            raise ValueError("Image file is empty")

        media_type, _ = mimetypes.guess_type(str(path))
        if media_type not in VALID_MIME_TYPES:
            raise ValueError(f"Unsupported image format: {media_type}")

        data = base64.b64encode(path.read_bytes()).decode("utf-8")
        return cls(source=path, media_type=media_type, data=data)

    @staticmethod
    def url_to_base64(url: str) -> str:
        """Safely fetch an image URL and return its base64 representation."""
        response = fetch_remote_content(url, max_bytes=MAX_IMAGE_BYTES, timeout=30)
        return base64.b64encode(response.content).decode("utf-8")

    def to_anthropic(self) -> dict[str, Any]:
        from instructor.v2.providers.anthropic.multimodal import image_to_anthropic

        return image_to_anthropic(self)

    def to_openai(self, mode: Mode) -> dict[str, Any]:
        from instructor.v2.providers.openai.multimodal import image_to_openai

        return image_to_openai(self, mode)

    def to_genai(self):
        from instructor.v2.providers.genai.multimodal import image_to_genai

        return image_to_genai(self)


class Audio(BaseModel):
    """Represents an audio that can be loaded from a URL or file path."""

    source: Union[str, Path] = Field(description="URL or file path of the audio")  # noqa: UP007
    data: Union[str, None] = Field(  # noqa: UP007
        None, description="Base64 encoded audio data", repr=False
    )
    media_type: str = Field(description="MIME type of the audio")

    @classmethod
    def autodetect(cls, source: str | Path) -> Audio:
        """Attempt to autodetect an audio from a source string or Path."""
        if isinstance(source, str):
            if cls.is_base64(source):
                return cls.from_base64(source)
            if source.startswith(("http://", "https://")):
                return cls.from_url(source)
            if source.startswith("gs://"):
                return cls.from_gs_url(source)
            # Since detecting the max length of a file universally cross-platform is difficult,
            # we'll just try/catch the Path conversion and file check
            try:
                path = Path(source)
                if path.is_file():
                    return cls.from_path(path)
            except OSError:
                pass  # Fall through to error

            raise ValueError("Unable to determine audio source")

        if isinstance(source, Path):
            return cls.from_path(source)

        raise ValueError(f"Unsupported audio source type: {type(source).__name__}")

    @classmethod
    def autodetect_safely(cls, source: Union[str, Path]) -> Union[Audio, str]:  # noqa: UP007
        """Safely attempt to autodetect an audio from a source string or path.

        Args:
            source (Union[str,path]): The source string or path.
        Returns:
            An Audio if the source is detected to be a valid audio, otherwise
            the source itself as a string.
        """
        try:
            return cls.autodetect(source)
        except ValueError:
            return str(source)

    @classmethod
    def is_base64(cls, s: str) -> bool:
        return bool(re.match(r"^data:audio/[a-zA-Z0-9+.-]+;base64,", s))

    @classmethod
    def from_base64(cls, data_uri: str) -> Audio:
        header, encoded = data_uri.split(",", 1)
        media_type = _normalize_audio_media_type(header.split(":")[1].split(";")[0])
        if media_type not in VALID_AUDIO_MIME_TYPES:
            raise ValueError(f"Unsupported audio format: {media_type}")
        return cls(
            source=data_uri,
            media_type=media_type,
            data=encoded,
        )

    @classmethod
    def from_url(cls, url: str) -> Audio:
        """Create an Audio instance from a URL."""
        if url.startswith("gs://"):
            return cls.from_gs_url(url)
        response = fetch_remote_content(url, max_bytes=MAX_AUDIO_BYTES, timeout=30)
        content_type = _normalize_audio_media_type(response.content_type)
        if content_type not in VALID_AUDIO_MIME_TYPES:
            raise ValueError(
                f"Unsupported audio format: {content_type}. "
                f"Must be one of: {', '.join(VALID_AUDIO_MIME_TYPES)}"
            )

        data = base64.b64encode(response.content).decode("utf-8")
        return cls(source=url, data=data, media_type=content_type)

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> Audio:  # noqa: UP007
        """Create an Audio instance from a file path."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        if path.stat().st_size == 0:
            raise ValueError("Audio file is empty")

        mime_type = _normalize_audio_media_type(mimetypes.guess_type(str(path))[0])

        if mime_type not in VALID_AUDIO_MIME_TYPES:
            raise ValueError(
                f"Unsupported audio format: {mime_type}. "
                f"Must be one of: {', '.join(VALID_AUDIO_MIME_TYPES)}"
            )

        data = base64.b64encode(path.read_bytes()).decode("utf-8")
        return cls(source=str(path), data=data, media_type=mime_type)

    @classmethod
    def from_gs_url(cls, data_uri: str, timeout: int = 30) -> Audio:
        """
        Create an Audio instance from a Google Cloud Storage URL.

        Args:
            data_uri: GCS URL starting with gs://
            timeout: Request timeout in seconds (default: 30)
        """
        if not data_uri.startswith("gs://"):
            raise ValueError("URL must start with gs://")

        public_url = f"https://storage.googleapis.com/{data_uri[5:]}"

        try:
            response = fetch_remote_content(
                public_url,
                max_bytes=MAX_AUDIO_BYTES,
                timeout=timeout,
            )
            media_type = _normalize_audio_media_type(response.content_type)
            if media_type not in VALID_AUDIO_MIME_TYPES:
                raise ValueError(f"Unsupported audio format: {media_type}")

            data = base64.b64encode(response.content).decode("utf-8")

            return cls(source=data_uri, media_type=media_type, data=data)
        except requests.RequestException as e:
            raise ValueError(
                "Failed to access GCS audio (must be publicly readable)"
            ) from e

    def to_openai(self, mode: Mode) -> dict[str, Any]:
        from instructor.v2.providers.openai.multimodal import audio_to_openai

        return audio_to_openai(self, mode)

    def to_anthropic(self) -> dict[str, Any]:
        from instructor.v2.providers.anthropic.multimodal import audio_to_anthropic

        return audio_to_anthropic(self)

    def to_genai(self):
        from instructor.v2.providers.genai.multimodal import audio_to_genai

        return audio_to_genai(self)


class ImageWithCacheControl(Image):
    """Image with Anthropic prompt caching support."""

    cache_control: OptionalCacheControlType = Field(
        None, description="Optional Anthropic cache control image"
    )

    @classmethod
    def from_image_params(cls, image_params: ImageParams) -> Image:
        source = image_params["source"]
        cache_control = image_params.get("cache_control")
        base_image = Image.autodetect(source)
        return cls(
            source=base_image.source,
            media_type=base_image.media_type,
            data=base_image.data,
            cache_control=cache_control,
        )

    def to_anthropic(self) -> dict[str, Any]:
        from instructor.v2.providers.anthropic.multimodal import (
            image_with_cache_control_to_anthropic,
        )

        return image_with_cache_control_to_anthropic(self)


class PDF(BaseModel):
    source: str | Path = Field(description="URL, file path, or base64 data of the PDF")
    media_type: str = Field(
        description="MIME type of the PDF", default="application/pdf"
    )
    data: str | None = Field(None, description="Base64 encoded PDF data", repr=False)

    @classmethod
    def autodetect(cls, source: str | Path) -> PDF:
        """Attempt to autodetect a PDF from a source string or Path.
        Args:
            source (Union[str,path]): The source string or path.
        Returns:
            A PDF if the source is detected to be a valid PDF.
        Raises:
            ValueError: If the source is not detected to be a valid PDF.
        """
        if isinstance(source, str):
            if cls.is_base64(source):
                return cls.from_base64(source)
            if source.startswith(("http://", "https://")):
                return cls.from_url(source)
            if source.startswith("gs://"):
                return cls.from_gs_url(source)

            try:
                if Path(source).is_file():
                    return cls.from_path(source)
            except FileNotFoundError as err:
                raise MultimodalError(
                    "PDF file not found",
                    content_type="pdf",
                    file_path=str(source),
                ) from err
            except OSError as e:
                if e.errno == 63:  # File name too long
                    raise MultimodalError(
                        "PDF file name too long",
                        content_type="pdf",
                        file_path=str(source),
                    ) from e
                raise MultimodalError(
                    "Unable to read PDF file",
                    content_type="pdf",
                    file_path=str(source),
                ) from e

            return cls.from_raw_base64(source)
        if isinstance(source, Path):
            return cls.from_path(source)

        raise ValueError(f"Unsupported PDF source type: {type(source).__name__}")

    @classmethod
    def autodetect_safely(cls, source: Union[str, Path]) -> Union[PDF, str]:  # noqa: UP007
        """Safely attempt to autodetect a PDF from a source string or path.

        Args:
            source (Union[str,path]): The source string or path.
        Returns:
            A PDF if the source is detected to be a valid PDF, otherwise
            the source itself as a string.
        """
        try:
            return cls.autodetect(source)
        except ValueError:
            return str(source)

    @classmethod
    def is_base64(cls, s: str) -> bool:
        return bool(re.match(r"^data:application/pdf;base64,", s))

    @classmethod
    def from_base64(cls, data_uri: str) -> PDF:
        header, encoded = data_uri.split(",", 1)
        media_type = header.split(":")[1].split(";")[0]
        if media_type not in VALID_PDF_MIME_TYPES:
            raise ValueError(f"Unsupported PDF format: {media_type}")
        return cls(
            source=data_uri,
            media_type=media_type,
            data=encoded,
        )

    @classmethod
    def from_path(cls, path: str | Path) -> PDF:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"PDF file not found: {path}")

        if path.stat().st_size == 0:
            raise ValueError("PDF file is empty")

        media_type, _ = mimetypes.guess_type(str(path))
        if media_type not in VALID_PDF_MIME_TYPES:
            raise ValueError(f"Unsupported PDF format: {media_type}")

        data = base64.b64encode(path.read_bytes()).decode("utf-8")
        return cls(source=path, media_type=media_type, data=data)

    @classmethod
    def from_raw_base64(cls, data: str) -> PDF:
        try:
            decoded = base64.b64decode(data)
            # Check if it's a valid PDF by looking for the PDF header
            if decoded.startswith(b"%PDF-"):
                return cls(
                    source=data,
                    media_type="application/pdf",
                    data=data,
                )
            raise ValueError("Invalid PDF format")
        except Exception as e:
            raise ValueError("Invalid or unsupported base64 PDF data") from e

    @classmethod
    def from_gs_url(cls, data_uri: str, timeout: int = 30) -> PDF:
        """
        Create a PDF instance from a Google Cloud Storage URL.

        Args:
            data_uri: GCS URL starting with gs://
            timeout: Request timeout in seconds (default: 30)
        """
        if not data_uri.startswith("gs://"):
            raise ValueError("URL must start with gs://")

        public_url = f"https://storage.googleapis.com/{data_uri[5:]}"

        try:
            response = fetch_remote_content(
                public_url,
                max_bytes=MAX_PDF_BYTES,
                timeout=timeout,
            )
            media_type = response.content_type or "application/pdf"
            if media_type not in VALID_PDF_MIME_TYPES:
                raise ValueError(f"Unsupported PDF format: {media_type}")

            data = base64.b64encode(response.content).decode("utf-8")

            return cls(source=data_uri, media_type=media_type, data=data)
        except requests.RequestException as e:
            raise ValueError(
                "Failed to access GCS PDF (must be publicly readable)"
            ) from e

    @classmethod
    def from_url(cls, url: str) -> PDF:
        if url.startswith("gs://"):
            return cls.from_gs_url(url)
        parsed_url = urlparse(url)
        media_type, _ = mimetypes.guess_type(parsed_url.path)

        if not media_type:
            try:
                media_type = probe_remote_content_type(url, timeout=30)
            except requests.RequestException as e:
                raise ValueError("Failed to fetch PDF from URL") from e

        if media_type not in VALID_PDF_MIME_TYPES:
            raise ValueError(f"Unsupported PDF format: {media_type}")
        return cls(source=url, media_type=media_type, data=None)

    def to_mistral(self) -> dict[str, Any]:
        from instructor.v2.providers.mistral.multimodal import pdf_to_mistral

        return pdf_to_mistral(self)

    def to_openai(self, mode: Mode) -> dict[str, Any]:
        from instructor.v2.providers.openai.multimodal import pdf_to_openai

        return pdf_to_openai(self, mode)

    def to_anthropic(self) -> dict[str, Any]:
        from instructor.v2.providers.anthropic.multimodal import pdf_to_anthropic

        return pdf_to_anthropic(self)

    def to_genai(self):
        from instructor.v2.providers.genai.multimodal import pdf_to_genai

        return pdf_to_genai(self)

    def to_bedrock(self, name: str | None = None) -> dict[str, Any]:
        """Convert to Bedrock's document format."""
        # Determine the document name
        if name is None:
            if isinstance(self.source, Path):
                name = self.source.name
            elif isinstance(self.source, str):
                # Try to extract filename from path or URL
                if self.source.startswith(("http://", "https://", "gs://")):
                    name = Path(urlparse(self.source).path).name or "document"
                else:
                    name = (
                        Path(self.source).name
                        if Path(self.source).exists()
                        else "document"
                    )
            else:
                name = "document"

        # Sanitize name according to Bedrock requirements
        # Only allow alphanumeric, whitespace (max one in row), hyphens, parentheses, square brackets
        name = re.sub(r"[^\w\s\-\(\)\[\]]", "", name)
        name = re.sub(r"\s+", " ", name)  # Consolidate whitespace
        name = name.strip()

        # Handle S3 URIs
        if isinstance(self.source, str) and self.source.startswith("s3://"):
            if not re.match(r"s3://[^/]+/.*", self.source):
                raise ValueError(f"Invalid S3 URI format: {self.source}")

            # Note: bucketOwner is optional but recommended for cross-account access
            return {
                "document": {
                    "format": "pdf",
                    "name": name,
                    "source": {
                        "s3Location": {
                            "uri": self.source
                            # "bucketOwner": "account-id"  # Optional, can be added by user
                        }
                    },
                }
            }

        # Handle bytes-based sources (base64 only)
        if not self.data:
            raise ValueError(
                "PDF data is missing. Provide base64-encoded data or use an s3:// source."
            )
        # Decode base64 data to bytes
        pdf_bytes = base64.b64decode(self.data)

        return {
            "document": {"format": "pdf", "name": name, "source": {"bytes": pdf_bytes}}
        }


class PDFWithCacheControl(PDF):
    """PDF with Anthropic prompt caching support."""

    def to_anthropic(self) -> dict[str, Any]:
        from instructor.v2.providers.anthropic.multimodal import (
            pdf_with_cache_control_to_anthropic,
        )

        return pdf_with_cache_control_to_anthropic(self)


class PDFWithGenaiFile(PDF):
    @classmethod
    def from_new_genai_file(
        cls, file_path: str, retry_delay: int = 10, max_retries: int = 20
    ) -> PDFWithGenaiFile:
        from instructor.v2.providers.genai.multimodal import upload_new_pdf_file

        return upload_new_pdf_file(cls, file_path, retry_delay, max_retries)

    @classmethod
    def from_existing_genai_file(cls, file_name: str) -> PDFWithGenaiFile:
        from instructor.v2.providers.genai.multimodal import load_existing_pdf_file

        return load_existing_pdf_file(cls, file_name)

    def to_genai(self):
        from instructor.v2.providers.genai.multimodal import uploaded_pdf_to_genai

        return uploaded_pdf_to_genai(self)


def convert_contents(
    contents: Union[  # noqa: UP007
        str,
        dict[str, Any],
        Image,
        Audio,
        PDF,
        list[Union[str, dict[str, Any], Image, Audio, PDF]],  # noqa: UP007
    ],
    mode: Mode,
) -> Union[str, list[dict[str, Any]]]:  # noqa: UP007
    """Convert content items to the appropriate format based on the specified mode."""
    if isinstance(contents, str):
        return contents
    if isinstance(contents, (Image, Audio, PDF)) or isinstance(contents, dict):
        contents = [contents]

    converted_contents: list[dict[str, Union[str, Image]]] = []  # noqa: UP007
    text_file_type = (
        "input_text"
        if mode in {Mode.RESPONSES_TOOLS, Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS}
        else "text"
    )
    for content in contents:
        if isinstance(content, str):
            converted_contents.append({"type": text_file_type, "text": content})
        elif isinstance(content, dict):
            converted_contents.append(content)
        elif isinstance(content, (Image, Audio, PDF)):
            if mode in {
                Mode.ANTHROPIC_JSON,
                Mode.ANTHROPIC_TOOLS,
                Mode.ANTHROPIC_REASONING_TOOLS,
            }:
                converted_contents.append(content.to_anthropic())
            elif mode in {Mode.GEMINI_JSON, Mode.GEMINI_TOOLS}:
                raise NotImplementedError("Gemini is not supported yet")
            elif mode in {
                Mode.MISTRAL_STRUCTURED_OUTPUTS,
                Mode.MISTRAL_TOOLS,
            } and isinstance(content, (PDF)):
                converted_contents.append(content.to_mistral())
            else:
                converted_contents.append(content.to_openai(mode))
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")
    return converted_contents


def autodetect_media(
    source: str | Path | Image | Audio | PDF,
) -> Image | Audio | PDF | str:
    """Autodetect images, audio, or PDFs from a given source.

    Args:
        source: URL, file path, Path, or data URI to inspect.

    Returns:
        The detected :class:`Image`, :class:`Audio`, or :class:`PDF` instance.
        If detection fails, the original source is returned.
    """
    if isinstance(source, (Image, Audio, PDF)):
        return source

    # Normalize once for cheap checks and mimetype guess
    source = str(source)

    if source.startswith("data:image/"):
        return Image.autodetect_safely(source)
    if source.startswith("data:audio/"):
        return Audio.autodetect_safely(source)
    if source.startswith("data:application/pdf"):
        return PDF.autodetect_safely(source)

    media_type, _ = mimetypes.guess_type(source)
    if media_type in VALID_MIME_TYPES:
        return Image.autodetect_safely(source)
    if media_type in VALID_AUDIO_MIME_TYPES:
        return Audio.autodetect_safely(source)
    if media_type in VALID_PDF_MIME_TYPES:
        return PDF.autodetect_safely(source)

    for cls in (Image, Audio, PDF):
        item = cls.autodetect_safely(source)  # type: ignore[arg-type]
        if not isinstance(item, str):
            return item
    return source


def convert_messages(
    messages: list[
        dict[
            str,
            Union[  # noqa: UP007
                str,
                dict[str, Any],
                Image,
                Audio,
                PDF,
                list[Union[str, dict[str, Any], Image, Audio, PDF]],  # noqa: UP007
            ],
        ]
    ],
    mode: Mode,
    autodetect_images: bool = False,
) -> list[dict[str, Any]]:
    """Convert messages to the appropriate format based on the specified mode."""
    converted_messages: list[dict[str, Any]] = []

    def is_image_params(x: Any) -> bool:
        return isinstance(x, dict) and x.get("type") == "image" and "source" in x

    for message in messages:
        if "type" in message:
            if message["type"] in {"audio", "image"}:
                converted_messages.append(message)
                continue
            raise ValueError(f"Unsupported message type: {message['type']}")
        role = message["role"]
        content = message["content"] or []
        other_kwargs = {
            k: v for k, v in message.items() if k not in ["role", "content", "type"]
        }
        if autodetect_images:
            if isinstance(content, list):
                new_content: list[str | dict[str, Any] | Image | Audio | PDF] = []  # noqa: UP007
                for item in content:
                    if isinstance(item, str):
                        new_content.append(autodetect_media(item))
                    elif is_image_params(item):
                        new_content.append(
                            ImageWithCacheControl.from_image_params(
                                cast(ImageParams, item)
                            )
                        )
                    else:
                        new_content.append(item)
                content = new_content
            elif isinstance(content, str):
                content = autodetect_media(content)
            elif is_image_params(content):
                content = ImageWithCacheControl.from_image_params(
                    cast(ImageParams, content)
                )
        if isinstance(content, str):
            converted_messages.append(
                {"role": role, "content": content, **other_kwargs}
            )
        else:
            # At this point content is narrowed to non-str types accepted by convert_contents
            converted_content = convert_contents(content, mode)
            converted_messages.append(
                {"role": role, "content": converted_content, **other_kwargs}
            )
    return converted_messages


def extract_genai_multimodal_content(
    contents: list[Any],
    autodetect_images: bool = True,
):
    """Compatibility wrapper for the GenAI-owned multimodal converter."""
    from instructor.v2.providers.genai.multimodal import extract_multimodal_content

    return extract_multimodal_content(contents, autodetect_images)
