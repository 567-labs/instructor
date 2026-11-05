import pytest
from tests.llm.test_openai.util import models
from instructor.processing.multimodal import Image, Audio
import instructor
from pydantic import Field, BaseModel
from itertools import product
import requests
from pathlib import Path
import base64
import os

audio_url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/gettysburg.wav"
image_url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/image.jpg"

pdf_url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/invoice.pdf"
curr_file = os.path.dirname(__file__)
pdf_path = os.path.join(curr_file, "../../assets/invoice.pdf")
pdf_base64 = base64.b64encode(open(pdf_path, "rb").read()).decode("utf-8")
pdf_base64_string = f"data:application/pdf;base64,{pdf_base64}"


modes = [
    instructor.Mode.TOOLS,
]


class LineItem(BaseModel):
    name: str
    price: int
    quantity: int


class Receipt(BaseModel):
    total: int
    items: list[str]


def gettysburg_audio():
    audio_file = Path("gettysburg.wav")
    if not audio_file.exists():
        response = requests.get(audio_url)
        response.raise_for_status()
        with open(audio_file, "wb") as f:
            f.write(response.content)
    return audio_file


@pytest.mark.parametrize(
    "audio_file, mode",
    [(Audio.from_url(audio_url), mode) for mode in modes],
)
def test_multimodal_audio_description(audio_file, mode, client):
    client = instructor.from_openai(client, mode=mode)

    if client.mode in {
        instructor.Mode.RESPONSES_TOOLS,
        instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
    }:
        pytest.skip("Audio isn't supported in responses for now")

    class AudioDescription(BaseModel):
        source: str

    response = client.chat.completions.create(
        model="gpt-audio-1.5",
        response_model=AudioDescription,
        modalities=["text"],
        messages=[
            {
                "role": "user",
                "content": [
                    "Where's this excerpt from?",
                    audio_file,
                ],  # type: ignore
            },
        ],
        audio={"voice": "alloy", "format": "wav"},  # type: ignore
    )


class ImageDescription(BaseModel):
    objects: list[str] = Field(..., description="The objects in the image")
    scene: str = Field(..., description="The scene of the image")
    colors: list[str] = Field(..., description="The colors in the image")


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_multimodal_image_description(model, mode, client):
    client = instructor.from_openai(client, mode=mode)
    response = client.chat.completions.create(
        model=model,  # Ensure this is a vision-capable model
        response_model=ImageDescription,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can describe images",
            },
            {
                "role": "user",
                "content": [
                    "What is this?",
                    Image.from_url(image_url),
                ],  # type: ignore
            },
        ],
    )

    # Assertions to validate the response
    assert isinstance(response, ImageDescription)
    assert len(response.objects) > 0
    assert response.scene != ""
    assert len(response.colors) > 0

    # Additional assertions can be added based on expected content of the sample image


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_multimodal_image_description_autodetect(model, mode, client):
    client = instructor.from_openai(client, mode=mode)
    response = client.chat.completions.create(
        model=model,  # Ensure this is a vision-capable model
        response_model=ImageDescription,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can describe images",
            },
            {
                "role": "user",
                "content": [
                    "What is this?",
                    image_url,
                ],  # type: ignore
            },
        ],
        autodetect_images=True,  # type: ignore
    )

    # Assertions to validate the response
    assert isinstance(response, ImageDescription)
    assert len(response.objects) > 0
    assert response.scene != ""
    assert len(response.colors) > 0

    # Additional assertions can be added based on expected content of the sample image


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_multimodal_image_description_autodetect_no_response_model(model, mode, client):
    client = instructor.from_openai(client, mode=mode)
    response = client.chat.completions.create(
        response_model=None,
        model=model,  # Ensure this is a vision-capable model
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can describe images. "
                "If looking at an image, reply with 'This is an image' and nothing else.",
            },
            {
                "role": "user",
                "content": image_url,
            },
        ],
        max_tokens=1000,
        temperature=1,
        autodetect_images=True,
    )

    if mode not in {
        instructor.Mode.RESPONSES_TOOLS,
        instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
    }:
        assert response.choices[0].message.content.startswith("This is an image")
    else:
        assert response.output[0].content[0].text


@pytest.mark.parametrize("pdf_source", [pdf_path, pdf_url, pdf_base64_string])
@pytest.mark.parametrize("model, mode", product(models, modes))
def test_multimodal_pdf_file(model, mode, client, pdf_source):
    client = instructor.from_openai(client, mode=mode)

    # Retry logic for flaky LLM responses
    max_retries = 3
    for attempt in range(max_retries):
        response = client.chat.completions.create(
            model=model,  # Ensure this is a vision-capable model
            messages=[
                {
                    "role": "system",
                    "content": "Extract the total and items from the invoice. Be precise and only extract the final total amount and list of item names. The total should be exactly 220.",
                },
                {
                    "role": "user",
                    "content": instructor.processing.multimodal.PDF.autodetect(
                        pdf_source
                    ),
                },
            ],
            autodetect_images=False,
            response_model=Receipt,
            temperature=0,  # Keep for consistent responses
        )

        if response.total == 220 and len(response.items) == 2:
            break
        elif attempt == max_retries - 1:
            pytest.fail(
                f"After {max_retries} attempts, got total={response.total}, items={response.items}, expected total=220, items=2"
            )

    assert response.total == 220
    assert len(response.items) == 2
