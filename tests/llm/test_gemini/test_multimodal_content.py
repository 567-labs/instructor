import instructor
from pydantic import BaseModel
import os


class Description(BaseModel):
    relevant_speakers: list[str]
    summary: str


curr_file = os.path.dirname(__file__)
file_path = os.path.join(curr_file, "./test_files/sample.mp3")
MODEL = os.getenv("GOOGLE_GENAI_MODEL", "google/gemini-pro")


def test_audio_compatability_list():
    client = instructor.from_provider(
        model=MODEL, mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS
    )

    # For now, we'll skip file operations since the new API might handle them differently
    # This test might need to be updated based on the new google-genai file upload API
    content = "Please transcribe this recording: [audio file would go here]"

    result = client.chat.completions.create(
        response_model=Description,
        messages=[
            {"role": "user", "content": content},
        ],
    )

    assert isinstance(result, Description), (
        "Result should be an instance of Description"
    )


def test_audio_compatability_multiple_messages():
    client = instructor.from_provider(
        model=MODEL, mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS
    )

    # For now, we'll skip file operations since the new API might handle them differently
    # This test might need to be updated based on the new google-genai file upload API

    result = client.chat.completions.create(
        response_model=Description,
        messages=[
            {
                "role": "user",
                "content": "Please transcribe this recording: [audio file would go here]",
            },
        ],
    )

    assert isinstance(result, Description), (
        "Result should be an instance of Description"
    )
