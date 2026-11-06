from openai import OpenAI
from pydantic import BaseModel
import instructor
from instructor.processing.multimodal import Audio
import base64

client = instructor.from_openai(OpenAI())


class Person(BaseModel):
    name: str
    age: int


with open("./output.wav", "rb") as f:
    encoded_string = base64.b64encode(f.read()).decode("utf-8")

resp = client.chat.completions.create(
    model="gpt-4o-audio-preview",
    response_model=Person,
    modalities=["text"],
    audio={"voice": "alloy", "format": "wav"},
    messages=[
        {
            "role": "user",
            "content": [
                "Extract the following information from the audio",
                Audio.from_path("./output.wav"),
            ],
        },
    ],
)  # type: ignore

print(resp)
# > Person(name='Jason', age=20)
