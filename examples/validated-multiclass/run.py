from pydantic import BaseModel, ValidationInfo, model_validator
import openai
import instructor
import asyncio

client = instructor.from_openai(
    openai.AsyncOpenAI(),
)


class Tag(BaseModel):
    id: int
    name: str

    @model_validator(mode="after")
    def validate_ids(self, info: ValidationInfo):
        context = info.context
        if context:
            tags: list[Tag] = context.get("tags")
            assert self.id in {tag.id for tag in tags}, (
                f"Tag ID {self.id} not found in context"
            )
            assert self.name in {tag.name for tag in tags}, (
                f"Tag name {self.name} not found in context"
            )
        return self


class TagWithInstructions(Tag):
    instructions: str


class TagRequest(BaseModel):
    texts: list[str]
    tags: list[TagWithInstructions]


class TagResponse(BaseModel):
    texts: list[str]
    predictions: list[Tag]


async def tag_single_request(text: str, tags: list[Tag]) -> Tag:
    allowed_tags = [(tag.id, tag.name) for tag in tags]
    allowed_tags_str = ", ".join([f"`{tag}`" for tag in allowed_tags])
    return await client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {
                "role": "system",
                "content": "You are a world-class text tagging system.",
            },
            {"role": "user", "content": f"Describe the following text: `{text}`"},
            {
                "role": "user",
                "content": f"Here are the allowed tags: {allowed_tags_str}",
            },
        ],
        response_model=Tag,
        # Minizises the hallucination of tags that are not in the allowed tags.
        validation_context={"tags": tags},
    )


async def tag_request(request: TagRequest) -> TagResponse:
    predictions = await asyncio.gather(
        *[tag_single_request(text, request.tags) for text in request.texts]
    )
    return TagResponse(
        texts=request.texts,
        predictions=predictions,
    )


if __name__ == "__main__":
    # Tags will be a range of different topics.
    # Such as personal, phone, email, etc.
    tags = [
        TagWithInstructions(id=0, name="personal", instructions="Personal information"),
        TagWithInstructions(id=1, name="phone", instructions="Phone number"),
        TagWithInstructions(id=2, name="email", instructions="Email address"),
        TagWithInstructions(id=3, name="address", instructions="Address"),
        TagWithInstructions(id=4, name="Other", instructions="Other information"),
    ]

    # Texts will be a range of different questions.
    # Such as "How much does it cost?", "What is your privacy policy?", etc.
    texts = [
        "What is your phone number?",
        "What is your email address?",
        "What is your address?",
        "What is your privacy policy?",
    ]

    # The request will contain the texts and the tags.
    request = TagRequest(texts=texts, tags=tags)

    # The response will contain the texts, the predicted tags, and the confidence.
    response = asyncio.run(tag_request(request))
    print(response.model_dump_json(indent=2))
