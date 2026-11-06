import uuid

import instructor
import openai
from burr.core import action, State, ApplicationBuilder
from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema
from youtube_transcript_api import YouTubeTranscriptApi


class QuestionAnswer(BaseModel):
    question: str = Field(description="Question about the topic")
    options: list[str] = Field(
        description="Potential answers to the question.", min_items=3, max_items=5
    )
    answer_index: int = Field(
        description="Index of the correct answer options (starting from 0).", ge=0, lt=5
    )
    difficulty: int = Field(
        description="Difficulty of this question from 1 to 5, 5 being the most difficult.",
        gt=0,
        le=5,
    )
    youtube_url: SkipJsonSchema[str | None] = None
    id: uuid.UUID = Field(description="Unique identifier", default_factory=uuid.uuid4)


@action(reads=[], writes=["youtube_url"])
def process_user_input(state: State, user_input: str) -> State:
    """Process user input and update the YouTube URL."""
    youtube_url = (
        user_input  # In practice, we would have more complex validation logic.
    )
    return state.update(youtube_url=youtube_url)


@action(reads=["youtube_url"], writes=["transcript"])
def get_youtube_transcript(state: State) -> State:
    """Get the official YouTube transcript for a video given it's URL"""
    youtube_url = state["youtube_url"]

    _, _, video_id = youtube_url.partition("?v=")
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    full_transcript = " ".join([entry["text"] for entry in transcript])

    # store the transcript in state
    return state.update(transcript=full_transcript, youtube_url=youtube_url)


@action(reads=["transcript", "youtube_url"], writes=["question_answers"])
def generate_question_and_answers(state: State) -> State:
    """Generate `QuestionAnswer` from a YouTube transcript using an LLM."""
    # read the transcript from state
    transcript = state["transcript"]
    youtube_url = state["youtube_url"]

    # create the instructor client
    instructor_client = instructor.from_openai(openai.OpenAI())
    system_prompt = (
        "Analyze the given YouTube transcript and generate question-answer pairs"
        " to help study and understand the topic better. Please rate all questions from 1 to 5"
        " based on their difficulty."
    )
    response = instructor_client.chat.completions.create_iterable(
        model="gpt-4o-mini",
        response_model=QuestionAnswer,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript},
        ],
    )

    # iterate over QuestionAnswer, add the `youtube_url`, and append to state
    for qna in response:
        qna.youtube_url = youtube_url
        # `State` is immutable, so `.append()` returns a new object with the appended value
        state = state.append(question_answers=qna)

    return state


def build_application():
    return (
        ApplicationBuilder()
        .with_actions(
            process_user_input,
            get_youtube_transcript,
            generate_question_and_answers,
        )
        .with_transitions(
            ("process_user_input", "get_youtube_transcript"),
            ("get_youtube_transcript", "generate_question_and_answers"),
            ("generate_question_and_answers", "process_user_input"),
        )
        .with_entrypoint("process_user_input")
        .with_tracker(project="youtube-qna")
        .build()
    )


if __name__ == "__main__":
    app = build_application()

    while True:
        user_input = input("Enter a YouTube URL (q to quit): ")
        if user_input.lower() == "q":
            break

        action_name, result, state = app.run(
            halt_before=["process_user_input"],
            inputs={"user_input": user_input},
        )
        print(f"{len(state['question_answers'])} question-answer pairs generated")

        print("Preview:\n")
        count = 0
        for qna in state["question_answers"]:
            if count > 3:
                break
            print(qna.question)
            print(qna.options)
            print()
            count += 1
