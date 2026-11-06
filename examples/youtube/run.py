import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from youtube_transcript_api import YouTubeTranscriptApi
from rich.console import Console
from rich.table import Table
from rich.live import Live

client = instructor.from_openai(OpenAI())


class Chapter(BaseModel):
    start_ts: float = Field(
        ...,
        description="The start timestamp indicating when the chapter starts in the video.",
    )
    end_ts: float = Field(
        ...,
        description="The end timestamp indicating when the chapter ends in the video.",
    )
    title: str = Field(
        ..., description="A concise and descriptive title for the chapter."
    )
    summary: str = Field(
        ...,
        description="A brief summary of the chapter's content, don't use words like 'the speaker'",
    )


def get_youtube_transcript(video_id: str) -> str:
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join(
            [f"ts={entry['start']} - {entry['text']}" for entry in transcript]
        )
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return ""


def extract_chapters(transcript: str):
    class Chapters(BaseModel):
        chapters: list[Chapter]

    return client.chat.completions.create_partial(
        model="gpt-4o",  # You can experiment with different models
        response_model=Chapters,
        messages=[
            {
                "role": "system",
                "content": "Analyze the given YouTube transcript and extract chapters. For each chapter, provide a start timestamp, end timestamp, title, and summary.",
            },
            {"role": "user", "content": transcript},
        ],
    )


if __name__ == "__main__":
    video_id = input("Enter a Youtube Url: ")
    video_id = video_id.split("v=")[1]
    console = Console()

    with console.status("[bold green]Processing YouTube URL...") as status:
        transcripts = get_youtube_transcript(video_id)
        status.update("[bold blue]Generating Clips...")
        chapters = extract_chapters(transcripts)

        table = Table(title="Video Chapters")
        table.add_column("Title", style="magenta")
        table.add_column("Description", style="green")
        table.add_column("Start", style="cyan")
        table.add_column("End", style="cyan")

        with Live(refresh_per_second=4) as live:
            for extraction in chapters:
                if not extraction.chapters:
                    continue

                new_table = Table(title="Video Chapters")
                new_table.add_column("Title", style="magenta")
                new_table.add_column("Description", style="green")
                new_table.add_column("Start", style="cyan")
                new_table.add_column("End", style="cyan")

                for chapter in extraction.chapters:
                    new_table.add_row(
                        chapter.title,
                        chapter.summary,
                        f"{chapter.start_ts:.2f}" if chapter.start_ts else "",
                        f"{chapter.end_ts:.2f}" if chapter.end_ts else "",
                    )
                    new_table.add_row("", "", "", "")  # Add an empty row for spacing

                live.update(new_table)

    console.print("\nChapter extraction complete!")
