import os
from typing import Optional, Literal
import asyncio
from openai import AsyncOpenAI
import typer
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from pydantic import BaseModel, Field
import instructor
import frontmatter

console = Console()
client = instructor.from_openai(AsyncOpenAI())


async def generate_ai_frontmatter(
    client: AsyncOpenAI, title: str, content: str, categories: list[str]
):
    """
    Generate a description and categories for the given content using AI.

    Args:
        client (AsyncOpenAI): The AsyncOpenAI client.
        title (str): The title of the markdown file.
        content (str): The content of the file.
        categories (List[str]): List of all available categories.

    Returns:
        DescriptionAndCategories: The generated description, categories, tags, and reasoning.
    """

    class DescriptionAndCategories(BaseModel):
        description: str
        reasoning: str = Field(
            ..., description="The reasoning for the correct categories"
        )
        tags: list[str]
        categories: list[
            Literal[
                "OpenAI",
                "Anthropic",
                "LLama",
                "LLM Observability",
                "Data Processing",
                "Python",
                "LLM Techniques",
                "Pydantic",
                "Performance Optimization",
                "Data Validation",
                "API Development",
                "Retrieval Augmented Generation",
            ]
        ]

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant that generates SEO-friendly descriptions for markdown files.",
            },
            {"role": "user", "content": f"Title: {title}\n\nContent: {content}"},
            {
                "role": "user",
                "content": f"Based on the title and content, generate a brief description (max 160 characters) that would be suitable for SEO purposes. Also, select up to 3 relevant categories from the following list: {', '.join(categories)}. Return both the description and the selected categories. The categories should be pretty strict, so only choose one if you're really sure it's the best choice. Also, suggest up to 5 relevant tags.",
            },
        ],
        max_tokens=150,
        response_model=DescriptionAndCategories,
    )
    return response


def get_all_categories(root_dir: str) -> set[str]:
    """
    Read all markdown files and extract unique categories.

    Args:
        root_dir (str): The root directory to start processing from.

    Returns:
        Set[str]: A set of unique categories.
    """
    categories = set()
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                post = frontmatter.load(file_path)
                if "categories" in post.metadata:
                    categories.update(post.metadata["categories"])
    return categories


def preview_categories(root_dir: str) -> None:
    """
    Preview all categories found in markdown files.

    Args:
        root_dir (str): The root directory to start processing from.
    """
    categories = get_all_categories(root_dir)

    table = Table(title="Categories Preview")
    table.add_column("Category", style="cyan")

    for category in sorted(categories):
        table.add_row(category)

    console.print(table)
    console.print(f"\nTotal categories found: {len(categories)}")


async def process_file(
    client: AsyncOpenAI, file_path: str, categories: list[str], enable_comments: bool
) -> None:
    """
    Process a single file, adding or updating the description and categories in the front matter.

    Args:
        client (AsyncOpenAI): The AsyncOpenAI client.
        file_path (str): The path to the file to process.
        categories (List[str]): List of all available categories.
        enable_comments (bool): Whether to enable comments in the front matter.
    """
    post = frontmatter.load(file_path)
    title = post.metadata.get("title", os.path.basename(file_path))

    response = await generate_ai_frontmatter(client, title, post.content, categories)
    post.metadata["description"] = response.description
    post.metadata["categories"] = response.categories
    post.metadata["tags"] = response.tags

    if enable_comments:
        post.metadata["comments"] = True

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(frontmatter.dumps(post))

    console.print(f"[green]Updated front matter in {file_path}[/green]")


async def process_files(
    root_dir: str,
    api_key: Optional[str] = None,  # noqa: ARG001
    use_categories: bool = False,
    enable_comments: bool = False,
) -> None:
    """
    Process all markdown files in the given directory and its subdirectories.

    Args:
        root_dir (str): The root directory to start processing from.
        api_key (Optional[str]): The OpenAI API key. If not provided, it will be read from the OPENAI_API_KEY environment variable.
        use_categories (bool): Whether to first read all files and generate a list of categories.
        enable_comments (bool): Whether to enable comments in the front matter.
    """
    markdown_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".md"):
                markdown_files.append(os.path.join(root, file))

    categories = list(get_all_categories(root_dir)) if use_categories else []

    with Progress() as progress:
        task = progress.add_task(
            "[green]Processing files...", total=len(markdown_files)
        )

        async def process_and_update(file_path: str) -> None:
            await process_file(client, file_path, categories, enable_comments)
            progress.update(task, advance=1)

        tasks = [process_and_update(file_path) for file_path in markdown_files]
        await asyncio.gather(*tasks)

    console.print("[bold green]All files processed successfully![/bold green]")


app = typer.Typer()


@app.command()
def main(
    root_dir: str = typer.Option("docs", help="Root directory to process"),
    api_key: Optional[str] = typer.Option(None, help="OpenAI API key"),
    use_categories: bool = typer.Option(False, help="Use categories from all files"),
    preview_only: bool = typer.Option(
        False, help="Preview categories without processing files"
    ),
    enable_comments: bool = typer.Option(
        False, help="Enable comments in the front matter"
    ),
):
    """
    Add or update description in front matter of markdown files in the given directory and its subdirectories.
    """
    if preview_only:
        preview_categories(root_dir)
    else:
        asyncio.run(process_files(root_dir, api_key, use_categories, enable_comments))


if __name__ == "__main__":
    app()
