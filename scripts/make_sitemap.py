import os
import asyncio
import yaml
from typing import Optional, Any
from collections.abc import Generator
from openai import AsyncOpenAI
import typer
from rich.console import Console
from rich.progress import Progress
import hashlib
from asyncio import as_completed
import tenacity
import re

console = Console()


def traverse_docs(
    root_dir: str = "docs",
) -> Generator[tuple[str, str, str], None, None]:
    """
    Recursively traverse the docs folder and yield the path, content, and content hash of each file.

    Args:
        root_dir (str): The root directory to start traversing from. Defaults to 'docs'.

    Yields:
        Tuple[str, str, str]: A tuple containing the relative path from 'docs', the file content, and the content hash.
    """
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".md"):  # Assuming we're only interested in Markdown files
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, root_dir)

                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                content_hash = hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()
                yield relative_path, content, content_hash


def extract_markdown_links(content: str) -> list[str]:
    """
    Extract all markdown links from the content.

    Args:
        content (str): The markdown content to analyze

    Returns:
        List[str]: List of extracted link paths
    """
    # Match markdown links [text](path)
    link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"
    matches = re.findall(link_pattern, content)

    links = []
    for _, link_path in matches:
        # Filter out external links and anchors
        if not link_path.startswith(("http://", "https://", "#", "mailto:")):
            # Clean up relative paths
            link_path = link_path.strip("/")
            if link_path.endswith(".md"):
                links.append(link_path)
            elif "." not in link_path:
                # Assume it's a directory reference, add index.md
                links.append(f"{link_path}/index.md")

    return links


def normalize_path(path: str, current_path: str) -> str:
    """
    Normalize a relative path based on the current file's location.

    Args:
        path (str): The path to normalize
        current_path (str): The current file's path

    Returns:
        str: The normalized path
    """
    if path.startswith("/"):
        # Absolute path from docs root
        return path.strip("/")

    # Relative path
    current_dir = os.path.dirname(current_path)
    if current_dir:
        normalized = os.path.normpath(os.path.join(current_dir, path))
        # Remove any leading '../' that go outside docs/
        while normalized.startswith("../"):
            normalized = normalized[3:]
        return normalized

    return path


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    retry=tenacity.retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: console.print(
        f"[yellow]Retrying analysis... (Attempt {retry_state.attempt_number})[/yellow]"
    ),
)
async def analyze_content(
    client: AsyncOpenAI, path: str, content: str
) -> dict[str, Any]:
    """
    Analyze the content of a file to extract summary, keywords, topics, and references.

    Args:
        client (AsyncOpenAI): The AsyncOpenAI client.
        path (str): The path of the file.
        content (str): The content of the file.

    Returns:
        Dict[str, Any]: Analysis results including summary, keywords, topics, and references.

    Raises:
        Exception: If all retry attempts fail.
    """
    try:
        response = await client.chat.completions.create(
            model="gpt-6-luna",
            reasoning_effort="none",
            messages=[
                {
                    "role": "system",
                    "content": """You are a documentation analyzer. Extract and return the following information in a structured format:
1. A concise summary (2-3 sentences) for SEO
2. A list of important keywords (5-10 words/phrases)
3. Main topics/concepts covered (3-5 topics)
4. Any references to other documentation pages mentioned in the text

Return the response in this exact format:
SUMMARY: [Your summary here]
KEYWORDS: [keyword1, keyword2, keyword3, ...]
TOPICS: [topic1, topic2, topic3, ...]
REFERENCES: [referenced_page1.md, referenced_page2.md, ...]

If no references are found, write: REFERENCES: none""",
                },
                {"role": "user", "content": content},
            ],
            max_completion_tokens=4000,
        )

        result_text = response.choices[0].message.content

        # Parse the structured response
        summary = ""
        keywords = []
        topics = []
        references = []

        if result_text:
            for line in result_text.split("\n"):
                line = line.strip()
                if line.startswith("SUMMARY:"):
                    summary = line[8:].strip()
                elif line.startswith("KEYWORDS:"):
                    keywords_text = line[9:].strip()
                    if keywords_text and keywords_text != "none":
                        keywords = [k.strip() for k in keywords_text.split(",")]
                elif line.startswith("TOPICS:"):
                    topics_text = line[7:].strip()
                    if topics_text and topics_text != "none":
                        topics = [t.strip() for t in topics_text.split(",")]
                elif line.startswith("REFERENCES:"):
                    refs_text = line[11:].strip()
                    if refs_text and refs_text != "none":
                        references = [r.strip() for r in refs_text.split(",")]

        return {
            "summary": summary,
            "keywords": keywords,
            "topics": topics,
            "ai_references": references,
        }

    except Exception as e:
        console.print(f"[bold red]Error analyzing {path}: {str(e)}[/bold red]")
        raise


async def generate_sitemap(
    root_dir: str,
    output_file: str,
    api_key: Optional[str] = None,
    max_concurrency: int = 5,
) -> None:
    """
    Generate a sitemap from the given root directory.

    Args:
        root_dir (str): The root directory to start traversing from.
        output_file (str): The output file to save the sitemap.
        api_key (Optional[str]): The OpenAI API key. If not provided, it will be read from the OPENAI_API_KEY environment variable.
        max_concurrency (int): The maximum number of concurrent tasks. Defaults to 5.
    """
    client = AsyncOpenAI(api_key=api_key)

    # Load existing sitemap if it exists
    existing_sitemap: dict[str, dict[str, Any]] = {}
    if os.path.exists(output_file):
        with open(output_file, encoding="utf-8") as sitemap_file:
            existing_sitemap = yaml.safe_load(sitemap_file) or {}

    sitemap_data: dict[str, dict[str, Any]] = {}

    async def process_file(
        path: str, content: str, content_hash: str
    ) -> tuple[str, dict[str, Any]]:
        # Check if we can reuse existing data
        if (
            path in existing_sitemap
            and existing_sitemap[path].get("hash") == content_hash
        ):
            # Extract markdown links even for cached content
            links = extract_markdown_links(content)
            normalized_links = []
            for link in links:
                normalized = normalize_path(link, path)
                if normalized:
                    normalized_links.append(normalized)

            existing_data = existing_sitemap[path].copy()
            existing_data["references"] = normalized_links
            return path, existing_data

        try:
            # Extract markdown links
            links = extract_markdown_links(content)
            normalized_links = []
            for link in links:
                normalized = normalize_path(link, path)
                if normalized:
                    normalized_links.append(normalized)

            # Get AI analysis
            analysis = await analyze_content(client, path, content)

            return path, {
                "summary": analysis["summary"],
                "keywords": analysis["keywords"],
                "topics": analysis["topics"],
                "references": normalized_links,
                "ai_references": analysis["ai_references"],
                "hash": content_hash,
            }
        except Exception as e:
            console.print(
                f"[bold red]Failed to analyze {path} after multiple attempts: {str(e)}[/bold red]"
            )
            return path, {
                "summary": "Failed to generate summary",
                "keywords": [],
                "topics": [],
                "references": normalized_links,
                "ai_references": [],
                "hash": content_hash,
            }

    files_to_process: list[tuple[str, str, str]] = list(traverse_docs(root_dir))
    total_files = len(files_to_process)

    with Progress() as progress:
        task = progress.add_task("[green]Processing files...", total=total_files)

        semaphore = asyncio.Semaphore(max_concurrency)

        async def bounded_process_file(*args):
            async with semaphore:
                return await process_file(*args)

        tasks = [
            bounded_process_file(path, content, content_hash)
            for path, content, content_hash in files_to_process
        ]

        for completed_task in as_completed(tasks):
            path, result = await completed_task
            sitemap_data[path] = result
            progress.update(task, advance=1)

    # Save final results
    with open(output_file, "w", encoding="utf-8") as sitemap_file:
        yaml.dump(sitemap_data, sitemap_file, default_flow_style=False, sort_keys=True)

    console.print(
        f"[bold green]Sitemap has been generated and saved to {output_file}[/bold green]"
    )
    console.print(f"[green]Processed {total_files} files[/green]")


app = typer.Typer()


@app.command()
def main(
    root_dir: str = typer.Option("docs", help="Root directory to traverse"),
    output_file: str = typer.Option("sitemap.yaml", help="Output file for the sitemap"),
    api_key: Optional[str] = typer.Option(None, help="OpenAI API key"),
    max_concurrency: int = typer.Option(5, help="Maximum number of concurrent tasks"),
):
    """
    Generate a sitemap with keywords, topics, and reference analysis.
    """
    asyncio.run(generate_sitemap(root_dir, output_file, api_key, max_concurrency))


if __name__ == "__main__":
    app()
