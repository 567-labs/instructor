from typing import Optional
import typer
from typer import Typer, launch
import instructor.cli.jobs as jobs
import instructor.cli.files as files
import instructor.cli.usage as usage
import instructor.cli.deprecated_hub as hub
import instructor.cli.batch as batch

app: Typer = typer.Typer()

app.add_typer(jobs.app, name="jobs", help="Monitor and create fine tuning jobs")
app.add_typer(files.app, name="files", help="Manage files on OpenAI's servers")
app.add_typer(usage.app, name="usage", help="Check OpenAI API usage data")
app.add_typer(
    hub.app, name="hub", help="[DEPRECATED] The instructor hub is no longer available"
)
app.add_typer(batch.app, name="batch", help="Manage OpenAI Batch jobs")


@app.command()
def docs(
    query: Optional[str] = typer.Argument(None, help="Search the documentation"),
) -> None:
    """
    Open the instructor documentation website.
    """
    if query:
        launch(f"https://python.useinstructor.com/?q={query}")
    else:
        launch("https://python.useinstructor.com/")


if __name__ == "__main__":
    app()
