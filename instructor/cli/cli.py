from typing import Optional, Callable, Any, Protocol, TypeVar

T = TypeVar('T')

class TyperLike(Protocol):
    def command(self, *args: Any, **kwargs: Any) -> Callable[[T], T]:
        ...
    
    def add_typer(self, typer_instance: Any, name: str, help: str) -> None:
        ...
    
    def __call__(self) -> Any:
        ...

app = None

try:
    import typer
    from typer import launch
    import instructor.cli.jobs as jobs
    import instructor.cli.files as files
    import instructor.cli.usage as usage
    import instructor.cli.deprecated_hub as hub
    import instructor.cli.batch as batch

    app = typer.Typer()
    
    app.add_typer(jobs.app, name="jobs", help="Monitor and create fine tuning jobs")
    app.add_typer(files.app, name="files", help="Manage files on OpenAI's servers")
    app.add_typer(usage.app, name="usage", help="Check OpenAI API usage data")
    app.add_typer(
        hub.app, name="hub", help="[DEPRECATED] The instructor hub is no longer available"
    )
    app.add_typer(batch.app, name="batch", help="Manage OpenAI Batch jobs")
except ImportError:
    class DummyTyper:
        def command(self, *_args: Any, **_kwargs: Any) -> Callable[[T], T]:
            def decorator(func: T) -> T:
                return func
            return decorator
        
        def add_typer(self, typer_instance: Any, name: str, help: str) -> None:
            pass
            
        def __call__(self) -> None:
            pass
    
    app = DummyTyper()

@app.command()
def docs(
    query: Optional[str] = None,
) -> None:
    """
    Open the instructor documentation website.
    """
    try:
        if query:
            launch(f"https://python.useinstructor.com/?q={query}")
        else:
            launch("https://python.useinstructor.com/")
    except NameError:
        pass

if __name__ == "__main__":
    if isinstance(app, DummyTyper):
        print("CLI dependencies not installed. Please install instructor with CLI support: pip install instructor[cli]")
    else:
        app()
