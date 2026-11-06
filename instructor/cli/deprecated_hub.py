from typer import Exit, echo, Typer

app: Typer = Typer(help="Instructor Hub CLI (Deprecated)")


@app.command(name="hub")
def hub() -> None:
    """
    This command has been deprecated. The instructor hub is no longer available.
    Please refer to our cookbook examples at https://python.useinstructor.com/examples/
    """
    echo(
        "The instructor hub has been deprecated. Please refer to our cookbook examples at https://python.useinstructor.com/examples/"
    )
    raise Exit(1)


if __name__ == "__main__":
    app()
