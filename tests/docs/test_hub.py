import pytest
from pytest_examples import CodeExample, EvalExample


@pytest.mark.skip(reason="Hub functionality is being removed")
def test_format_blog(example: CodeExample, eval_example: EvalExample) -> None:
    """This test is being skipped as the hub functionality is being removed."""
    excluded_sources: list[str] = [
        "mistral",
        "ollama",
        "llama_cpp",
        "groq",
        "youtube",
        "contact",
        "langsmith",
    ]  # sources that are not supported in testing
    if any(source in example.source for source in excluded_sources):
        return

    if eval_example.update_examples:
        eval_example.format(example)
        eval_example.run_print_update(example)
    else:
        eval_example.lint(example)
        eval_example.run(example)
