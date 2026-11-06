import pytest
from pytest_examples import find_examples, CodeExample, EvalExample


@pytest.mark.parametrize("example", find_examples("README.md"), ids=str)
def test_readme(example: CodeExample, eval_example: EvalExample):
    if eval_example.update_examples:
        eval_example.format(example)
    else:
        eval_example.lint(example)


@pytest.mark.parametrize("example", find_examples("docs/index.md"), ids=str)
def test_index(example: CodeExample, eval_example: EvalExample):
    if eval_example.update_examples:
        eval_example.format(example)
    else:
        eval_example.lint(example)
