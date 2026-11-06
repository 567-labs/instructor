import pytest
from pytest_examples import find_examples, CodeExample, EvalExample


@pytest.mark.parametrize("example", find_examples("docs/prompting"), ids=str)
@pytest.mark.skip(reason="Skipping this for now")
def test_format_concepts(example: CodeExample, eval_example: EvalExample):
    if eval_example.update_examples:
        eval_example.format(example)
        # eval_example.run_print_update(example)
    else:
        eval_example.lint(example)
        # eval_example.run(example)
