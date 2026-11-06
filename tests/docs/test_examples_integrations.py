import pytest
from pytest_examples import CodeExample, EvalExample

from tests.docs._example_groups import INTEGRATIONS, collect_examples, example_paths

code_examples = collect_examples(example_paths(INTEGRATIONS))


@pytest.mark.parametrize("example", code_examples, ids=str)
def test_examples_integrations(example: CodeExample, eval_example: EvalExample):
    if eval_example.update_examples:
        eval_example.format(example)
        eval_example.run_print_update(example)
    else:
        eval_example.lint(example)
