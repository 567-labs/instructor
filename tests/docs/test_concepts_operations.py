import pytest
from pytest_examples import CodeExample, EvalExample

from tests.docs._concept_groups import OPERATIONS, collect_examples, concept_paths

code_examples = collect_examples(concept_paths(OPERATIONS))


@pytest.mark.parametrize("example", code_examples, ids=str)
def test_format_concepts_operations(example: CodeExample, eval_example: EvalExample):
    if eval_example.update_examples:
        eval_example.format(example)
        eval_example.run_print_update(example)
    else:
        eval_example.lint(example)
