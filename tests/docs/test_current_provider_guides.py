"""Lint runnable examples in the maintained provider guides."""

import pytest
from pytest_examples import CodeExample, EvalExample, find_examples


@pytest.mark.parametrize(
    "example",
    find_examples(
        *(
            f"docs/integrations/{name}.md"
            for name in (
                "openai",
                "anthropic",
                "google",
                "cohere",
                "groq",
                "mistral",
                "fireworks",
                "cerebras",
                "writer",
                "xai",
                "perplexity",
                "deepseek",
                "openrouter",
                "together",
                "bedrock",
                "vertex",
            )
        )
    ),
    ids=str,
)
def test_current_provider_guide(
    example: CodeExample, eval_example: EvalExample
) -> None:
    if eval_example.update_examples:
        eval_example.format(example)
    else:
        eval_example.lint(example)
