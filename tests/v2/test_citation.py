"""Tests for CitationMixin span resolution.

Covers the regex-escaping behavior in ``_get_span``: LLM-generated quotes may
contain regex metacharacters, which must be matched literally rather than
compiled as a pattern.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from instructor.v2.dsl.citation import CitationMixin


class Answer(CitationMixin):
    pass


def test_quote_with_regex_metacharacters_does_not_crash() -> None:
    """A quote with unbalanced parentheses must resolve, not raise regex.error."""
    context = "The margin is 50% (approx) this quarter."

    answer = Answer.model_validate(
        {"substring_quotes": ["50% (approx"]},
        context={"context": context},
    )

    # The span is found and normalized back to the exact context substring.
    assert answer.substring_quotes == ["50% (approx"]


@pytest.mark.parametrize(
    "quote",
    [
        "cost [USD]",  # brackets
        "a+b*c",  # quantifiers
        "path\\to\\file",  # backslashes
        "who? (maybe)",  # optional + parens
    ],
)
def test_various_metacharacter_quotes_resolve(quote: str) -> None:
    context = f"prefix {quote} suffix"

    answer = Answer.model_validate(
        {"substring_quotes": [quote]},
        context={"context": context},
    )

    assert answer.substring_quotes == [quote]


def test_non_matching_quote_is_dropped() -> None:
    """Quotes absent from the context are removed rather than kept."""
    context = "Nothing relevant here."

    answer = Answer.model_validate(
        {"substring_quotes": ["(totally unrelated)"]},
        context={"context": context},
    )

    assert answer.substring_quotes == []


def test_fuzzy_matching_still_works_after_escaping() -> None:
    """Escaping the literal quote must not disable fuzzy (edit-distance) matching."""
    context = "The margin is 50% (approx) this quarter."

    # One-character typo ("aprox") within the default edit budget.
    answer = Answer.model_validate(
        {"substring_quotes": ["50% (aprox)"]},
        context={"context": context},
    )

    assert answer.substring_quotes == ["50% (approx)"]


def test_no_context_leaves_quotes_untouched() -> None:
    answer = Answer.model_validate({"substring_quotes": ["anything (raw"]})

    assert answer.substring_quotes == ["anything (raw"]


@pytest.mark.parametrize("container", [list, tuple])
def test_list_context_chunks_resolve_without_typeerror(container: type) -> None:
    """List/tuple RAG contexts must resolve quotes, not raise TypeError."""
    chunks = [
        "Betty was a student.",
        "Jason was a student. Jason is 20 years old.",
    ]

    answer = Answer.model_validate(
        {"substring_quotes": ["Jason is 20 years old", "missing quote"]},
        context={"context": container(chunks)},
    )

    assert answer.substring_quotes == ["Jason is 20 years old"]


@pytest.mark.parametrize("context", [{"nested": "dict"}, 42, ["source", 42]])
def test_non_text_context_fails_validation(context: object) -> None:
    with pytest.raises(ValidationError, match="Citation context must be text"):
        Answer.model_validate(
            {"substring_quotes": ["unverified quote"]}, context={"context": context}
        )


def test_quotes_cannot_span_independent_chunks() -> None:
    answer = Answer.model_validate(
        {"substring_quotes": ["abcdefghijklmnop\nqrstuvwxyzabcdef"]},
        context={"context": ["abcdefghijklmnop", "qrstuvwxyzabcdef"]},
    )
    assert answer.substring_quotes == []


def test_empty_chunks_drop_unverified_quotes() -> None:
    answer = Answer.model_validate(
        {"substring_quotes": ["unverified quote"]}, context={"context": []}
    )
    assert answer.substring_quotes == []


def test_quote_within_error_tolerance_matches() -> None:
    context = "0123456789"

    answer = Answer.model_validate(
        {"substring_quotes": ["0123456ZZZ"]},
        context={"context": context},
    )

    assert answer.substring_quotes == [context]


def test_quote_beyond_error_tolerance_is_dropped() -> None:
    context = "0123456789"

    answer = Answer.model_validate(
        {"substring_quotes": ["0123ZZZZZZ"]},
        context={"context": context},
    )

    assert answer.substring_quotes == []
