from __future__ import annotations

import glob
import os
from collections.abc import Iterable

from pytest_examples import find_examples

CORE = {
    "alias.md",
    "dictionary_operations.md",
    "distillation.md",
    "enums.md",
    "fastapi.md",
    "fields.md",
    "index.md",
    "iterable.md",
    "lists.md",
    "logging.md",
    "maybe.md",
    "models.md",
    "parallel.md",
    "partial.md",
    "philosophy.md",
    "prompting.md",
    "typeadapter.md",
    "typeddicts.md",
    "types.md",
    "union.md",
    "unions.md",
    "validation.md",
}

OPERATIONS = {
    "caching.md",
    "prompt_caching.md",
    "raw_response.md",
    "retrying.md",
    "error_handling.md",
}

PROVIDERS = {
    "from_provider.md",
    "migration.md",
    "mode-migration.md",
    "patching.md",
    "usage.md",
}

ADVANCED = {
    "batch.md",
    "hooks.md",
    "multimodal.md",
    "reask_validation.md",
    "semantic_validation.md",
    "templating.md",
}


def concept_paths(names: Iterable[str]) -> list[str]:
    return [os.path.join("docs", "concepts", name) for name in names]


def all_concept_files() -> list[str]:
    return sorted(glob.glob("docs/concepts/*.md"))


def core_concept_files() -> list[str]:
    excluded = OPERATIONS | PROVIDERS | ADVANCED
    return [
        path for path in all_concept_files() if os.path.basename(path) not in excluded
    ]


def collect_examples(files: Iterable[str]):
    examples = []
    for markdown_file in files:
        examples.extend(find_examples(markdown_file))
    return examples
