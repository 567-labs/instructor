from __future__ import annotations

import glob
import os
from collections.abc import Iterable

from pytest_examples import find_examples

EXCLUDED = {
    "ollama.md",
    "watsonx.md",
    "local_classification.md",
}

BATCH = {
    "batch_classification_langsmith.md",
    "batch_in_memory.md",
    "batch_job_oai.md",
}

MULTIMODAL = {
    "audio_extraction.md",
    "extract_slides.md",
    "extracting_receipts.md",
    "image_to_ad_copy.md",
    "multi_modal_gemini.md",
    "tables_from_vision.md",
    "youtube_clips.md",
}

PROVIDERS = {
    "groq.md",
    "mistral.md",
    "open_source.md",
}

INTEGRATIONS = {
    "search.md",
    "tracing_with_langfuse.md",
}


def example_paths(names: Iterable[str]) -> list[str]:
    return [os.path.join("docs", "examples", name) for name in names]


def all_example_files() -> list[str]:
    return sorted(glob.glob("docs/examples/*.md"))


def core_example_files() -> list[str]:
    excluded = EXCLUDED | BATCH | MULTIMODAL | PROVIDERS | INTEGRATIONS
    return [
        path for path in all_example_files() if os.path.basename(path) not in excluded
    ]


def collect_examples(files: Iterable[str]):
    examples = []
    for markdown_file in files:
        examples.extend(find_examples(markdown_file))
    return examples
