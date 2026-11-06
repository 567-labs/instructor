#!/usr/bin/env python3
"""
Fix old client initialization patterns in documentation files.

Replaces old initialization patterns with from_provider:
- instructor.from_openai(OpenAI()) → instructor.from_provider("openai/model-name")
- instructor.from_anthropic(Anthropic()) → instructor.from_provider("anthropic/model-name")
- instructor.patch(OpenAI()) → instructor.from_provider("openai/model-name")
- Similar patterns for all other providers
"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple


# Mapping of provider names to their from_provider identifiers
PROVIDER_MAPPING = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google",
    "cohere": "cohere",
    "mistral": "mistral",
    "groq": "groq",
    "litellm": "litellm",
    "ollama": "ollama",
    "azure": "azure",
    "bedrock": "bedrock",
    "vertex": "vertex",
    "genai": "google",  # Google GenAI
    "deepseek": "deepseek",
    "fireworks": "fireworks",
    "cerebras": "cerebras",
    "together": "together",
    "anyscale": "anyscale",
    "perplexity": "perplexity",
    "writer": "writer",
    "openrouter": "openrouter",
    "sambanova": "sambanova",
    "truefoundry": "truefoundry",
    "cortex": "cortex",
    "databricks": "databricks",
    "xai": "xai",
}


def find_markdown_files(docs_dir: Path) -> List[Path]:
    """Find all markdown files in the docs directory."""
    return list(docs_dir.rglob("*.md")) + list(docs_dir.rglob("*.ipynb"))


def extract_model_name(content: str, match_start: int, match_end: int) -> str:
    """
    Try to extract model name from context around the match.
    Looks for common patterns like model="...", model='...', or model_name=...
    """
    # Look backwards and forwards for model parameter
    context_start = max(0, match_start - 200)
    context_end = min(len(content), match_end + 200)
    context = content[context_start:context_end]

    # Try to find model parameter
    model_match = re.search(
        r'model\s*[=:]\s*["\']([^"\']+)["\']', context, re.IGNORECASE
    )
    if model_match:
        return model_match.group(1)

    # Default model names by provider
    return "gpt-4o"  # Will need manual review for accuracy


def replace_from_pattern(
    content: str, provider: str, dry_run: bool = False
) -> Tuple[str, int]:
    """
    Replace instructor.from_PROVIDER(Provider()) patterns.

    Pattern: instructor.from_openai(OpenAI(model="..."))
    → instructor.from_provider("openai/model-name")
    """
    replacements = 0

    # Pattern: instructor.from_PROVIDER(ProviderClass(...))
    pattern = rf"instructor\.from_{provider}\((\w+)(\([^)]*\))?\)"

    def replacer(match):
        nonlocal replacements
        provider_class = match.group(1)
        args = match.group(2) or ""

        # Try to extract model name from args
        model_match = re.search(r'model\s*=\s*["\']([^"\']+)["\']', args)
        if model_match:
            model_name = model_match.group(1)
        else:
            # Default model - may need manual review
            model_name = (
                "gpt-4o" if provider == "openai" else "claude-3-5-sonnet-20241022"
            )

        replacements += 1
        return f'instructor.from_provider("{provider}/{model_name}")'

    new_content = re.sub(pattern, replacer, content, flags=re.IGNORECASE)
    return new_content, replacements


def replace_patch_pattern(content: str, dry_run: bool = False) -> Tuple[str, int]:
    """
    Replace instructor.patch(Provider()) patterns.

    Pattern: instructor.patch(OpenAI(model="..."))
    → instructor.from_provider("openai/model-name")
    """
    replacements = 0

    # Pattern: instructor.patch(ProviderClass(...))
    # Match common provider classes
    provider_classes = "|".join(
        [
            "OpenAI",
            "Anthropic",
            "GoogleGenerativeAI",
            "Cohere",
            "Mistral",
            "Groq",
            "LiteLLM",
            "Ollama",
            "Bedrock",
            "VertexAI",
        ]
    )

    pattern = rf"instructor\.patch\(({provider_classes})(\([^)]*\))?\)"

    def replacer(match):
        nonlocal replacements
        provider_class = match.group(1)
        args = match.group(2) or ""

        # Map class name to provider identifier
        class_to_provider = {
            "OpenAI": "openai",
            "Anthropic": "anthropic",
            "GoogleGenerativeAI": "google",
            "Cohere": "cohere",
            "Mistral": "mistral",
            "Groq": "groq",
            "LiteLLM": "litellm",
            "Ollama": "ollama",
            "Bedrock": "bedrock",
            "VertexAI": "vertex",
        }

        provider = class_to_provider.get(provider_class, "openai")

        # Try to extract model name from args
        model_match = re.search(r'model\s*=\s*["\']([^"\']+)["\']', args)
        if model_match:
            model_name = model_match.group(1)
        else:
            # Default models
            defaults = {
                "openai": "gpt-4o",
                "anthropic": "claude-3-5-sonnet-20241022",
                "google": "gemini-1.5-pro",
            }
            model_name = defaults.get(provider, "gpt-4o")

        replacements += 1
        return f'instructor.from_provider("{provider}/{model_name}")'

    new_content = re.sub(pattern, replacer, content)
    return new_content, replacements


def replace_old_patterns(content: str, dry_run: bool = False) -> Tuple[str, int]:
    """
    Replace all old initialization patterns.

    Returns:
        Tuple of (new_content, total_replacements)
    """
    total_replacements = 0
    new_content = content

    # Replace instructor.patch() patterns first
    new_content, patch_replacements = replace_patch_pattern(new_content, dry_run)
    total_replacements += patch_replacements

    # Replace instructor.from_* patterns for each provider
    for provider in PROVIDER_MAPPING.keys():
        new_content, from_replacements = replace_from_pattern(
            new_content, provider, dry_run
        )
        total_replacements += from_replacements

    return new_content, total_replacements


def process_file(file_path: Path, dry_run: bool = False) -> int:
    """Process a single file and return number of replacements."""
    try:
        content = file_path.read_text(encoding="utf-8")
        new_content, replacements = replace_old_patterns(content, dry_run)

        if replacements > 0:
            if dry_run:
                print(f"Would fix {replacements} instances in {file_path}")
            else:
                file_path.write_text(new_content, encoding="utf-8")
                print(f"Fixed {replacements} instances in {file_path}")

        return replacements
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Replace old client initialization patterns with from_provider"
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("docs"),
        help="Directory containing documentation files (default: docs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without making changes",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Process a single file instead of all files",
    )

    args = parser.parse_args()

    if args.file:
        files = [args.file]
    else:
        files = find_markdown_files(args.docs_dir)

    total_replacements = 0
    files_modified = 0

    for file_path in files:
        replacements = process_file(file_path, args.dry_run)
        if replacements > 0:
            total_replacements += replacements
            files_modified += 1

    print(f"\nSummary:")
    print(f"  Files processed: {len(files)}")
    print(f"  Files modified: {files_modified}")
    print(f"  Total replacements: {total_replacements}")

    if args.dry_run:
        print("\nRun without --dry-run to apply changes")
    else:
        print("\n⚠️  Note: Please review model names - defaults may need adjustment")


if __name__ == "__main__":
    main()
