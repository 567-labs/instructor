#!/usr/bin/env python3
"""
Fix API calls in documentation files.

Replaces old API patterns with simplified versions:
- client.chat.completions.create → client.create
- client.chat.completions.create_partial → client.create_partial
- client.chat.completions.create_iterable → client.create_iterable
- client.chat.completions.create_with_completion → client.create_with_completion
"""

import argparse
import re
from pathlib import Path


def find_markdown_files(docs_dir: Path) -> list[Path]:
    """Find all markdown files in the docs directory."""
    return list(docs_dir.rglob("*.md")) + list(docs_dir.rglob("*.ipynb"))


def replace_api_calls(content: str, dry_run: bool = False) -> tuple[str, int]:  # noqa: ARG001
    """
    Replace old API call patterns with simplified versions.

    Returns:
        Tuple of (new_content, number_of_replacements)
    """
    replacements = 0

    # Pattern mappings: (old_pattern, new_pattern)
    patterns = [
        (
            r"client\.chat\.completions\.create_with_completion\(",
            "client.create_with_completion(",
        ),
        (r"client\.chat\.completions\.create_partial\(", "client.create_partial("),
        (r"client\.chat\.completions\.create_iterable\(", "client.create_iterable("),
        (r"client\.chat\.completions\.create\(", "client.create("),
    ]

    new_content = content
    for old_pattern, new_pattern in patterns:
        matches = len(re.findall(old_pattern, new_content))
        if matches > 0:
            new_content = re.sub(old_pattern, new_pattern, new_content)
            replacements += matches

    return new_content, replacements


def process_file(file_path: Path, dry_run: bool = False) -> int:
    """Process a single file and return number of replacements."""
    try:
        content = file_path.read_text(encoding="utf-8")
        new_content, replacements = replace_api_calls(content, dry_run)

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
        description="Replace old API call patterns with simplified versions"
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


if __name__ == "__main__":
    main()
