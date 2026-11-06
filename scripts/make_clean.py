#!/usr/bin/env python3
"""
Clean markdown files in the docs directory.

This script:
- Recursively finds all .md files in the docs directory
- Strips special whitespace characters (non-breaking spaces, zero-width spaces, etc.)
- Replaces em dashes (—) with regular dashes (-)
- Preserves the original file structure
"""

import re
import unicodedata
from pathlib import Path


def clean_markdown_content(content: str) -> str:
    """
    Clean markdown content by removing special whitespace and replacing em dashes.

    Args:
        content: The original markdown content

    Returns:
        The cleaned markdown content
    """
    # Replace em dashes with regular dashes
    content = content.replace("—", "-")
    content = content.replace("–", "-")  # en dash as well

    # Remove special whitespace characters
    # This includes non-breaking spaces, zero-width spaces, and other Unicode whitespace
    cleaned_lines = []
    for line in content.split("\n"):
        # Normalize Unicode characters and remove special whitespace
        cleaned_line = unicodedata.normalize("NFKC", line)
        # Remove zero-width characters and other special whitespace
        cleaned_line = re.sub(r"[\u200B\u200C\u200D\uFEFF]", "", cleaned_line)
        # Replace non-breaking spaces with regular spaces
        cleaned_line = cleaned_line.replace("\u00a0", " ")
        # Strip leading/trailing whitespace but preserve intentional indentation
        cleaned_line = cleaned_line.rstrip()
        cleaned_lines.append(cleaned_line)

    return "\n".join(cleaned_lines)


def process_markdown_files(docs_dir: str = "docs", dry_run: bool = False) -> None:
    """
    Process all markdown files in the docs directory.

    Args:
        docs_dir: Path to the docs directory (default: "docs")
        dry_run: If True, show what would be changed without modifying files
    """
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        print(f"Error: Directory '{docs_dir}' does not exist.")
        return

    if not docs_path.is_dir():
        print(f"Error: '{docs_dir}' is not a directory.")
        return

    # Find all markdown files recursively
    md_files = list(docs_path.rglob("*.md"))

    if not md_files:
        print(f"No markdown files found in '{docs_dir}' directory.")
        return

    mode_text = "DRY RUN - " if dry_run else ""
    print(f"{mode_text}Found {len(md_files)} markdown files to process...")

    processed_count = 0
    modified_count = 0

    for md_file in md_files:
        try:
            # Read the original content
            with open(md_file, encoding="utf-8") as f:
                original_content = f.read()

            # Clean the content
            cleaned_content = clean_markdown_content(original_content)

            # Check if content was modified
            if cleaned_content != original_content:
                if dry_run:
                    print(f"Would modify: {md_file}")
                    # Show a sample of the changes
                    original_lines = original_content.split("\n")
                    cleaned_lines = cleaned_content.split("\n")
                    for i, (orig, clean) in enumerate(
                        zip(original_lines, cleaned_lines)
                    ):
                        if orig != clean:
                            print(f"  Line {i + 1}:")
                            print(f"    Original: {repr(orig)}")
                            print(f"    Cleaned:  {repr(clean)}")
                            # Only show first difference per file
                            break
                else:
                    # Write the cleaned content back to the file
                    with open(md_file, "w", encoding="utf-8") as f:
                        f.write(cleaned_content)
                    print(f"Modified: {md_file}")
                modified_count += 1
            else:
                if not dry_run:
                    print(f"No changes needed: {md_file}")

            processed_count += 1

        except Exception as e:
            print(f"Error processing {md_file}: {e}")

    action_text = "would be" if dry_run else "were"
    print(f"\nProcessing complete!")
    print(f"Total files processed: {processed_count}")
    print(f"Files {action_text} modified: {modified_count}")


def main():
    """Main function to handle command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean markdown files by removing special whitespace and replacing em dashes"
    )
    parser.add_argument(
        "--docs-dir", default="docs", help="Path to docs directory (default: docs)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )

    args = parser.parse_args()

    process_markdown_files(docs_dir=args.docs_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
