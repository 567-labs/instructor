#!/usr/bin/env python3
"""
Validate heading structure in documentation files.

Checks for:
- Multiple H1 tags (should only have one)
- Heading hierarchy violations (e.g., H1 → H3 skipping H2)
- Missing H1 tags
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path


def find_markdown_files(docs_dir: Path) -> list[Path]:
    """Find all markdown files in the docs directory."""
    return list(docs_dir.rglob("*.md"))


def extract_headings(content: str) -> list[tuple[int, str, int]]:
    """
    Extract all headings from markdown content.

    Returns:
        List of (level, text, line_number) tuples
    """
    headings = []
    lines = content.split("\n")

    for line_num, line in enumerate(lines, 1):
        # Match markdown headings: # Title, ## Title, etc.
        match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if match:
            level = len(match.group(1))
            text = match.group(2).strip()
            headings.append((level, text, line_num))

    return headings


def validate_headings(headings: list[tuple[int, str, int]]) -> dict[str, list[str]]:
    """Validate heading structure."""
    issues = {}

    if not headings:
        issues["no_headings"] = ["No headings found in file"]
        return issues

    # Check for H1
    h1_headings = [h for h in headings if h[0] == 1]
    if not h1_headings:
        issues["missing_h1"] = ["No H1 heading found"]
    elif len(h1_headings) > 1:
        issues["multiple_h1"] = [
            f"Line {line}: {text}" for level, text, line in h1_headings
        ]

    # Check heading hierarchy
    prev_level = 0
    hierarchy_violations = []
    for level, text, line_num in headings:
        if prev_level > 0 and level > prev_level + 1:
            hierarchy_violations.append(
                f"Line {line_num}: Skipped from H{prev_level} to H{level}: {text[:50]}"
            )
        prev_level = level

    if hierarchy_violations:
        issues["hierarchy_violations"] = hierarchy_violations

    return issues


def process_file(file_path: Path) -> dict[str, list[str]]:
    """Process a single file and return issues."""
    try:
        content = file_path.read_text(encoding="utf-8")
        headings = extract_headings(content)
        return validate_headings(headings)
    except Exception as e:
        return {"error": [str(e)]}


def main():
    parser = argparse.ArgumentParser(
        description="Validate heading structure in documentation files"
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("docs"),
        help="Directory containing documentation files (default: docs)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show only summary statistics",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Validate a single file instead of all files",
    )

    args = parser.parse_args()

    if args.file:
        files = [args.file]
    else:
        files = find_markdown_files(args.docs_dir)

    all_issues = {}
    total_counts = defaultdict(int)

    for file_path in files:
        issues = process_file(file_path)
        if issues:
            all_issues[str(file_path)] = issues
            for issue_type, messages in issues.items():
                total_counts[issue_type] += len(messages)

    if args.summary:
        print("Summary Statistics:")
        print("=" * 60)
        for issue_type, count in sorted(total_counts.items()):
            print(f"  {issue_type.replace('_', ' ').title()}: {count}")
    else:
        # Detailed report
        for file_path, issues in sorted(all_issues.items()):
            print(f"\n{file_path}:")
            for issue_type, messages in issues.items():
                print(f"  {issue_type.replace('_', ' ').title()}:")
                for message in messages:
                    print(f"    {message}")

    print(f"\nTotal files checked: {len(files)}")
    print(f"Files with issues: {len(all_issues)}")


if __name__ == "__main__":
    main()
