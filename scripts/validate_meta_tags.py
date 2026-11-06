#!/usr/bin/env python3
"""
Validate frontmatter meta tags in documentation files.

Checks for:
- Missing title/description
- Title length (50-60 chars recommended)
- Description length (150-160 chars recommended)
- Duplicate titles/descriptions
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def find_markdown_files(docs_dir: Path) -> List[Path]:
    """Find all markdown files in the docs directory."""
    return list(docs_dir.rglob("*.md"))


def extract_frontmatter(content: str) -> Dict[str, str]:
    """Extract frontmatter from markdown content."""
    frontmatter = {}

    # Match YAML frontmatter between --- markers
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not match:
        return frontmatter

    yaml_content = match.group(1)

    # Extract title
    title_match = re.search(r"^title:\s*(.+)$", yaml_content, re.MULTILINE)
    if title_match:
        frontmatter["title"] = title_match.group(1).strip(" \"'")

    # Extract description
    desc_match = re.search(r"^description:\s*(.+)$", yaml_content, re.MULTILINE)
    if desc_match:
        frontmatter["description"] = desc_match.group(1).strip(" \"'")

    # Extract keywords
    keywords_match = re.search(r"^keywords:\s*(.+)$", yaml_content, re.MULTILINE)
    if keywords_match:
        frontmatter["keywords"] = keywords_match.group(1).strip(" \"'")

    return frontmatter


def validate_file(file_path: Path) -> Dict[str, List[str]]:
    """Validate a single file's frontmatter."""
    issues = {}

    try:
        content = file_path.read_text(encoding="utf-8")
        frontmatter = extract_frontmatter(content)

        # Check for missing frontmatter
        if not frontmatter:
            issues["missing_frontmatter"] = ["No frontmatter found"]
            return issues

        # Check title
        if "title" not in frontmatter:
            issues["missing_title"] = ["Title missing from frontmatter"]
        else:
            title = frontmatter["title"]
            title_len = len(title)
            if title_len < 50:
                issues["title_too_short"] = [
                    f"Title is {title_len} chars (recommend 50-60 for SEO)"
                ]
            elif title_len > 60:
                issues["title_too_long"] = [
                    f"Title is {title_len} chars (recommend 50-60 for SEO)"
                ]

        # Check description
        if "description" not in frontmatter:
            issues["missing_description"] = ["Description missing from frontmatter"]
        else:
            desc = frontmatter["description"]
            desc_len = len(desc)
            if desc_len < 150:
                issues["description_too_short"] = [
                    f"Description is {desc_len} chars (recommend 150-160 for SEO)"
                ]
            elif desc_len > 160:
                issues["description_too_long"] = [
                    f"Description is {desc_len} chars (recommend 150-160 for SEO)"
                ]

        return issues
    except Exception as e:
        return {"error": [str(e)]}


def main():
    parser = argparse.ArgumentParser(
        description="Validate frontmatter meta tags in documentation files"
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
    parser.add_argument(
        "--check-duplicates",
        action="store_true",
        help="Check for duplicate titles and descriptions",
    )

    args = parser.parse_args()

    if args.file:
        files = [args.file]
    else:
        files = find_markdown_files(args.docs_dir)

    all_issues = {}
    total_counts = defaultdict(int)

    # Track titles and descriptions for duplicate checking
    titles = defaultdict(list)
    descriptions = defaultdict(list)

    for file_path in files:
        issues = validate_file(file_path)
        if issues:
            all_issues[str(file_path)] = issues
            for issue_type, messages in issues.items():
                total_counts[issue_type] += len(messages)

        # Collect titles and descriptions for duplicate checking
        if args.check_duplicates:
            content = file_path.read_text(encoding="utf-8")
            frontmatter = extract_frontmatter(content)
            if "title" in frontmatter:
                titles[frontmatter["title"]].append(str(file_path))
            if "description" in frontmatter:
                descriptions[frontmatter["description"]].append(str(file_path))

    if args.summary:
        print("Summary Statistics:")
        print("=" * 60)
        for issue_type, count in sorted(total_counts.items()):
            print(f"  {issue_type.replace('_', ' ').title()}: {count} files")
    else:
        # Detailed report
        for file_path, issues in sorted(all_issues.items()):
            print(f"\n{file_path}:")
            for issue_type, messages in issues.items():
                for message in messages:
                    print(f"  - {message}")

    # Check for duplicates
    if args.check_duplicates:
        print("\n" + "=" * 60)
        print("Duplicate Titles:")
        print("=" * 60)
        for title, file_list in sorted(titles.items()):
            if len(file_list) > 1:
                print(f"\n{title}")
                for f in file_list:
                    print(f"  - {f}")

        print("\n" + "=" * 60)
        print("Duplicate Descriptions:")
        print("=" * 60)
        for desc, file_list in sorted(descriptions.items()):
            if len(file_list) > 1:
                print(f"\n{desc}")
                for f in file_list:
                    print(f"  - {f}")

    print(f"\nTotal files checked: {len(files)}")
    print(f"Files with issues: {len(all_issues)}")


if __name__ == "__main__":
    main()
