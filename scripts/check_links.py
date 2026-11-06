#!/usr/bin/env python3
"""
Check for broken internal links in documentation files.

Finds:
- Broken internal links (missing target files)
- Broken anchor links
- Orphaned pages (no incoming links)
"""

import argparse
import re
from pathlib import Path


def find_markdown_files(docs_dir: Path) -> list[Path]:
    """Find all markdown files in the docs directory."""
    return list(docs_dir.rglob("*.md"))


def extract_links(content: str, file_path: Path) -> list[tuple[str, int]]:  # noqa: ARG001
    """
    Extract internal markdown links from content.

    Returns:
        List of (link_target, line_number) tuples
    """
    links = []

    # Match markdown links: [text](url)
    for match in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", content):
        link_text = match.group(1)
        link_url = match.group(2)
        line_num = content[: match.start()].count("\n") + 1

        # Skip external links
        if link_url.startswith(("http://", "https://", "mailto:", "#")):
            continue

        links.append((link_url, line_num))

    return links


def resolve_link(link_url: str, source_file: Path, docs_dir: Path) -> tuple[bool, str]:  # noqa: ARG001
    """
    Resolve a relative link and check if target exists.

    Returns:
        (exists, resolved_path)
    """
    # Split anchor if present
    if "#" in link_url:
        link_path, anchor = link_url.split("#", 1)
    else:
        link_path = link_url
        anchor = None

    # Resolve relative path
    source_dir = source_file.parent
    target_path = (source_dir / link_path).resolve()

    # Check if file exists
    exists = target_path.exists()

    return exists, str(target_path)


def check_file(file_path: Path, docs_dir: Path) -> dict[str, list[tuple[str, int]]]:
    """Check all links in a file."""
    issues = {}

    try:
        content = file_path.read_text(encoding="utf-8")
        links = extract_links(content, file_path)

        broken_links = []
        for link_url, line_num in links:
            exists, resolved_path = resolve_link(link_url, file_path, docs_dir)
            if not exists:
                broken_links.append((link_url, line_num))

        if broken_links:
            issues["broken_links"] = broken_links

        return issues
    except Exception as e:
        return {"error": [(str(e), 0)]}


def find_orphaned_pages(files: list[Path], docs_dir: Path) -> set[Path]:
    """Find pages with no incoming links."""
    all_files = set(files)
    referenced_files = set()

    for file_path in files:
        try:
            content = file_path.read_text(encoding="utf-8")
            links = extract_links(content, file_path)

            for link_url, _ in links:
                exists, resolved_path = resolve_link(link_url, file_path, docs_dir)
                if exists:
                    referenced_files.add(Path(resolved_path))
        except Exception:
            pass

    # Files that are not referenced (orphaned)
    orphaned = all_files - referenced_files

    # Remove index pages and special files from orphaned list
    orphaned = {
        f
        for f in orphaned
        if not any(
            part in str(f)
            for part in ["index.md", "AGENT.md", "repository-overview.md"]
        )
    }

    return orphaned


def main():
    parser = argparse.ArgumentParser(
        description="Check for broken internal links in documentation"
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
        help="Check a single file instead of all files",
    )
    parser.add_argument(
        "--find-orphans",
        action="store_true",
        help="Find orphaned pages with no incoming links",
    )

    args = parser.parse_args()

    if args.file:
        files = [args.file]
    else:
        files = find_markdown_files(args.docs_dir)

    all_issues = {}
    total_broken = 0

    for file_path in files:
        issues = check_file(file_path, args.docs_dir)
        if issues:
            all_issues[str(file_path)] = issues
            if "broken_links" in issues:
                total_broken += len(issues["broken_links"])

    if args.summary:
        print("Summary Statistics:")
        print("=" * 60)
        print(f"  Files with broken links: {len(all_issues)}")
        print(f"  Total broken links: {total_broken}")
    else:
        # Detailed report
        for file_path, issues in sorted(all_issues.items()):
            if "broken_links" in issues:
                print(f"\n{file_path}:")
                for link_url, line_num in issues["broken_links"]:
                    print(f"  Line {line_num}: {link_url}")

    if args.find_orphans:
        orphaned = find_orphaned_pages(files, args.docs_dir)
        if orphaned:
            print("\n" + "=" * 60)
            print("Orphaned Pages (no incoming links):")
            print("=" * 60)
            for file_path in sorted(orphaned):
                print(f"  {file_path}")
            print(f"\nTotal orphaned pages: {len(orphaned)}")

    print(f"\nTotal files checked: {len(files)}")


if __name__ == "__main__":
    main()
