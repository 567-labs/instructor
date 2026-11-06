#!/usr/bin/env python3
"""
Audit documentation files for old patterns that need to be updated.

Reports:
- Old API call patterns (client.chat.completions.*)
- Old initialization patterns (instructor.from_*, instructor.patch)
- Unused imports
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def find_markdown_files(docs_dir: Path) -> List[Path]:
    """Find all markdown files in the docs directory."""
    return list(docs_dir.rglob("*.md")) + list(docs_dir.rglob("*.ipynb"))


def audit_api_calls(content: str, file_path: Path) -> Dict[str, List[int]]:
    """Find old API call patterns."""
    issues = defaultdict(list)

    patterns = {
        "client.chat.completions.create": r"client\.chat\.completions\.create\(",
        "client.chat.completions.create_partial": r"client\.chat\.completions\.create_partial\(",
        "client.chat.completions.create_iterable": r"client\.chat\.completions\.create_iterable\(",
        "client.chat.completions.create_with_completion": r"client\.chat\.completions\.create_with_completion\(",
    }

    for name, pattern in patterns.items():
        for match in re.finditer(pattern, content):
            line_num = content[: match.start()].count("\n") + 1
            issues[name].append(line_num)

    return issues


def audit_old_init_patterns(content: str, file_path: Path) -> Dict[str, List[int]]:
    """Find old initialization patterns."""
    issues = defaultdict(list)

    # Find instructor.from_* patterns
    from_pattern = r"instructor\.from_(\w+)\("
    for match in re.finditer(from_pattern, content):
        provider = match.group(1)
        line_num = content[: match.start()].count("\n") + 1
        issues[f"instructor.from_{provider}"].append(line_num)

    # Find instructor.patch patterns
    patch_pattern = r"instructor\.patch\("
    for match in re.finditer(patch_pattern, content):
        line_num = content[: match.start()].count("\n") + 1
        issues["instructor.patch"].append(line_num)

    return issues


def audit_unused_imports(content: str, file_path: Path) -> Dict[str, List[int]]:
    """Find potentially unused imports when from_provider is used."""
    issues = defaultdict(list)

    # Check if from_provider is used
    uses_from_provider = "from_provider" in content or "from_provider" in content

    if not uses_from_provider:
        return issues

    # Find provider imports
    import_patterns = {
        "import openai": r"^import\s+openai\b",
        "from openai import": r"^from\s+openai\s+import",
        "import anthropic": r"^import\s+anthropic\b",
        "from anthropic import": r"^from\s+anthropic\s+import",
    }

    lines = content.split("\n")
    for line_num, line in enumerate(lines, 1):
        for name, pattern in import_patterns.items():
            if re.search(pattern, line):
                # Check if the import is actually used
                if name.startswith("import "):
                    module = name.split()[1]
                    # Simple check - if module name appears elsewhere, might be used
                    if content.count(module) <= 2:  # Just import and maybe one use
                        issues[name].append(line_num)

    return issues


def process_file(file_path: Path) -> Dict[str, Dict[str, List[int]]]:
    """Process a single file and return all issues."""
    try:
        content = file_path.read_text(encoding="utf-8")

        return {
            "api_calls": audit_api_calls(content, file_path),
            "old_init": audit_old_init_patterns(content, file_path),
            "unused_imports": audit_unused_imports(content, file_path),
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {"api_calls": {}, "old_init": {}, "unused_imports": {}}


def main():
    parser = argparse.ArgumentParser(
        description="Audit documentation files for old patterns"
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("docs"),
        help="Directory containing documentation files (default: docs)",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Audit a single file instead of all files",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show only summary statistics",
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
        if any(issues.values()):
            all_issues[str(file_path)] = issues

            # Count totals
            for issue_type, patterns in issues.items():
                for pattern, line_nums in patterns.items():
                    total_counts[f"{issue_type}:{pattern}"] += len(line_nums)

    if args.summary:
        print("Summary Statistics:")
        print("=" * 60)
        for key, count in sorted(total_counts.items()):
            issue_type, pattern = key.split(":", 1)
            print(f"  {pattern}: {count} instances")
    else:
        # Detailed report
        for file_path, issues in sorted(all_issues.items()):
            print(f"\n{file_path}:")
            print("-" * 60)

            for issue_type, patterns in issues.items():
                if patterns:
                    print(f"  {issue_type.replace('_', ' ').title()}:")
                    for pattern, line_nums in sorted(patterns.items()):
                        lines_str = ", ".join(map(str, line_nums[:10]))
                        if len(line_nums) > 10:
                            lines_str += f", ... ({len(line_nums)} total)"
                        print(f"    {pattern}: lines {lines_str}")

    print(f"\nTotal files with issues: {len(all_issues)}")
    print(f"Total issues found: {sum(total_counts.values())}")


if __name__ == "__main__":
    main()
