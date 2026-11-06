#!/usr/bin/env python3
"""
Check if blog posts contain the <!-- more --> tag for excerpts.

This script:
- Recursively finds all .md files in the docs/blog/posts directory
- Checks if each file contains the <!-- more --> tag
- Reports files that are missing the tag
- Exits with error code 1 if any files are missing the tag
"""

import sys
from pathlib import Path


def check_blog_excerpts(blog_posts_dir: str = "docs/blog/posts") -> bool:
    """
    Check if blog posts contain the <!-- more --> tag.

    Args:
        blog_posts_dir: Path to the blog posts directory (default: "docs/blog/posts")

    Returns:
        True if all files have the tag, False if any are missing it
    """
    blog_path = Path(blog_posts_dir)

    if not blog_path.exists():
        print(f"Error: Directory '{blog_posts_dir}' does not exist.")
        return False

    if not blog_path.is_dir():
        print(f"Error: '{blog_posts_dir}' is not a directory.")
        return False

    # Find all markdown files recursively
    md_files = list(blog_path.rglob("*.md"))

    if not md_files:
        print(f"No markdown files found in '{blog_posts_dir}' directory.")
        return True

    print(f"Checking {len(md_files)} blog post files for <!-- more --> tag...")

    missing_tag_files = []

    for md_file in md_files:
        try:
            # Read the file content
            with open(md_file, encoding="utf-8") as f:
                content = f.read()

            # Check if the file contains the <!-- more --> tag
            if "<!-- more -->" not in content:
                missing_tag_files.append(md_file)
                print(f"Missing <!-- more --> tag: {md_file}")
            else:
                print(f"✓ Has <!-- more --> tag: {md_file}")

        except Exception as e:
            print(f"Error reading {md_file}: {e}")
            missing_tag_files.append(md_file)

    # Summary
    if missing_tag_files:
        print(f"\n❌ Found {len(missing_tag_files)} files missing <!-- more --> tag:")
        for file in missing_tag_files:
            print(f"  - {file}")
        print(
            f"\nPlease add <!-- more --> tag to these files for proper excerpt handling."
        )
        return False
    else:
        print(f"\n✅ All {len(md_files)} blog post files have the <!-- more --> tag!")
        return True


def main():
    """Main function to handle command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Check if blog posts contain the <!-- more --> tag for excerpts"
    )
    parser.add_argument(
        "--blog-posts-dir",
        default="docs/blog/posts",
        help="Path to blog posts directory (default: docs/blog/posts)",
    )

    args = parser.parse_args()

    success = check_blog_excerpts(blog_posts_dir=args.blog_posts_dir)

    # Exit with appropriate code for pre-commit
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
