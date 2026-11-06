#!/usr/bin/env python3
"""Fix doc test formatting issues using --update-examples for each test file."""

import subprocess
import sys
from pathlib import Path

test_files = [
    "tests/docs/test_concepts_operations.py",
    "tests/docs/test_examples_batch.py",
    "tests/docs/test_examples_integrations.py",
    "tests/docs/test_examples_multimodal.py",
    "tests/docs/test_posts.py",
]


def run_update(test_file: str) -> bool:
    """Run --update-examples on a test file."""
    print(f"\n{'=' * 60}")
    print(f"Processing: {test_file}")
    print(f"{'=' * 60}")

    cmd = ["uv", "run", "pytest", test_file, "--update-examples", "-q", "--tb=no"]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent
        )

        if result.returncode == 0:
            print(f"✓ Successfully updated {test_file}")
            return True
        else:
            # Even with errors, some files might have been updated
            print(f"⚠ Completed {test_file} with exit code {result.returncode}")
            if result.stdout:
                print("STDOUT:", result.stdout[-500:])  # Last 500 chars
            return False
    except Exception as e:
        print(f"✗ Error processing {test_file}: {e}")
        return False


if __name__ == "__main__":
    success_count = 0
    for test_file in test_files:
        if run_update(test_file):
            success_count += 1

    print(f"\n{'=' * 60}")
    print(f"Summary: {success_count}/{len(test_files)} files processed")
    print(f"{'=' * 60}")

    sys.exit(0 if success_count == len(test_files) else 1)
