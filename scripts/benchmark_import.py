"""Benchmark instructor import costs.

Run with:

    python scripts/benchmark_import.py
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def run_import_benchmark(code: str, label: str) -> dict[str, str]:
    """Run a Python snippet in a subprocess and capture import stats."""
    script = textwrap.dedent(
        f"""
        import sys
        import time
        import tracemalloc

        tracemalloc.start()
        start = time.time()

        {code}

        elapsed = time.time() - start
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"{{elapsed:.3f}}|{{current}}|{{peak}}|{{len(sys.modules)}}")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        return {
            "label": label,
            "time": "ERROR",
            "memory_mb": "ERROR",
            "peak_mb": "ERROR",
            "modules": "ERROR",
            "error": result.stderr.strip(),
        }

    elapsed, current, peak, mod_count = result.stdout.strip().split("|")
    return {
        "label": label,
        "time": f"{float(elapsed):.3f}s",
        "memory_mb": f"{int(current) / 1024 / 1024:.1f} MB",
        "peak_mb": f"{int(peak) / 1024 / 1024:.1f} MB",
        "modules": mod_count,
    }


def main() -> None:
    print("=" * 70)
    print(" INSTRUCTOR IMPORT BENCHMARK")
    print("=" * 70)
    print(f" Python: {sys.version.split()[0]}")
    print(f" Executable: {sys.executable}")
    print("=" * 70)
    print()

    benchmarks = [
        ("import sys", "baseline (sys only)"),
        ("import instructor", "import instructor"),
        ("import instructor; _ = instructor.from_provider", "access from_provider"),
        ("import instructor; _ = instructor.from_openai", "access from_openai"),
        ("import instructor; _ = instructor.from_anthropic", "access from_anthropic"),
        ("import instructor; _ = instructor.from_genai", "access from_genai"),
        ("import instructor; _ = instructor.Mode", "access Mode"),
        ("import instructor; _ = instructor.Instructor", "access Instructor class"),
        ("import instructor; _ = instructor.Partial", "access Partial"),
    ]

    results = [run_import_benchmark(code, label) for code, label in benchmarks]

    header = f"{'Benchmark':<30} {'Time':>8} {'Memory':>10} {'Peak':>10} {'Modules':>8}"
    print(header)
    print("-" * len(header))

    for result in results:
        if result.get("error"):
            print(f"{result['label']:<30} {'ERROR':>8} (missing dependency?)")
        else:
            print(
                f"{result['label']:<30} {result['time']:>8} "
                f"{result['memory_mb']:>10} {result['peak_mb']:>10} "
                f"{result['modules']:>8}"
            )

    print()
    print("Notes:")
    print("  - 'import instructor' should stay small because of lazy loading.")
    print("  - Provider access loads each SDK on demand.")


if __name__ == "__main__":
    main()
