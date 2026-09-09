"""Summarize pytest JUnit evidence without inferring live provider certification."""

from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import xml.etree.ElementTree as ET


def summarize(report: Path) -> str:
    """Count recorded outcomes; absent/malformed reports are not zero tests."""
    heading = f"### {report.stem}\n\n"
    try:
        root = ET.parse(report).getroot()
        if root.tag not in {"testsuites", "testsuite"}:
            raise ValueError("Not a JUnit report")
    except (ET.ParseError, OSError, ValueError):
        return heading + "No usable JUnit report; execution and counts are unknown.\n"
    counts: Counter[str] = Counter()
    skips: Counter[str] = Counter()
    for case in root.iter("testcase"):
        if case.find("error") is not None:
            counts["errors"] += 1
        elif case.find("failure") is not None:
            counts["failed"] += 1
        elif (skip := case.find("skipped")) is not None:
            counts["skipped"] += 1
            reason = (skip.get("message", "") + (skip.text or "")).lower()
            if skip.get("type") == "pytest.xfail":
                category = "expected failure"
            elif any(
                word in reason
                for word in ("api_key", "api key", "secret", "credential")
            ):
                category = "credentials unavailable"
            elif any(word in reason for word in ("not support", "unsupported")):
                category = "unsupported by test capability policy"
            elif any(
                word in reason for word in ("not installed", "no module", "package")
            ):
                category = "SDK/package unavailable"
            else:
                category = "other (see pytest -rs logs)"
            skips[category] += 1
        else:
            counts["passed"] += 1
    lines = [
        heading,
        "| Passed | Failed | Errors | Skipped |",
        "| ---: | ---: | ---: | ---: |",
    ]
    lines.append(
        "| "
        + " | ".join(
            str(counts[key]) for key in ("passed", "failed", "errors", "skipped")
        )
        + " |"
    )
    if not counts:
        lines.append("\nNo test outcomes recorded; this is not provider validation.")
    elif counts["passed"] == 0:
        lines.append("\nNo passing tests recorded; this is not provider validation.")
    for reason, count in sorted(skips.items()):
        lines.append(f"\n- {reason}: {count}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument(
        "--requirements",
        nargs="*",
        default=os.environ.get("PROVIDER_TEST_REQUIREMENTS", "").split(),
    )
    args = parser.parse_args()
    lines = [
        "## Provider test evidence\n",
        "Passing tests are evidence only for their selected cases, not provider certification. Counts are JUnit outcomes, not API calls. Initial and retry runs remain separate.\n",
    ]
    for key in args.requirements:
        state = "present" if os.environ.get(key) else "missing"
        lines.append(f"- `{key}`: {state}")
    # Always represent the initial run, even when only a later report exists.
    reports = sorted(
        {args.reports / "provider-evidence-initial.xml"}
        | set(args.reports.glob("provider-evidence-*.xml"))
    )
    lines.extend(summarize(report) for report in reports)
    output = "\n".join(lines)
    print(output)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a", encoding="utf-8") as stream:
            stream.write(output + "\n")


if __name__ == "__main__":
    main()
