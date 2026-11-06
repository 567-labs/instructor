"""Check package URLs before committing a lockfile to the public repository."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlsplit

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def package_urls(value: object) -> Iterator[str]:
    if isinstance(value, dict):
        for item in value.values():
            yield from package_urls(item)
    elif isinstance(value, list):
        for item in value:
            yield from package_urls(item)
    elif isinstance(value, str) and "://" in value:
        yield value


def check_public_lock(data: bytes) -> None:
    for value in package_urls(tomllib.loads(data.decode())):
        url = urlsplit(value)
        if (
            url.scheme != "https"
            or url.netloc not in {"pypi.org", "files.pythonhosted.org"}
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "Lockfile contains a non-public or credential-bearing package URL"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", help="Lockfile path, or - to read a Git blob from stdin"
    )
    path = parser.parse_args().path
    try:
        check_public_lock(
            sys.stdin.buffer.read() if path == "-" else Path(path).read_bytes()
        )
    except (UnicodeError, ValueError) as exc:
        # A malformed URL or TOML error can contain credentials. Keep CI output generic.
        raise SystemExit("Public lockfile source validation failed") from exc
