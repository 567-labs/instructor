"""Verify Instructor distributions against a clean checkout before installation."""

from __future__ import annotations

import argparse
import base64
import configparser
import csv
import hashlib
import io
import sys
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version


class EntryPointConfig(configparser.ConfigParser):
    def optionxform(self, optionstr: str) -> str:
        return optionstr


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def expected_requirements(project: dict) -> set[Requirement]:
    requirements = {Requirement(value) for value in project.get("dependencies", [])}
    for extra, values in project.get("optional-dependencies", {}).items():
        for value in values:
            requirement = Requirement(value)
            extra_marker = f'extra == "{canonicalize_name(extra)}"'
            requirement.marker = Marker(
                f"({requirement.marker}) and {extra_marker}"
                if requirement.marker
                else extra_marker
            )
            requirements.add(requirement)
    return requirements


def verify_metadata(payload: bytes, project: dict) -> None:
    metadata = BytesParser().parsebytes(payload)
    require(
        canonicalize_name(metadata["Name"]) == canonicalize_name(project["name"]),
        "Package name differs from source",
    )
    require(
        Version(metadata["Version"]) == Version(project["version"]),
        "Package version differs from source",
    )
    require(
        SpecifierSet(metadata["Requires-Python"])
        == SpecifierSet(project["requires-python"]),
        "Python requirement differs from source",
    )
    require(
        {Requirement(value) for value in metadata.get_all("Requires-Dist", [])}
        == expected_requirements(project),
        "Distribution dependencies differ from source",
    )
    require(
        {canonicalize_name(value) for value in metadata.get_all("Provides-Extra", [])}
        == {
            canonicalize_name(value)
            for value in project.get("optional-dependencies", {})
        },
        "Distribution extras differ from source",
    )


def verify_wheel(
    path: Path, source: Path, package: dict[str, Path], project: dict
) -> bytes:
    info = f"instructor-{project['version']}.dist-info"
    record = f"{info}/RECORD"
    generated = {
        f"{info}/{name}"
        for name in (
            "METADATA",
            "WHEEL",
            "entry_points.txt",
            "licenses/LICENSE",
            "RECORD",
        )
    }
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        require(
            len(names) == len(set(names)) and set(names) == package.keys() | generated,
            "Unexpected, missing, or duplicate wheel files",
        )
        rows: dict[str, tuple[str, str]] = {}
        payloads: dict[str, bytes] = {}
        for name in names:
            member = wheel.getinfo(name)
            require(
                (member.external_attr >> 16) & 0o170000 != 0o120000,
                "Wheel contains a symbolic link",
            )
            original = package.get(name)
            limit = original.stat().st_size if original is not None else 1024 * 1024
            require(member.file_size <= limit, f"Unexpected wheel member size: {name}")
            value = wheel.read(member)
            if original is not None:
                require(
                    value == original.read_bytes(),
                    f"Wheel file differs from source: {name}",
                )
            else:
                payloads[name] = value
            digest = (
                base64.urlsafe_b64encode(hashlib.sha256(value).digest())
                .rstrip(b"=")
                .decode()
            )
            rows[name] = ("sha256=" + digest, str(len(value)))
        rows[record] = ("", "")
        declared = list(csv.reader(io.StringIO(payloads[record].decode())))
        require(
            len(declared) == len(rows) and all(len(row) == 3 for row in declared),
            "Invalid wheel RECORD",
        )
        require(
            {row[0]: (row[1], row[2]) for row in declared} == rows,
            "Wheel RECORD hashes or sizes differ",
        )
        metadata = payloads[f"{info}/METADATA"]
        verify_metadata(metadata, project)
        wheel_info = BytesParser().parsebytes(payloads[f"{info}/WHEEL"])
        require(
            wheel_info.get_all("Tag") == ["py3-none-any"]
            and wheel_info["Root-Is-Purelib"] == "true",
            "Unexpected wheel platform or install layout",
        )
        entries = EntryPointConfig(interpolation=None)
        entries.read_string(payloads[f"{info}/entry_points.txt"].decode())
        require(
            entries.sections() == ["console_scripts"]
            and dict(entries["console_scripts"]) == project.get("scripts", {}),
            "Entry points differ from source",
        )
        require(
            payloads[f"{info}/licenses/LICENSE"] == (source / "LICENSE").read_bytes(),
            "License differs from source",
        )
        return metadata


def verify_sdist(
    path: Path, files: dict[str, Path], metadata: bytes, version: str
) -> None:
    expected = set(files) | {"PKG-INFO"}
    directories = {
        str(parent) for name in expected for parent in PurePosixPath(name).parents
    }
    seen: set[str] = set()
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            prefix, _, name = member.name.partition("/")
            require(prefix == f"instructor-{version}", "Unexpected source archive root")
            if member.isdir():
                require(
                    (name or ".") in directories, "Unexpected source archive directory"
                )
                continue
            require(
                member.isfile() and name in expected and name not in seen,
                "Unexpected, missing, or duplicate source archive member",
            )
            seen.add(name)
            original = metadata if name == "PKG-INFO" else files[name].read_bytes()
            stream = archive.extractfile(member)
            require(
                stream is not None and member.size == len(original),
                f"Unexpected source archive member size: {name}",
            )
            if stream is None or stream.read() != original:
                raise ValueError(f"Source archive file differs from checkout: {name}")
    require(seen == expected, "Source archive is incomplete")


def verify(source: Path, dist: Path) -> None:
    project = tomllib.loads((source / "pyproject.toml").read_text())["project"]
    version = project["version"]
    wheel = dist / f"instructor-{version}-py3-none-any.whl"
    sdist = dist / f"instructor-{version}.tar.gz"
    require(
        {file for file in dist.iterdir() if file.name != ".gitignore"}
        == {wheel, sdist},
        "Expected exactly one Instructor wheel and one source archive",
    )
    package = {
        file.relative_to(source).as_posix(): file
        for file in (source / "instructor").rglob("*")
        if file.is_file() and "__pycache__" not in file.parts and file.suffix != ".pyc"
    }
    require(
        not any(
            file.is_symlink()
            or name.endswith(".pth")
            or file.name in {"sitecustomize.py", "usercustomize.py"}
            for name, file in package.items()
        ),
        "Unexpected package link or Python startup hook",
    )
    metadata = verify_wheel(wheel, source, package, project)
    files = {
        **package,
        **{
            name: source / name
            for name in (".gitignore", "LICENSE", "README.md", "pyproject.toml")
        },
    }
    verify_sdist(sdist, files, metadata, version)
    print(
        f"Verified {len(package)} package files, metadata, entry points and RECORD against the checkout."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist", type=Path)
    parser.add_argument("--source", type=Path, default=Path("."))
    args = parser.parse_args()
    verify(args.source.resolve(), args.dist.resolve())
