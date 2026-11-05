from pathlib import Path
import tarfile
import zipfile

import pytest

from scripts.verify_release_artifacts import verify_metadata, verify_sdist, verify_wheel


def test_wheel_rejects_an_extra_python_startup_hook(tmp_path: Path) -> None:
    wheel = tmp_path / "instructor.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("unexpected.pth", "import sys\n")
    with pytest.raises(ValueError, match="Unexpected.*wheel files"):
        verify_wheel(wheel, tmp_path, {}, {"version": "1"})


def test_wheel_rejects_an_undeclared_dependency() -> None:
    project = {"name": "instructor", "version": "1", "requires-python": ">=3.10"}
    metadata = b"Name: instructor\nVersion: 1\nRequires-Python: >=3.10\nRequires-Dist: unexpected\n"
    with pytest.raises(ValueError, match="dependencies differ"):
        verify_metadata(metadata, project)


def test_source_archive_rejects_an_external_link(tmp_path: Path) -> None:
    sdist = tmp_path / "instructor.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        member = tarfile.TarInfo("instructor-1/README.md")
        member.type = tarfile.SYMTYPE
        member.linkname = "../outside"
        archive.addfile(member)
    with pytest.raises(ValueError, match="source archive member"):
        verify_sdist(sdist, {"README.md": tmp_path / "README.md"}, b"", "1")
