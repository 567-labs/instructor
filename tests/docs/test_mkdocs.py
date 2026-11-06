import pathlib
import pytest
import importlib
from typing import Any, cast


# Note the use of `str`, makes for pretty output
@pytest.mark.parametrize(
    "fpath", pathlib.Path("docs/examples").glob("**/*.md"), ids=str
)
@pytest.mark.skip(reason="This test is not yet implemented")
def test_files_good(fpath):
    mktestdocs = cast(Any, importlib.import_module("mktestdocs"))
    check_md_file = mktestdocs.check_md_file

    check_md_file(fpath=fpath, memory=True)
