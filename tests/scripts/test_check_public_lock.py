import pytest

from scripts.check_public_lock import check_public_lock


def test_public_lock_accepts_index_and_distribution_urls() -> None:
    check_public_lock(
        b'[[package]]\nsource = {registry = "https://pypi.org/simple"}\nwheels = [{url = "https://files.pythonhosted.org/packages/example.whl"}]'
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://registry.example.invalid/simple",
        "https://reader@pypi.org/simple",
        "https://files.pythonhosted.org/packages/example.whl?signature=example",
        "http://pypi.org/simple",
    ],
)
def test_public_lock_rejects_private_or_credential_bearing_urls(url: str) -> None:
    with pytest.raises(ValueError, match="non-public or credential-bearing") as error:
        check_public_lock(f'[[package]]\nsource = {{registry = "{url}"}}'.encode())
    assert url not in str(error.value)
