"""Tests for from_provider() routing to v2.

Verifies that from_provider("anthropic/...") routes to v2 implementation.
"""

import importlib.util
import pytest


def _clear_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "ALL_PROXY",
        "all_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.skip(reason="Requires Anthropic API key")
@pytest.mark.parametrize("async_client", [False, True], ids=["sync", "async"])
def test_from_provider_routes_to_v2(async_client: bool):
    """Test that from_provider() routes Anthropic to v2."""
    import instructor

    # from_provider should route to v2 for Anthropic
    client = instructor.from_provider(
        "anthropic/claude-3-5-sonnet-20241022",
        async_client=async_client,
    )

    assert client is not None
    # Verify it's using v2 by checking the mode is a tuple
    assert isinstance(client.mode, tuple)
    assert len(client.mode) == 2

    if async_client:
        from instructor import AsyncInstructor

        assert isinstance(client, AsyncInstructor)


@pytest.mark.parametrize(
    "client_class_name",
    ["Anthropic", "AsyncAnthropic"],
    ids=["sync", "async"],
)
def test_top_level_from_anthropic_routes_to_v2(
    client_class_name: str, monkeypatch: pytest.MonkeyPatch
):
    """Top-level from_anthropic should now route directly to the v2 implementation."""
    if importlib.util.find_spec("anthropic") is None:
        pytest.skip("anthropic package is not installed")
    import anthropic
    from instructor import from_anthropic

    _clear_proxy_env(monkeypatch)

    client_class = getattr(anthropic, client_class_name)
    client = client_class()

    instructor_client = from_anthropic(client)
    assert instructor_client is not None


@pytest.mark.skip(reason="Requires Anthropic API key")
def test_from_provider_with_mode_compatibility():
    """Test that from_provider() handles v1 Mode enum for compatibility."""
    import instructor

    # Passing v1 Mode should still work (gets converted to v2 Mode)
    client = instructor.from_provider(
        "anthropic/claude-3-5-sonnet-20241022", mode=instructor.Mode.TOOLS
    )

    assert client is not None
    # Should be converted to v2 tuple mode
    assert isinstance(client.mode, tuple)
