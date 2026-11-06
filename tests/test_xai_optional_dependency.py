import sys
import pytest


def test_from_provider_xai_requires_optional_extra(monkeypatch):
    """Test that from_provider('xai/...') raises ConfigurationError when xai_sdk is missing."""
    import instructor
    from instructor.core.exceptions import ConfigurationError

    # Simulate xai_sdk being absent by temporarily hiding it
    monkeypatch.setitem(sys.modules, "xai_sdk", None)
    monkeypatch.setitem(sys.modules, "xai_sdk.sync", None)
    monkeypatch.setitem(sys.modules, "xai_sdk.sync.client", None)
    monkeypatch.setitem(sys.modules, "xai_sdk.aio", None)
    monkeypatch.setitem(sys.modules, "xai_sdk.aio.client", None)

    with pytest.raises(ConfigurationError) as excinfo:
        instructor.from_provider("xai/grok-3-mini", api_key="test-key")

    msg = str(excinfo.value)
    assert "instructor[xai]" in msg
    assert "uv pip install" in msg


def test_direct_from_xai_has_clear_error_when_sdk_missing(monkeypatch):
    """Test that from_xai() raises ClientError when xai_sdk is missing."""
    from instructor.v2.core.errors import ClientError
    import instructor.v2.providers.xai.client as xai_client

    # Simulate xai_sdk being absent by setting the module-level sentinels to None
    monkeypatch.setattr(xai_client, "SyncClient", None)
    monkeypatch.setattr(xai_client, "AsyncClient", None)
    monkeypatch.setattr(xai_client, "xchat", None)

    with pytest.raises(ClientError) as excinfo:
        xai_client.from_xai(
            object()
        )  # ty: ignore[no-matching-overload] - deliberately invalid client

    msg = str(excinfo.value)
    assert "instructor[xai]" in msg
    assert "xai-sdk" in msg
