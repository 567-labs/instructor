"""Provider-specific tests for Bedrock v2 client factory."""

from __future__ import annotations

import pytest

from instructor import Mode


def test_from_provider_constructs_real_bedrock_sdk_client_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("boto3")
    from botocore.client import BaseClient

    from instructor import AsyncInstructor, from_provider

    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "environment-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "environment-secret")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-2")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)

    wrapped = from_provider(
        "bedrock/anthropic.claude-test",
        async_client=True,
        region="us-west-2",
        aws_access_key_id="explicit-access",
        aws_secret_access_key="explicit-secret",
    )

    assert isinstance(wrapped, AsyncInstructor)
    assert isinstance(wrapped.client, BaseClient)
    assert wrapped.client.meta.service_model.service_name == "bedrock-runtime"
    assert wrapped.client.meta.region_name == "us-west-2"
    assert wrapped.mode is Mode.TOOLS
    credentials = wrapped.client._request_signer._credentials
    assert credentials.access_key == "explicit-access"
    assert credentials.secret_key == "explicit-secret"


class TestBedrockClientWithSDK:
    """Tests for Bedrock client factory that require botocore."""

    @pytest.fixture
    def bedrock_available(self):
        """Check if botocore is available."""
        try:
            from botocore.client import BaseClient  # noqa: F401

            return True
        except ImportError:
            return False

    def test_from_bedrock_raises_without_sdk(self, bedrock_available):
        """from_bedrock should raise when botocore is missing."""
        if bedrock_available:
            pytest.skip(
                "botocore is installed"  # ty: ignore[too-many-positional-arguments]
            )

        from instructor.v2.providers.bedrock.client import from_bedrock
        from instructor.core.exceptions import ClientError

        with pytest.raises(ClientError, match="botocore is not installed"):
            from_bedrock(None)  # ty: ignore[no-matching-overload]

    def test_from_bedrock_with_invalid_client(self, bedrock_available):
        """from_bedrock should reject non-BaseClient objects."""
        if not bedrock_available:
            pytest.skip(
                "botocore not installed"  # ty: ignore[too-many-positional-arguments]
            )

        from instructor.v2.providers.bedrock.client import from_bedrock
        from instructor.core.exceptions import ClientError

        with pytest.raises(ClientError, match="BaseClient"):
            from_bedrock("not a client")  # ty: ignore[no-matching-overload]

    def test_from_bedrock_with_invalid_mode(self, bedrock_available):
        """from_bedrock should raise for unsupported modes."""
        if not bedrock_available:
            pytest.skip(
                "botocore not installed"  # ty: ignore[too-many-positional-arguments]
            )

        from botocore.client import BaseClient
        from instructor.v2.providers.bedrock.client import from_bedrock
        from instructor.core.exceptions import ModeError

        def _converse(**_kwargs):
            return {}

        client = BaseClient.__new__(BaseClient)
        client.converse = _converse  # type: ignore[assignment]

        with pytest.raises(ModeError):
            from_bedrock(client, mode=Mode.PARALLEL_TOOLS)
