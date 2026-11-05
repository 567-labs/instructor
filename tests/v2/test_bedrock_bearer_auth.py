"""Verify the bearer token on real botocore requests without sending them."""

import os

import pytest

from instructor import from_provider


def test_bedrock_bearer_token_is_scoped_to_each_client(monkeypatch):
    pytest.importorskip("boto3")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "ambient")
    model = "us.anthropic.claude-opus-5-5"
    clients = [
        from_provider(f"bedrock/{model}", api_key=token, region="us-east-1").client
        for token in ("first", "second")
    ]
    clients.append(from_provider(f"bedrock/{model}", region="us-east-1").client)

    class Captured(Exception):
        pass

    def capture(request, **_kwargs):
        raise Captured(request.headers["Authorization"])

    for client, token in zip(clients, ("first", "second", "ambient")):
        assert client is not None
        client.meta.events.register_first(
            "before-send.bedrock-runtime.Converse", capture
        )
        with pytest.raises(Captured) as caught:
            client.converse(
                modelId=model,
                messages=[{"role": "user", "content": [{"text": "hello"}]}],
            )
        assert caught.value.args == (f"Bearer {token}".encode(),)
        client.close()
    assert os.environ["AWS_BEARER_TOKEN_BEDROCK"] == "ambient"
