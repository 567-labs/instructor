from __future__ import annotations

from typing import Any, cast

import pytest

from instructor import Mode
from instructor.v2.core.client import Instructor
from instructor.v2.core.errors import ClientError, InstructorRetryException
from instructor.v2.core.providers import Provider
from instructor.v2.providers.genai import client as genai_client


def test_anthropic_factory_reports_missing_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    anthropic_client = pytest.importorskip("instructor.v2.providers.anthropic.client")
    monkeypatch.setattr(anthropic_client, "anthropic", None)

    with pytest.raises(ClientError, match="anthropic is not installed"):
        anthropic_client.from_anthropic(object())


def test_genai_factory_exposes_normalized_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModels:
        def generate_content(self, **_kwargs: Any) -> None:
            return None

        def generate_content_stream(self, **_kwargs: Any) -> None:
            return None

    class FakeClient:
        def __init__(self) -> None:
            self.models = FakeModels()

    monkeypatch.setattr(genai_client, "Client", FakeClient)

    client = genai_client.from_genai(cast(Any, FakeClient()), mode=Mode.GENAI_TOOLS)

    assert isinstance(client, Instructor)
    assert client.mode is Mode.TOOLS


def test_openai_connection_errors_only_skip_outside_strict_provider_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.v2.test_provider_modes import _skip_on_provider_quota

    exc = InstructorRetryException(
        ValueError("Connection error"),
        last_completion=None,
        n_attempts=1,
        messages=[],
        create_kwargs={},
        total_usage=0,
    )

    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("INSTRUCTOR_STRICT_PROVIDER_TESTS", raising=False)
    with pytest.raises(pytest.skip.Exception):
        _skip_on_provider_quota(Provider.OPENAI, exc)

    monkeypatch.setenv("CI", "true")
    _skip_on_provider_quota(Provider.OPENAI, exc)
