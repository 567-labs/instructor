"""Regression tests for #2732: usage helpers must tolerate missing openai."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from instructor.v2.core import usage as usage_mod


def test_has_compatible_usage_without_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "openai" or name.startswith("openai."):
            raise ImportError("No module named 'openai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert (
        usage_mod.has_compatible_usage(SimpleNamespace(usage=object()), object())
        is False
    )


def test_update_total_usage_without_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "openai" or name.startswith("openai."):
            raise ImportError("No module named 'openai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    response = SimpleNamespace(usage=object())
    # total_usage type is opaque here; without openai/anthropic match we no-op.
    assert usage_mod.update_total_usage(response, object()) is response
