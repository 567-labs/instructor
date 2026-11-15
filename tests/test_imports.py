"""Lazy imports and missing optional SDKs.

Cold-import checks share one subprocess: each step accesses more of the public
API and records which modules are loaded, so a regression names the access
that pulled in the heavy module.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any, cast

import pytest

import instructor
from instructor import Mode
from instructor.core.exceptions import ConfigurationError
from instructor.v2.core.client import Instructor
from instructor.v2.core.errors import ClientError
from instructor.v2.providers.genai import client as genai_client

COLD_IMPORT_PROBE = """
import json
import sys

steps = {}


def record(step):
    steps[step] = {
        "instructor": sorted(
            name
            for name in sys.modules
            if name == "instructor" or name.startswith("instructor.")
        ),
        "openai": sorted(name for name in sys.modules if name.startswith("openai")),
    }


import instructor

record("import")
instructor.v2
record("v2")
from instructor import Mode

assert Mode.TOOLS.value == "tool_call"
record("Mode")
for symbol in ("from_provider", "Instructor", "AsyncInstructor", "Partial", "Maybe"):
    assert getattr(instructor, symbol) is not None
    record(symbol)
from instructor.v2 import auto_client

record("auto_client")
assert instructor.from_openai is not None
record("from_openai")
print(json.dumps(steps))
"""


def test_cold_import_stays_lazy() -> None:
    result = subprocess.run(
        [sys.executable, "-c", COLD_IMPORT_PROBE],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    steps: dict[str, dict[str, list[str]]] = json.loads(result.stdout)

    # The OpenAI SDK loads only when the OpenAI factory is requested.
    for step, state in steps.items():
        if step != "from_openai":
            assert state["openai"] == [], f"openai loaded after {step}"
        assert "instructor.processing.response" not in state["instructor"], step
        assert "instructor.processing.multimodal" not in state["instructor"], step
    assert (
        "instructor.v2.providers.openai.client"
        not in steps["auto_client"]["instructor"]
    )
    assert "instructor.v2.providers.openai.client" in steps["from_openai"]["instructor"]

    assert steps["v2"]["instructor"] == ["instructor", "instructor.v2"]
    mode_modules = steps["Mode"]["instructor"]
    assert len(mode_modules) <= 7
    assert "instructor.v2.core.mode" in mode_modules
    assert not any(name.startswith("instructor.v2.providers.") for name in mode_modules)
    auto_modules = steps["auto_client"]["instructor"]
    assert "instructor.v2.auto_client" in auto_modules
    assert "instructor.v2.providers.bedrock.client" not in auto_modules


def test_v2_gateway_factories_are_available_from_all() -> None:
    from instructor import v2

    assert "from_zai" in v2.__all__
    assert v2.from_zai is instructor.from_zai


def test_from_provider_xai_requires_optional_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "xai_sdk",
        "xai_sdk.sync",
        "xai_sdk.sync.client",
        "xai_sdk.aio",
        "xai_sdk.aio.client",
    ):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(ConfigurationError) as excinfo:
        instructor.from_provider("xai/grok-3-mini", api_key="test-key")
    assert "instructor[xai]" in str(excinfo.value)
    assert "uv pip install" in str(excinfo.value)


@pytest.mark.parametrize(
    "module,sentinels,factory,message",
    [
        (
            "instructor.v2.providers.xai.client",
            ("SyncClient", "AsyncClient", "xchat"),
            "from_xai",
            r"instructor\[xai\].*xai-sdk|xai-sdk.*instructor\[xai\]",
        ),
        (
            "instructor.v2.providers.anthropic.client",
            ("anthropic",),
            "from_anthropic",
            "anthropic is not installed",
        ),
    ],
    ids=["xai", "anthropic"],
)
def test_direct_factory_reports_missing_sdk(
    monkeypatch: pytest.MonkeyPatch,
    module: str,
    sentinels: tuple[str, ...],
    factory: str,
    message: str,
) -> None:
    client_module = pytest.importorskip(module)
    for name in sentinels:
        monkeypatch.setattr(client_module, name, None)
    with pytest.raises(ClientError, match=message):
        getattr(client_module, factory)(object())


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
