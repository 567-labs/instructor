import importlib
import json
import subprocess
import sys
from typing import TypedDict


class ColdImportState(TypedDict):
    modules: list[str]
    count: int


def _cold_import_state(code: str) -> ColdImportState:
    probe = """
import json
import sys
exec(sys.argv[1], {})
mods = sorted(
    name for name in sys.modules
    if name == "instructor" or name.startswith("instructor.")
)
print(json.dumps(mods))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe, code],
        check=True,
        capture_output=True,
        text=True,
    )
    modules = json.loads(result.stdout)
    return {"modules": modules, "count": len(modules)}


def test_top_level_exports_load_lazily():
    sys.modules.pop("instructor", None)
    sys.modules.pop("instructor.processing.response", None)
    sys.modules.pop("instructor.processing.multimodal", None)

    instructor = importlib.import_module("instructor")

    assert "instructor.processing.response" not in sys.modules
    assert "instructor.processing.multimodal" not in sys.modules
    assert instructor.Mode.TOOLS.value == "tool_call"
    assert "instructor.processing.response" not in sys.modules


def test_top_level_openai_factory_uses_v2_module():
    sys.modules.pop("instructor", None)
    sys.modules.pop("instructor.v2.providers.openai.client", None)

    instructor = importlib.import_module("instructor")

    assert "instructor.v2.providers.openai.client" not in sys.modules
    assert instructor.from_openai is not None
    assert "instructor.v2.providers.openai.client" in sys.modules


def test_mode_export_stays_lightweight():
    state = _cold_import_state("from instructor import Mode")

    assert state["count"] <= 7
    assert "instructor.v2.core.mode" in state["modules"]
    assert not any(
        module.startswith("instructor.v2.providers.") for module in state["modules"]
    )


def test_v2_module_export_stays_lightweight():
    state = _cold_import_state("import instructor; instructor.v2")

    assert state["count"] <= 6
    assert state["modules"] == [
        "instructor",
        "instructor.v2",
        "instructor.v2.core",
        "instructor.v2.core.mode",
        "instructor.v2.core.provider_specs",
        "instructor.v2.core.providers",
    ]


def test_auto_client_keeps_bedrock_provider_lazy():
    state = _cold_import_state("from instructor.v2 import auto_client")

    assert "instructor.v2.auto_client" in state["modules"]
    assert "instructor.v2.providers.bedrock.client" not in state["modules"]
