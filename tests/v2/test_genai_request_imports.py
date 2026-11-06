"""Import-boundary tests for GenAI request configuration ownership."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


def _run_isolated(code: str) -> None:
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        check=True,
        text=True,
        timeout=30,
    )


def test_public_utility_resolves_to_genai_owner() -> None:
    from instructor import utils
    from instructor.v2.providers.genai.request import update_genai_kwargs

    assert utils.update_genai_kwargs is update_genai_kwargs


@pytest.mark.parametrize(
    "first_provider, second_provider",
    [
        ("instructor.v2.providers.gemini", "instructor.v2.providers.genai"),
        ("instructor.v2.providers.genai", "instructor.v2.providers.gemini"),
    ],
)
def test_provider_import_orders_preserve_compatibility(
    first_provider: str, second_provider: str
) -> None:
    _run_isolated(
        f"""
        import importlib

        importlib.import_module({first_provider!r})
        importlib.import_module({second_provider!r})

        from instructor.providers.gemini.utils import update_genai_kwargs as legacy
        from instructor.utils import update_genai_kwargs as public
        from instructor.v2.providers.gemini.utils import update_genai_kwargs as legacy_v2
        from instructor.v2.providers.genai.request import update_genai_kwargs as owned

        assert legacy is legacy_v2
        assert public is owned
        """
    )


@pytest.mark.parametrize(
    ("blocked_sdk", "provider_client"),
    [
        ("google.genai", "instructor.v2.providers.genai.client"),
        ("google.generativeai", "instructor.v2.providers.gemini.client"),
    ],
    ids=["without-google-genai", "without-google-generativeai"],
)
def test_provider_imports_remain_lazy_without_each_optional_sdk(
    blocked_sdk: str, provider_client: str
) -> None:
    _run_isolated(
        f"""
        import importlib
        import importlib.util
        import sys

        blocked = {blocked_sdk!r}
        real_find_spec = importlib.util.find_spec

        def find_spec(name, *args, **kwargs):
            return None if name == blocked else real_find_spec(name, *args, **kwargs)

        importlib.util.find_spec = find_spec
        sys.modules[blocked] = None

        importlib.import_module({provider_client!r})
        from instructor.providers.gemini.utils import update_genai_kwargs as legacy
        from instructor.utils import update_genai_kwargs as public
        from instructor.v2.providers.genai.request import update_genai_kwargs as owned

        assert callable(legacy)
        assert public is owned
        """
    )
