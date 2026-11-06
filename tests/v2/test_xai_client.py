"""Provider-specific tests for xAI v2 client factory.

Note: Common tests (mode normalization, registry, imports, errors) are unified in
test_client_unified.py. This file only contains xAI-specific helper function tests.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel


class Answer(BaseModel):
    """Simple answer model for testing."""

    answer: float


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestClientHelperFunctions:
    """Tests for client helper functions."""

    def test_get_model_schema(self):
        """Test _get_model_schema extracts schema from BaseModel."""
        from instructor.v2.providers.xai.client import _get_model_schema

        schema = _get_model_schema(Answer)

        assert "properties" in schema
        assert "answer" in schema["properties"]
        assert schema["properties"]["answer"]["type"] == "number"

    def test_get_model_schema_with_no_schema_method(self):
        """Test _get_model_schema returns empty dict for non-models."""
        from instructor.v2.providers.xai.client import _get_model_schema

        class NoSchema:
            pass

        schema = _get_model_schema(NoSchema)

        assert schema == {}

    def test_get_model_name(self):
        """Test _get_model_name extracts model name."""
        from instructor.v2.providers.xai.client import _get_model_name

        name = _get_model_name(Answer)

        assert name == "Answer"

    def test_get_model_name_with_class(self):
        """Test _get_model_name extracts name from class."""
        from instructor.v2.providers.xai.client import _get_model_name

        class CustomModel:
            pass

        name = _get_model_name(CustomModel)
        assert name == "CustomModel"

    def test_finalize_parsed_response_with_base_model(self):
        """Test _finalize_parsed_response attaches raw response to BaseModel."""
        from instructor.v2.providers.xai.client import _finalize_parsed_response

        parsed = Answer(answer=42.0)
        raw_response = {"test": "response"}

        result = _finalize_parsed_response(parsed, raw_response)

        assert result is parsed
        assert hasattr(result, "_raw_response")
        assert result._raw_response == raw_response

    def test_add_md_json_instructions_prepends_system_message(self):
        from instructor.v2.providers.xai.client import _add_md_json_instructions

        messages = [{"role": "user", "content": "hello"}]

        result = _add_md_json_instructions(messages, Answer)

        assert result[0]["role"] == "system"
        assert "Schema:" in result[0]["content"]
        assert result[1:] == messages

    def test_add_md_json_instructions_appends_to_existing_system_message(self):
        from instructor.v2.providers.xai.client import _add_md_json_instructions

        messages = [
            {"role": "system", "content": "Be terse"},
            {"role": "user", "content": "hello"},
        ]

        result = _add_md_json_instructions(messages, Answer)

        assert result[0]["content"].startswith("Be terse\n\nReturn your answer as JSON")

    def test_convert_messages_uses_xchat_constructors(self, monkeypatch):
        from instructor.v2.providers.xai import client as xai_client

        xchat = SimpleNamespace(
            text=lambda value: ("text", value),
            user=lambda value: ("user", value),
            assistant=lambda value: ("assistant", value),
            system=lambda value: ("system", value),
            tool_result=lambda value: ("tool", value),
        )
        monkeypatch.setattr(xai_client, "xchat", xchat)

        result = xai_client._convert_messages(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "there"},
                {"role": "system", "content": "rules"},
                {"role": "tool", "content": "tool output"},
            ]
        )

        assert result == [
            ("user", ("text", "hi")),
            ("assistant", ("text", "there")),
            ("system", ("text", "rules")),
            ("tool", "tool output"),
        ]

    def test_convert_messages_rejects_non_string_content(self, monkeypatch):
        from instructor.v2.providers.xai import client as xai_client

        monkeypatch.setattr(
            xai_client,
            "xchat",
            SimpleNamespace(text=lambda value: value, user=lambda value: value),
        )

        with pytest.raises(ValueError, match="Only string content supported"):
            xai_client._convert_messages([{"role": "user", "content": ["bad"]}])

    def test_iter_tool_call_arg_deltas_emits_only_new_suffixes(self):
        from instructor.v2.providers.xai.client import _iter_tool_call_arg_deltas

        def call(tool_id: str, arguments):
            return SimpleNamespace(
                id=tool_id,
                function=SimpleNamespace(arguments=arguments),
            )

        stream = iter(
            [
                (SimpleNamespace(tool_calls=[call("1", '{"a":1')]), None),
                (SimpleNamespace(tool_calls=[call("1", '{"a":1,"b":2}')]), None),
                (SimpleNamespace(tool_calls=[call("2", {"ok": True})]), None),
            ]
        )

        assert list(_iter_tool_call_arg_deltas(stream)) == [
            '{"a":1',
            ',"b":2}',
            '{"ok": true}',
        ]

    @pytest.mark.asyncio
    async def test_aiter_tool_call_arg_deltas_emits_only_new_suffixes(self):
        from instructor.v2.providers.xai.client import _aiter_tool_call_arg_deltas

        def call(tool_id: str, arguments):
            return SimpleNamespace(
                id=tool_id,
                function=SimpleNamespace(arguments=arguments),
            )

        async def stream():
            yield SimpleNamespace(tool_calls=[call("1", '{"a":1')]), None
            yield SimpleNamespace(tool_calls=[call("1", '{"a":1,"b":2}')]), None

        assert [delta async for delta in _aiter_tool_call_arg_deltas(stream())] == [
            '{"a":1',
            ',"b":2}',
        ]


# ============================================================================
# Provider-Specific Tests
# ============================================================================
# Note: Common tests (mode normalization, registry, imports, errors) are
# unified in test_client_unified.py. This file only contains xAI-specific
# helper function tests.


# ============================================================================
# Integration Tests (require xAI SDK but not API key)
# ============================================================================


@pytest.mark.skipif(
    True,  # Skip by default since xAI SDK may not be installed
    reason="xAI SDK not installed",
)
class TestXAIClientWithSDK:
    """Tests that require xAI SDK but not API key."""

    def test_from_xai_with_invalid_client(self):
        """Test from_xai raises error with invalid client."""
        from instructor.v2.providers.xai.client import from_xai
        from instructor.core.exceptions import ClientError

        with pytest.raises(ClientError, match="must be an instance"):
            from_xai("not a client")  # ty: ignore[no-matching-overload]

    def test_from_xai_with_invalid_mode(self):
        """Test from_xai raises error with invalid mode."""

        # This would require a valid client, so we skip
        pass
