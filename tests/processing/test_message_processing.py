"""
Tests for message processing optimizations.
"""

from typing import cast

from openai.types.chat import ChatCompletionMessageParam

from instructor.utils import (
    merge_consecutive_messages,
    get_message_content,
    transform_to_gemini_prompt,
    update_gemini_kwargs,
    combine_system_messages,
    extract_system_messages,
    SystemMessage,
)


class TestMergeConsecutiveMessages:
    """Test the merge_consecutive_messages function."""

    def test_empty_messages(self):
        """Test merging empty messages list."""
        result = merge_consecutive_messages([])
        assert result == []

    def test_single_message(self):
        """Test merging a single message."""
        messages = [{"role": "user", "content": "Hello"}]
        result = merge_consecutive_messages(messages)
        assert result == messages

    def test_consecutive_same_role(self):
        """Test merging consecutive messages with the same role."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "World"},
        ]
        result = merge_consecutive_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "Hello" in result[0]["content"]
        assert "World" in result[0]["content"]

    def test_alternating_roles(self):
        """Test merging messages with alternating roles."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]
        result = merge_consecutive_messages(messages)
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "user"

    def test_mixed_content_types(self):
        """Test merging messages with mixed content types."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": [{"type": "text", "text": "World"}]},
        ]
        result = merge_consecutive_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 2

    def test_multiple_consecutive(self):
        """Test merging multiple consecutive messages."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "World"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "assistant", "content": "How can I help?"},
            {"role": "user", "content": "I need help"},
        ]
        result = merge_consecutive_messages(messages)
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert "Hello" in result[0]["content"]
        assert "World" in result[0]["content"]
        assert result[1]["role"] == "assistant"
        assert "Hi there" in result[1]["content"]
        assert "How can I help?" in result[1]["content"]
        assert result[2]["role"] == "user"
        assert "I need help" in result[2]["content"]

    def test_consecutive_none_content(self):
        """Consecutive same-role messages with content=None (e.g. tool-call-only
        assistant turns) must not crash the merge."""
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1"}],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_2"}],
            },
        ]
        result = merge_consecutive_messages(messages)
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[1]["tool_calls"] == [{"id": "call_1"}]
        assert result[2]["tool_calls"] == [{"id": "call_2"}]

    def test_tool_call_fields_are_preserved(self):
        """Tool-call history must keep the fields that link calls to results."""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": "result",
                "tool_call_id": "call_1",
            },
        ]

        result = merge_consecutive_messages(messages)

        assert result[0]["tool_calls"] == messages[0]["tool_calls"]
        assert result[1]["tool_call_id"] == "call_1"

    def test_merging_list_content_does_not_mutate_input(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "first"}]},
            {"role": "user", "content": [{"type": "text", "text": "second"}]},
        ]
        original_content = list(messages[0]["content"])

        result = merge_consecutive_messages(messages)

        assert messages[0]["content"] == original_content
        assert result[0]["content"] == [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]

    def test_large_string_run_preserves_order_and_input(self):
        contents = [f"fragment {index}" for index in range(2000)]
        messages = [{"role": "user", "content": content} for content in contents]

        assert merge_consecutive_messages(messages) == [
            {"role": "user", "content": "\n\n".join(contents)}
        ]
        assert [message["content"] for message in messages] == contents

    def test_string_runs_stop_at_role_and_protocol_boundaries(self):
        messages = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "third"},
            {"role": "assistant", "content": "fourth"},
            {"role": "assistant", "content": "named", "name": "helper"},
            {"role": "assistant", "content": "fifth"},
            {"role": "assistant", "content": "sixth"},
        ]

        assert merge_consecutive_messages(messages) == [
            {"role": "user", "content": "first\n\nsecond"},
            {"role": "assistant", "content": "third\n\nfourth"},
            {"role": "assistant", "content": "named", "name": "helper"},
            {"role": "assistant", "content": "fifth\n\nsixth"},
        ]


class TestGetMessageContent:
    """Test the get_message_content function."""

    def test_string_content(self):
        """Test getting content from a message with string content."""
        message = cast(ChatCompletionMessageParam, {"role": "user", "content": "Hello"})
        result = get_message_content(message)
        assert result == ["Hello"]

    def test_list_content(self):
        """Test getting content from a message with list content."""
        message = cast(
            ChatCompletionMessageParam,
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        )
        result = get_message_content(message)
        assert result == [{"type": "text", "text": "Hello"}]

    def test_empty_content(self):
        """Test getting content from a message with empty content."""
        message = cast(ChatCompletionMessageParam, {"role": "user", "content": ""})
        result = get_message_content(message)
        assert result == [""]

    def test_none_content(self):
        """Test getting content from a message with None content."""
        message = cast(ChatCompletionMessageParam, {"role": "user", "content": None})
        result = get_message_content(message)
        assert result == [""]

    def test_missing_content(self):
        """Test getting content from a message with missing content."""
        message = cast(ChatCompletionMessageParam, {"role": "user"})
        result = get_message_content(message)
        assert result == [""]

    def test_empty_message(self):
        """Test getting content from an empty message."""
        message = cast(ChatCompletionMessageParam, {})
        result = get_message_content(message)
        assert result == [""]


class TestTransformToGeminiPrompt:
    """Test the transform_to_gemini_prompt function."""

    def test_empty_messages(self):
        """Test transforming empty messages."""
        result = transform_to_gemini_prompt([])
        assert result == []

    def test_user_message(self):
        """Test transforming a user message."""
        messages = [{"role": "user", "content": "Hello"}]
        result = transform_to_gemini_prompt(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["parts"] == ["Hello"]

    def test_assistant_message(self):
        """Test transforming an assistant message."""
        messages = [{"role": "assistant", "content": "Hello"}]
        result = transform_to_gemini_prompt(messages)
        assert len(result) == 1
        assert result[0]["role"] == "model"
        assert result[0]["parts"] == ["Hello"]

    def test_system_message(self):
        """Test transforming a system message."""
        messages = [{"role": "system", "content": "You are an AI assistant"}]
        result = transform_to_gemini_prompt(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "*You are an AI assistant*" in result[0]["parts"][0]

    def test_full_conversation(self):
        """Test transforming a full conversation."""
        messages = [
            {"role": "system", "content": "You are an AI assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]
        result = transform_to_gemini_prompt(messages)
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert "*You are an AI assistant*" in result[0]["parts"][0]
        assert "Hello" in result[0]["parts"][1]
        assert result[1]["role"] == "model"
        assert result[1]["parts"] == ["Hi there"]
        assert result[2]["role"] == "user"
        assert result[2]["parts"] == ["How are you?"]

    def test_multiple_system_messages(self):
        """Test transforming multiple system messages."""
        messages = [
            {"role": "system", "content": "You are an AI assistant"},
            {"role": "system", "content": "Be helpful and concise"},
            {"role": "user", "content": "Hello"},
        ]
        result = transform_to_gemini_prompt(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert any("You are an AI assistant" in part for part in result[0]["parts"])
        assert any("Be helpful and concise" in part for part in result[0]["parts"])
        assert any("Hello" in part for part in result[0]["parts"])


class TestUpdateGeminiKwargs:
    """Test the update_gemini_kwargs function."""

    def test_transform_messages(self):
        """Test transforming messages to Gemini format."""
        kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
        result = update_gemini_kwargs(kwargs)
        assert "contents" in result
        assert "messages" not in result
        assert len(result["contents"]) == 1
        assert result["contents"][0]["role"] == "user"

    def test_generation_config(self):
        """Test updating generation config."""
        kwargs = {
            "messages": [{"role": "user", "content": "Hello"}],
            "generation_config": {
                "max_tokens": 100,
                "temperature": 0.7,
                "n": 3,
                "top_p": 0.9,
                "stop": ["END"],
            },
        }
        result = update_gemini_kwargs(kwargs)
        assert "generation_config" in result
        assert "max_output_tokens" in result["generation_config"]
        assert "candidate_count" in result["generation_config"]
        assert "stop_sequences" in result["generation_config"]
        assert "max_tokens" not in result["generation_config"]
        assert "n" not in result["generation_config"]
        assert "stop" not in result["generation_config"]

    def test_safety_settings(self):
        """Test setting safety settings."""
        kwargs = {
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = update_gemini_kwargs(kwargs)
        assert "safety_settings" in result
        assert len(result["safety_settings"]) >= 3  # At least 3 safety settings

    def test_existing_safety_settings(self):
        """Test respecting existing safety settings."""
        from google.genai.types import HarmCategory, HarmBlockThreshold

        kwargs = {
            "messages": [{"role": "user", "content": "Hello"}],
            "safety_settings": {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
            },
        }
        result = update_gemini_kwargs(kwargs)
        assert (
            result["safety_settings"][HarmCategory.HARM_CATEGORY_HATE_SPEECH]
            == HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        )


class TestSystemMessages:
    """Test the system message utility functions."""

    def test_combine_system_messages_strings(self):
        """Test combining two string system messages."""
        existing = "You are an AI assistant"
        new = "Be helpful"
        result = combine_system_messages(existing, new)
        assert result == "You are an AI assistant\n\nBe helpful"

    def test_combine_system_messages_lists(self):
        """Test combining two list system messages."""
        existing = [SystemMessage(type="text", text="You are an AI assistant")]
        new = [SystemMessage(type="text", text="Be helpful")]
        result = combine_system_messages(existing, new)
        assert len(result) == 2
        assert result[0]["text"] == "You are an AI assistant"
        assert result[1]["text"] == "Be helpful"

    def test_combine_system_messages_mixed(self):
        """Test combining mixed system message types."""
        existing = "You are an AI assistant"
        new = [SystemMessage(type="text", text="Be helpful")]
        result = combine_system_messages(existing, new)
        assert len(result) == 2
        assert result[0]["text"] == "You are an AI assistant"
        assert result[1]["text"] == "Be helpful"

    def test_combine_system_messages_none(self):
        """Test combining None with a system message."""
        existing = None
        new = "Be helpful"
        result = combine_system_messages(existing, new)
        assert result == "Be helpful"

    def test_extract_system_messages_empty(self):
        """Test extracting system messages from an empty list."""
        messages = []
        result = extract_system_messages(messages)
        assert result == []

    def test_extract_system_messages_no_system(self):
        """Test extracting system messages when there are none."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = extract_system_messages(messages)
        assert result == []

    def test_extract_system_messages_string(self):
        """Test extracting string system messages."""
        messages = [
            {"role": "system", "content": "You are an AI assistant"},
            {"role": "user", "content": "Hello"},
        ]
        result = extract_system_messages(messages)
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "You are an AI assistant"

    def test_extract_system_messages_list(self):
        """Test extracting list system messages."""
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an AI assistant"}],
            },
            {"role": "user", "content": "Hello"},
        ]
        result = extract_system_messages(messages)
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "You are an AI assistant"

    def test_extract_system_messages_multiple(self):
        """Test extracting multiple system messages."""
        messages = [
            {"role": "system", "content": "You are an AI assistant"},
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
        ]
        result = extract_system_messages(messages)
        assert len(result) == 2
        assert result[0]["text"] == "You are an AI assistant"
        assert result[1]["text"] == "Be helpful"
