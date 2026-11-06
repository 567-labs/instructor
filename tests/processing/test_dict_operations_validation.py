"""Tests to validate that the optimized dictionary operations provide the same results as before."""

from instructor.core.retry import extract_messages
from instructor.utils import (
    combine_system_messages,
    extract_system_messages,
    update_gemini_kwargs,
    SystemMessage,
)


class TestDictOperationsValidation:
    """Test suite for validating dictionary operations behavior."""

    def test_extract_messages_validation(self):
        """Validate extract_messages returns the same results after optimization."""
        # Test with messages key
        sample_messages = [{"role": "user", "content": "Hello"}]
        kwargs = {"messages": sample_messages}
        result = extract_messages(kwargs)
        assert result == sample_messages

        # Test with contents key
        sample_contents = [{"role": "user", "parts": ["Hello"]}]
        kwargs = {"contents": sample_contents}
        result = extract_messages(kwargs)
        assert result == sample_contents

        # Test with chat_history key
        sample_chat_history = [{"role": "user", "message": "Hello"}]
        kwargs = {"chat_history": sample_chat_history}
        result = extract_messages(kwargs)
        assert result == sample_chat_history

        # Test with empty dict
        kwargs = {}
        result = extract_messages(kwargs)
        assert result == []

        # Test with mixed keys (should prioritize messages)
        kwargs = {
            "messages": sample_messages,
            "contents": sample_contents,
            "chat_history": sample_chat_history,
        }
        result = extract_messages(kwargs)
        assert result == sample_messages

    def test_combine_system_messages_validation(self):
        """Validate combine_system_messages returns the same results after optimization."""
        # Test with both strings
        existing = "You are a helpful assistant."
        new = "You should be concise."
        expected = "You are a helpful assistant.\n\nYou should be concise."
        result = combine_system_messages(existing, new)
        assert result == expected

        # Test with both lists
        existing_list = [
            SystemMessage(type="text", text="You are a helpful assistant.")
        ]
        new_list = [SystemMessage(type="text", text="You should be concise.")]
        result = combine_system_messages(existing_list, new_list)
        assert len(result) == 2
        assert result[0]["text"] == "You are a helpful assistant."
        assert result[1]["text"] == "You should be concise."

        # Test with existing string, new list
        result = combine_system_messages(existing, new_list)
        assert len(result) == 2
        assert result[0]["text"] == "You are a helpful assistant."
        assert result[1]["text"] == "You should be concise."

        # Test with existing list, new string
        result = combine_system_messages(existing_list, new)
        assert len(result) == 2
        assert result[0]["text"] == "You are a helpful assistant."
        assert result[1]["text"] == "You should be concise."

        # Test with None existing
        result = combine_system_messages(None, new)
        assert result == new

        result = combine_system_messages(None, new_list)
        assert result == new_list

    def test_extract_system_messages_validation(self):
        """Validate extract_system_messages returns the same results after optimization."""
        # Test with system messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        result = extract_system_messages(messages)
        assert len(result) == 1
        assert result[0]["text"] == "You are a helpful assistant."

        # Test with multiple system messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "system", "content": "You should be concise."},
            {"role": "user", "content": "Hello"},
        ]
        result = extract_system_messages(messages)
        assert len(result) == 2
        assert result[0]["text"] == "You are a helpful assistant."
        assert result[1]["text"] == "You should be concise."

        # Test with no system messages
        messages = [{"role": "user", "content": "Hello"}]
        result = extract_system_messages(messages)
        assert result == []

        # Test with empty messages
        result = extract_system_messages([])
        assert result == []

        # Test with system message and list content
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {"role": "user", "content": "Hello"},
        ]
        result = extract_system_messages(messages)
        assert len(result) == 1
        assert result[0]["text"] == "You are a helpful assistant."

    def test_update_gemini_kwargs_validation(self):
        """Validate update_gemini_kwargs returns the same results after optimization."""
        # Test with complete kwargs
        kwargs = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ],
            "max_tokens": 1000,
            "temperature": 0.7,
            "generation_config": {
                "max_tokens": 2000,
                "temperature": 0.5,
                "top_p": 0.9,
                "n": 1,
                "stop": ["###"],
            },
        }

        result = update_gemini_kwargs(kwargs)

        # Check that it contains contents transformed from messages
        assert "contents" in result
        assert (
            len(result["contents"]) == 1
        )  # System messages are merged into first user message

        # Check that generation_config was updated properly
        assert "max_output_tokens" in result["generation_config"]
        assert result["generation_config"]["max_output_tokens"] == 2000
        assert "candidate_count" in result["generation_config"]
        assert result["generation_config"]["candidate_count"] == 1
        assert "stop_sequences" in result["generation_config"]
        assert result["generation_config"]["stop_sequences"] == ["###"]

        # Check that safety settings were added
        assert "safety_settings" in result

        # Ensure the original kwargs wasn't modified
        assert "contents" not in kwargs
        assert "messages" in kwargs
