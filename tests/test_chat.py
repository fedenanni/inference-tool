"""Tests for the chat template formatting module.

These tests verify the conversion between chat message formats:
- Input: List of message dicts with 'role' and 'content' keys
- Output: Gemma-formatted string with special tokens

Gemma-3 chat format:
    <start_of_turn>user
    {user_message}<end_of_turn>
    <start_of_turn>model
    {model_response}<end_of_turn>

Special tokens:
    - <start_of_turn>: Marks the beginning of a turn
    - <end_of_turn>: Marks the end of a turn
    - <bos>: Beginning of sequence (optional, usually added by tokenizer)
"""

import pytest


# =============================================================================
# Chat Template Tests
# =============================================================================


class TestFormatMessages:
    """Tests for converting message lists to Gemma format.

    The format_messages function takes a list of messages and returns
    a string formatted according to Gemma's chat template.
    """

    # Test single user message formatting
    def test_single_user_message(self):
        from inference_tool.chat import format_messages

        messages = [{"role": "user", "content": "Hello!"}]

        result = format_messages(messages)

        expected = "<start_of_turn>user\nHello!<end_of_turn>\n<start_of_turn>model\n"
        assert result == expected

    # Test multi-turn conversation formatting
    def test_multi_turn_conversation(self):
        from inference_tool.chat import format_messages

        messages = [
            {"role": "user", "content": "Who are you?"},
            {"role": "assistant", "content": "I am Gemma."},
            {"role": "user", "content": "Nice to meet you!"},
        ]

        result = format_messages(messages)

        expected = (
            "<start_of_turn>user\n"
            "Who are you?<end_of_turn>\n"
            "<start_of_turn>model\n"
            "I am Gemma.<end_of_turn>\n"
            "<start_of_turn>user\n"
            "Nice to meet you!<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        assert result == expected

    # Test that 'assistant' role maps to 'model' in output
    def test_assistant_role_becomes_model(self):
        from inference_tool.chat import format_messages

        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]

        result = format_messages(messages)

        assert "<start_of_turn>model\nHello!" in result
        assert "assistant" not in result

    # Test system message is prepended to first user message
    def test_system_message_handling(self):
        from inference_tool.chat import format_messages

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]

        result = format_messages(messages)

        # System message should be incorporated (Gemma typically prepends to user turn)
        assert "You are a helpful assistant." in result
        assert "Hello!" in result

    # Test empty messages list
    def test_empty_messages(self):
        from inference_tool.chat import format_messages

        messages = []

        result = format_messages(messages)

        assert result == ""

    # Test message with newlines in content
    def test_message_with_newlines(self):
        from inference_tool.chat import format_messages

        messages = [{"role": "user", "content": "Line 1\nLine 2\nLine 3"}]

        result = format_messages(messages)

        assert "Line 1\nLine 2\nLine 3" in result

    # Test message with special characters
    def test_message_with_special_characters(self):
        from inference_tool.chat import format_messages

        messages = [{"role": "user", "content": "What is 2 + 2? <test> & 'quotes'"}]

        result = format_messages(messages)

        assert "What is 2 + 2? <test> & 'quotes'" in result

    # Test ends with model turn prompt for generation
    def test_ends_with_model_turn(self):
        from inference_tool.chat import format_messages

        messages = [{"role": "user", "content": "Hello!"}]

        result = format_messages(messages)

        assert result.endswith(
            "<start_of_turn>model\n"
        ), "Should end with model turn prompt for generation"

    # Test only assistant message (edge case)
    def test_only_assistant_message(self):
        from inference_tool.chat import format_messages

        messages = [{"role": "assistant", "content": "I'm here to help."}]

        result = format_messages(messages)

        # Should still format correctly
        assert "<start_of_turn>model\nI'm here to help." in result


class TestParseResponse:
    """Tests for parsing model response from generated text.

    The parse_response function extracts the assistant's response
    from the full generated text, removing special tokens.
    """

    # Test extracting response from generated text
    def test_extracts_model_response(self):
        from inference_tool.chat import parse_response

        generated = (
            "<start_of_turn>user\n"
            "Hello!<end_of_turn>\n"
            "<start_of_turn>model\n"
            "Hi there! How can I help you today?<end_of_turn>\n"
        )

        result = parse_response(generated)

        assert result == "Hi there! How can I help you today?"

    # Test handling text without end_of_turn (still generating)
    def test_handles_incomplete_response(self):
        from inference_tool.chat import parse_response

        generated = (
            "<start_of_turn>user\n"
            "Hello!<end_of_turn>\n"
            "<start_of_turn>model\n"
            "Hi there! I am"
        )

        result = parse_response(generated)

        assert result == "Hi there! I am"

    # Test extracting last model response from multi-turn
    def test_extracts_last_model_response(self):
        from inference_tool.chat import parse_response

        generated = (
            "<start_of_turn>user\n"
            "Who are you?<end_of_turn>\n"
            "<start_of_turn>model\n"
            "I am Gemma.<end_of_turn>\n"
            "<start_of_turn>user\n"
            "What can you do?<end_of_turn>\n"
            "<start_of_turn>model\n"
            "I can help with many tasks!<end_of_turn>\n"
        )

        result = parse_response(generated)

        assert result == "I can help with many tasks!"


class TestChatTemplateTokens:
    """Tests for special token constants used in chat formatting."""

    def test_special_tokens_defined(self):
        from inference_tool.chat import START_OF_TURN, END_OF_TURN

        assert START_OF_TURN == "<start_of_turn>"
        assert END_OF_TURN == "<end_of_turn>"

    def test_role_mapping(self):
        from inference_tool.chat import ROLE_MAP

        assert ROLE_MAP["user"] == "user"
        assert ROLE_MAP["assistant"] == "model"
        assert ROLE_MAP.get("system") is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestChatFormatIntegration:
    """Integration tests for chat formatting with tokenizer."""

    # Test that formatted messages tokenize correctly
    def test_formatted_messages_tokenize(self):
        from inference_tool.chat import format_messages

        messages = [{"role": "user", "content": "Hello!"}]
        formatted = format_messages(messages)

        # Should not raise any errors
        assert isinstance(formatted, str)
        assert len(formatted) > 0

    # Test round-trip: format -> tokenize -> decode preserves content
    @pytest.mark.slow
    def test_format_tokenize_decode_roundtrip(self):
        from inference_tool.chat import format_messages
        from inference_tool.tokenizer import Tokenizer

        tokenizer = Tokenizer("google/gemma-3-270m-it")

        messages = [{"role": "user", "content": "What is Python?"}]
        formatted = format_messages(messages)

        # Tokenize and decode
        tokens = tokenizer.encode(formatted)
        decoded = tokenizer.decode(tokens)

        # Content should be preserved
        assert "What is Python?" in decoded
