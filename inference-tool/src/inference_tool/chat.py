"""Chat template formatting for Gemma models.

This module handles conversion between chat message formats:
- Input: List of message dicts with 'role' and 'content' keys
- Output: Gemma-formatted string with special tokens

Gemma-3 chat format:
    <start_of_turn>user
    {user_message}<end_of_turn>
    <start_of_turn>model
    {model_response}<end_of_turn>

Usage:
    from chat import format_messages, parse_response

    messages = [
        {"role": "user", "content": "Who are you?"},
    ]

    formatted = format_messages(messages)
    # Result: "<start_of_turn>user\nWho are you?<end_of_turn>\n<start_of_turn>model\n"

    # After generation, parse the response:
    response = parse_response(generated_text)
"""

from typing import TypedDict


# =============================================================================
# Special Tokens
# =============================================================================

START_OF_TURN = "<start_of_turn>"
END_OF_TURN = "<end_of_turn>"

# Role mapping from standard names to Gemma's format
ROLE_MAP = {
    "user": "user",
    "assistant": "model",
    "model": "model",
    "system": "system",
}


# =============================================================================
# Type Definitions
# =============================================================================


class Message(TypedDict):
    """A chat message with role and content."""

    role: str
    content: str


# =============================================================================
# Formatting Functions
# =============================================================================


def format_messages(messages: list[Message]) -> str:
    """Format a list of chat messages into Gemma's chat template format.

    Converts messages into the format expected by Gemma:
        <start_of_turn>role
        content<end_of_turn>
        ...

    The output always ends with "<start_of_turn>model\n" to prompt
    the model to generate a response.

    System messages are handled by prepending them to the first user message.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
                 Roles can be 'user', 'assistant', 'model', or 'system'.

    Returns:
        Formatted string ready for tokenization.

    Example:
        >>> messages = [{"role": "user", "content": "Hello!"}]
        >>> format_messages(messages)
        '<start_of_turn>user\\nHello!<end_of_turn>\\n<start_of_turn>model\\n'
    """
    if not messages:
        return ""

    # Separate system message if present
    system_content = None
    regular_messages = []

    for msg in messages:
        if msg["role"] == "system":
            # Collect system message (take the last one if multiple)
            system_content = msg["content"]
        else:
            regular_messages.append(msg)

    # Build formatted string
    parts = []

    for i, msg in enumerate(regular_messages):
        role = ROLE_MAP.get(msg["role"], msg["role"])
        content = msg["content"]

        # Prepend system message to first user message
        if i == 0 and system_content and role == "user":
            content = f"{system_content}\n\n{content}"

        # Format this turn
        turn = f"{START_OF_TURN}{role}\n{content}{END_OF_TURN}\n"
        parts.append(turn)

    # Join all parts
    result = "".join(parts)

    # If the last message is not from the model, add model turn prompt
    if regular_messages:
        last_role = ROLE_MAP.get(
            regular_messages[-1]["role"], regular_messages[-1]["role"]
        )
        if last_role != "model":
            result += f"{START_OF_TURN}model\n"
    else:
        # If only system message was provided, start with model turn
        if system_content:
            result = f"{START_OF_TURN}user\n{system_content}{END_OF_TURN}\n{START_OF_TURN}model\n"

    return result


def parse_response(generated_text: str) -> str:
    """Extract the model's response from generated text.

    Parses the generated text to extract the last model response,
    removing the special tokens and formatting.

    Args:
        generated_text: Full text output from the model including
                       the prompt and special tokens.

    Returns:
        The extracted response text without special tokens.

    Example:
        >>> text = "<start_of_turn>user\\nHi<end_of_turn>\\n<start_of_turn>model\\nHello!<end_of_turn>"
        >>> parse_response(text)
        'Hello!'
    """
    # Find the last model turn
    model_marker = f"{START_OF_TURN}model\n"

    # Find all model turn starts
    last_model_start = generated_text.rfind(model_marker)

    if last_model_start == -1:
        # No model marker found, return empty or the whole text
        return generated_text.strip()

    # Extract text after the model marker
    response_start = last_model_start + len(model_marker)
    response_text = generated_text[response_start:]

    # Remove end_of_turn marker if present
    end_marker_pos = response_text.find(END_OF_TURN)
    if end_marker_pos != -1:
        response_text = response_text[:end_marker_pos]

    return response_text.strip()


def format_single_turn(user_message: str, system_prompt: str | None = None) -> str:
    """Convenience function for single-turn conversations.

    Args:
        user_message: The user's message content.
        system_prompt: Optional system prompt to prepend.

    Returns:
        Formatted string ready for tokenization.
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": user_message})

    return format_messages(messages)
