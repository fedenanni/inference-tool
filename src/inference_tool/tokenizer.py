"""Tokenizer for text encoding and decoding using BPE (Byte Pair Encoding).

This is a simple, readable implementation that loads tokenizer data from
Hugging Face and performs BPE tokenization without external tokenizer libraries.
"""

import json
from huggingface_hub import hf_hub_download


class Tokenizer:
    """A BPE tokenizer that loads vocabulary and merges from Hugging Face."""

    def __init__(self, model_name: str):
        """Initialize the tokenizer by loading vocabulary and merges from Hugging Face.

        Args:
            model_name: The Hugging Face model name (e.g., "google/gemma-3-270m-it")
        """
        # Download and load the tokenizer.json file
        tokenizer_path = hf_hub_download(repo_id=model_name, filename="tokenizer.json")
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            tokenizer_data = json.load(f)

        model_data = tokenizer_data["model"]

        # Load vocabulary: maps token string -> token id
        self._vocab: dict[str, int] = model_data["vocab"]
        # Reverse vocabulary: maps token id -> token string
        self._id_to_token: dict[int, str] = {v: k for k, v in self._vocab.items()}

        # Load merge rules: list of (token1, token2) pairs in priority order
        # Earlier merges have higher priority
        self._merges: dict[tuple[str, str], int] = {}
        for i, merge in enumerate(model_data["merges"]):
            self._merges[(merge[0], merge[1])] = i

        # Load added tokens (special tokens that should be matched as whole units)
        # These are tokens like <start_of_turn>, <end_of_turn>, etc.
        self._added_tokens: dict[str, int] = {}
        for token_info in tokenizer_data.get("added_tokens", []):
            content = token_info["content"]
            token_id = token_info["id"]
            self._added_tokens[content] = token_id

        # Special token IDs
        self.bos_token_id: int = self._vocab.get("<bos>", 2)
        self.eos_token_id: int = self._vocab.get("<eos>", 1)
        self.vocab_size: int = len(self._vocab)

        # Build byte fallback tokens for handling unknown characters
        self._byte_tokens: dict[int, str] = {}
        for byte_val in range(256):
            token = f"<0x{byte_val:02X}>"
            if token in self._vocab:
                self._byte_tokens[byte_val] = token

        # Sort added tokens by length (longest first) for greedy matching
        self._sorted_added_tokens = sorted(
            self._added_tokens.keys(), key=len, reverse=True
        )

    def _normalize(self, text: str) -> str:
        """Normalize text by replacing spaces with the special space character ▁."""
        return text.replace(" ", "▁")

    def _denormalize(self, text: str) -> str:
        """Denormalize text by replacing the special space character ▁ with spaces."""
        return text.replace("▁", " ")

    def _split_by_added_tokens(self, text: str) -> list[tuple[str, bool]]:
        """Split text into segments, identifying added tokens.

        Returns a list of (segment, is_added_token) tuples.
        Added tokens should be encoded as single tokens, not split by BPE.
        """
        result = []
        remaining = text

        while remaining:
            # Try to match an added token at the start
            matched = False
            for token in self._sorted_added_tokens:
                if remaining.startswith(token):
                    result.append((token, True))
                    remaining = remaining[len(token) :]
                    matched = True
                    break

            if not matched:
                # Find the next added token position
                next_pos = len(remaining)
                for token in self._sorted_added_tokens:
                    pos = remaining.find(token)
                    if pos != -1 and pos < next_pos:
                        next_pos = pos

                # Add the regular text segment
                result.append((remaining[:next_pos], False))
                remaining = remaining[next_pos:]

        return result

    def _text_to_tokens(self, text: str) -> list[str]:
        """Convert text to initial tokens (characters or byte fallbacks).

        Each character becomes a token. If a character isn't in the vocabulary,
        we fall back to byte tokens.
        """
        tokens = []
        for char in text:
            if char in self._vocab:
                tokens.append(char)
            else:
                # Byte fallback: encode character as UTF-8 bytes
                for byte in char.encode("utf-8"):
                    tokens.append(self._byte_tokens.get(byte, "<unk>"))
        return tokens

    def _merge_tokens(self, tokens: list[str]) -> list[str]:
        """Apply BPE merges to a list of tokens until no more merges are possible.

        This repeatedly finds the highest-priority merge pair and applies it.
        """
        while len(tokens) >= 2:
            # Find the merge with the lowest index (highest priority)
            best_merge = None
            best_priority = float("inf")
            best_idx = -1

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self._merges:
                    priority = self._merges[pair]
                    if priority < best_priority:
                        best_priority = priority
                        best_merge = pair
                        best_idx = i

            if best_merge is None:
                break  # No more merges possible

            # Apply the merge: combine the two tokens
            merged_token = best_merge[0] + best_merge[1]
            tokens = tokens[:best_idx] + [merged_token] + tokens[best_idx + 2 :]

        return tokens

    def encode(self, text: str) -> list[int]:
        """Encode text into a list of token IDs.

        Args:
            text: The input text to encode.

        Returns:
            A list of integer token IDs.
        """
        if text == "":
            return []

        result = []

        # Step 1: Split by added tokens (special tokens that shouldn't be split)
        segments = self._split_by_added_tokens(text)

        for segment, is_added_token in segments:
            if not segment:
                continue

            if is_added_token:
                # Added tokens are encoded directly to their ID
                result.append(self._added_tokens[segment])
            else:
                # Regular text: normalize, tokenize, and apply BPE
                normalized = self._normalize(segment)
                tokens = self._text_to_tokens(normalized)
                tokens = self._merge_tokens(tokens)
                result.extend(
                    self._vocab.get(token, self._vocab.get("<unk>", 3))
                    for token in tokens
                )

        return result

    def decode(self, token_ids: list[int]) -> str:
        """Decode a list of token IDs back into text.

        Args:
            token_ids: A list of integer token IDs.

        Returns:
            The decoded text string.
        """
        if not token_ids:
            return ""

        # Convert IDs to tokens
        tokens = [self._id_to_token.get(tid, "<unk>") for tid in token_ids]

        # Join tokens and handle byte tokens
        text = ""
        byte_buffer = []

        for token in tokens:
            if token.startswith("<0x") and token.endswith(">"):
                # This is a byte token - accumulate bytes
                try:
                    byte_val = int(token[3:-1], 16)
                    byte_buffer.append(byte_val)
                except ValueError:
                    # Flush any pending bytes and add the token as-is
                    if byte_buffer:
                        text += bytes(byte_buffer).decode("utf-8", errors="replace")
                        byte_buffer = []
                    text += token
            else:
                # Flush any pending bytes
                if byte_buffer:
                    text += bytes(byte_buffer).decode("utf-8", errors="replace")
                    byte_buffer = []
                # Skip special tokens like <pad>, <eos>, <bos>, <unk>
                if not (token.startswith("<") and token.endswith(">")):
                    text += token

        # Flush any remaining bytes
        if byte_buffer:
            text += bytes(byte_buffer).decode("utf-8", errors="replace")

        # Denormalize (replace ▁ with spaces)
        return self._denormalize(text)
