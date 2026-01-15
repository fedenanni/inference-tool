"""Tokenizer for text encoding and decoding using SentencePiece."""

from pathlib import Path
from huggingface_hub import hf_hub_download
import sentencepiece as spm


class Tokenizer:
    """A tokenizer that uses SentencePiece for encoding and decoding text."""

    def __init__(self, model_name: str):
        """Initialize the tokenizer by downloading the model from Hugging Face.

        Args:
            model_name: The Hugging Face model name (e.g., "google/gemma-3-270m-it")
        """
        # Download the tokenizer model file from Hugging Face
        tokenizer_path = hf_hub_download(
            repo_id=model_name,
            filename="tokenizer.model",
        )

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(tokenizer_path)

        # Expose special tokens and vocabulary info
        self.bos_token_id: int = self.sp.bos_id()
        self.eos_token_id: int = self.sp.eos_id()
        self.vocab_size: int = self.sp.GetPieceSize()

    def encode(self, text: str) -> list[int]:
        """Encode text into a list of token IDs.

        Args:
            text: The input text to encode.

        Returns:
            A list of integer token IDs.
        """
        if text == "":
            return []
        return self.sp.EncodeAsIds(text)

    def decode(self, token_ids: list[int]) -> str:
        """Decode a list of token IDs back into text.

        Args:
            token_ids: A list of integer token IDs.

        Returns:
            The decoded text string.
        """
        if not token_ids:
            return ""
        return self.sp.DecodeIds(token_ids)
