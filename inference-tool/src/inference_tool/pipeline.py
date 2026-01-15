"""High-level pipeline for text generation with Gemma models.

This module provides a HuggingFace-compatible Pipeline interface for
easy text generation with Gemma models.

Usage:
    from pipeline import Pipeline

    # Initialize pipeline
    pipe = Pipeline("google/gemma-3-270m-it")

    # Generate response (HuggingFace-style)
    response = pipe([{"role": "user", "content": "Who are you?"}])
    print(response)

    # With custom parameters
    response = pipe(
        [{"role": "user", "content": "Write a poem"}],
        max_new_tokens=200,
        temperature=0.8,
        top_p=0.9,
    )

    # Streaming (yields tokens as generated)
    for token in pipe.stream([{"role": "user", "content": "Hello!"}]):
        print(token, end="", flush=True)

The Pipeline class handles:
- Model and tokenizer loading
- Chat template formatting
- Text generation with configurable parameters
- Response parsing and formatting
"""

from typing import Iterator, Optional

from .model import GemmaModel
from .tokenizer import Tokenizer
from .chat import format_messages, parse_response
from .generate import generate


class Pipeline:
    """High-level pipeline for text generation with Gemma models.

    This class provides a simple interface matching HuggingFace's pipeline
    for text generation tasks.

    Attributes:
        model: The loaded GemmaModel instance
        tokenizer: The loaded Tokenizer instance
        default_max_new_tokens: Default maximum tokens to generate
        default_temperature: Default sampling temperature
        default_top_k: Default top-k filtering value
        default_top_p: Default top-p (nucleus) filtering value
        default_repetition_penalty: Default repetition penalty
    """

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
    ):
        """Initialize the pipeline with a model.

        Args:
            model_name: HuggingFace model name (e.g., "google/gemma-3-270m-it")
            max_new_tokens: Default maximum number of tokens to generate
            temperature: Default sampling temperature (0 = greedy)
            top_k: Default top-k filtering (None = disabled)
            top_p: Default top-p filtering (None = disabled)
            repetition_penalty: Default penalty for token repetition
        """
        # Load model and tokenizer
        self.model = GemmaModel(model_name)
        self.tokenizer = Tokenizer(model_name)

        # Store default generation parameters
        self.default_max_new_tokens = max_new_tokens
        self.default_temperature = temperature
        self.default_top_k = top_k
        self.default_top_p = top_p
        self.default_repetition_penalty = repetition_penalty

    def __call__(
        self,
        messages: list[dict],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> str:
        """Generate a response for the given messages.

        This is the main entry point for generation, matching the
        HuggingFace pipeline interface.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                     Example: [{"role": "user", "content": "Hello!"}]
            max_new_tokens: Maximum tokens to generate (overrides default)
            temperature: Sampling temperature (overrides default)
            top_k: Top-k filtering (overrides default)
            top_p: Top-p filtering (overrides default)
            repetition_penalty: Repetition penalty (overrides default)

        Returns:
            The generated response text (assistant's reply only).

        Raises:
            ValueError: If messages list is empty.
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")

        result = self.generate(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        return result["response"]

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> dict:
        """Generate a response with detailed output.

        This method provides more detailed output than __call__,
        including token counts and the full generated text.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k filtering parameter.
            top_p: Top-p filtering parameter.
            repetition_penalty: Repetition penalty.

        Returns:
            Dictionary containing:
            - response: The parsed assistant response
            - full_text: The complete generated text including prompt
            - prompt_tokens: Number of tokens in the prompt
            - generated_tokens: Number of newly generated tokens
        """
        # Use defaults if not specified
        max_new_tokens = (
            max_new_tokens
            if max_new_tokens is not None
            else self.default_max_new_tokens
        )
        temperature = (
            temperature if temperature is not None else self.default_temperature
        )
        top_k = top_k if top_k is not None else self.default_top_k
        top_p = top_p if top_p is not None else self.default_top_p
        repetition_penalty = (
            repetition_penalty
            if repetition_penalty is not None
            else self.default_repetition_penalty
        )

        # Format messages using chat template
        formatted_prompt = format_messages(messages)

        # Tokenize the formatted prompt
        prompt_tokens = self.tokenizer.encode(formatted_prompt)
        prompt_length = len(prompt_tokens)

        # Generate tokens
        output_tokens = generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # Decode the full output
        full_text = self.tokenizer.decode(output_tokens)

        # Parse to extract just the response
        response = parse_response(full_text)

        return {
            "response": response,
            "full_text": full_text,
            "prompt_tokens": prompt_length,
            "generated_tokens": len(output_tokens) - prompt_length,
        }

    def stream(
        self,
        messages: list[dict],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> Iterator[str]:
        """Generate a response with streaming output.

        Yields tokens as they are generated, allowing for real-time
        display of the response.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k filtering parameter.
            top_p: Top-p filtering parameter.
            repetition_penalty: Repetition penalty.

        Yields:
            Generated text chunks (typically single tokens decoded).
        """
        import numpy as np
        from generate import sample_token, apply_repetition_penalty

        # Use defaults if not specified
        max_new_tokens = (
            max_new_tokens
            if max_new_tokens is not None
            else self.default_max_new_tokens
        )
        temperature = (
            temperature if temperature is not None else self.default_temperature
        )
        top_k = top_k if top_k is not None else self.default_top_k
        top_p = top_p if top_p is not None else self.default_top_p
        repetition_penalty = (
            repetition_penalty
            if repetition_penalty is not None
            else self.default_repetition_penalty
        )

        # Format messages using chat template
        formatted_prompt = format_messages(messages)

        # Tokenize the formatted prompt
        prompt_tokens = self.tokenizer.encode(formatted_prompt)
        output_tokens = list(prompt_tokens)

        # Convert prompt to numpy array for model input
        token_ids = np.array([prompt_tokens], dtype=np.int32)

        # Prefill: process entire prompt at once
        logits, kv_cache = self.model.forward(
            token_ids, position_offset=0, kv_cache=None
        )

        # Get logits for the last position
        next_token_logits = logits[0, -1, :].copy()

        # Apply repetition penalty
        if repetition_penalty != 1.0:
            next_token_logits = apply_repetition_penalty(
                next_token_logits, output_tokens, repetition_penalty
            )

        # Sample next token
        next_token = sample_token(
            next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        output_tokens.append(next_token)

        # Decode and yield just the new token
        # We decode incrementally to get proper subword handling
        prev_text = self.tokenizer.decode(output_tokens[:-1])
        curr_text = self.tokenizer.decode(output_tokens)
        yield curr_text[len(prev_text) :]

        # Check for EOS
        if next_token == self.tokenizer.eos_token_id:
            return

        # Decode: generate remaining tokens one at a time with KV cache
        for _ in range(max_new_tokens - 1):
            # Current position is length of generated sequence
            position_offset = len(output_tokens) - 1

            # Create input with just the last token
            token_ids = np.array([[next_token]], dtype=np.int32)

            # Forward pass with KV cache
            logits, kv_cache = self.model.forward(
                token_ids,
                position_offset=position_offset,
                kv_cache=kv_cache,
            )

            # Get logits for the (only) position
            next_token_logits = logits[0, -1, :].copy()

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                next_token_logits = apply_repetition_penalty(
                    next_token_logits, output_tokens, repetition_penalty
                )

            # Sample next token
            next_token = sample_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            output_tokens.append(next_token)

            # Decode and yield just the new token
            prev_text = curr_text
            curr_text = self.tokenizer.decode(output_tokens)
            yield curr_text[len(prev_text) :]

            # Check for EOS
            if next_token == self.tokenizer.eos_token_id:
                break


def create_pipeline(
    model_name: str = "google/gemma-3-270m-it",
    **kwargs,
) -> Pipeline:
    """Factory function to create a Pipeline instance.

    This is a convenience function matching common patterns in
    ML libraries.

    Args:
        model_name: HuggingFace model name
        **kwargs: Additional arguments passed to Pipeline

    Returns:
        Configured Pipeline instance
    """
    return Pipeline(model_name, **kwargs)
