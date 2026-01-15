"""Text generation engine for Gemma inference.

This module provides autoregressive text generation with various sampling strategies:
- Greedy decoding (temperature=0)
- Temperature scaling
- Top-k filtering
- Top-p (nucleus) sampling
- Repetition penalty

The generate function uses KV caching for efficient autoregressive generation.

Usage:
    from generate import generate
    from model import GemmaModel
    from tokenizer import Tokenizer

    model = GemmaModel("google/gemma-3-270m-it")
    tokenizer = Tokenizer("google/gemma-3-270m-it")

    prompt_tokens = tokenizer.encode("Hello, world!")
    output_tokens = generate(
        model=model,
        tokenizer=tokenizer,
        prompt_tokens=prompt_tokens,
        max_new_tokens=50,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
    )
    print(tokenizer.decode(output_tokens))
"""

import numpy as np
from typing import Optional


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for logits.

    Args:
        x: Input logits array

    Returns:
        Probability distribution (sums to 1)
    """
    # Subtract max for numerical stability
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)


def apply_repetition_penalty(
    logits: np.ndarray,
    previous_tokens: list[int],
    penalty: float = 1.0,
) -> np.ndarray:
    """Apply repetition penalty to reduce probability of repeated tokens.

    For tokens that have appeared before, we divide positive logits by the penalty
    and multiply negative logits by the penalty. This reduces the probability
    of selecting previously used tokens.

    Args:
        logits: Logits array of shape (vocab_size,)
        previous_tokens: List of token IDs that have been generated
        penalty: Repetition penalty factor (1.0 = no penalty, >1.0 = penalize repeats)

    Returns:
        Modified logits array
    """
    if penalty == 1.0 or not previous_tokens:
        return logits

    for token_id in previous_tokens:
        if token_id < len(logits):
            if logits[token_id] > 0:
                logits[token_id] = logits[token_id] / penalty
            else:
                logits[token_id] = logits[token_id] * penalty

    return logits


def sample_token(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> int:
    """Sample a token from logits using various sampling strategies.

    The sampling process:
    1. Apply temperature scaling (if temperature > 0)
    2. Apply top-k filtering (if specified)
    3. Apply top-p (nucleus) filtering (if specified)
    4. Sample from the resulting distribution

    Args:
        logits: Logits array of shape (vocab_size,)
        temperature: Temperature for scaling (0 = greedy, higher = more random)
        top_k: If set, only consider the top k tokens
        top_p: If set, only consider tokens until cumulative probability >= top_p

    Returns:
        Sampled token ID
    """
    # Greedy decoding: return argmax
    if temperature == 0.0:
        return int(np.argmax(logits))

    # Top-k=1 is equivalent to greedy
    if top_k == 1:
        return int(np.argmax(logits))

    # Apply temperature scaling
    scaled_logits = logits / temperature

    # Apply top-k filtering
    if top_k is not None and top_k > 0:
        # Get indices of top-k logits
        top_k_indices = np.argsort(scaled_logits)[-top_k:]

        # Create a mask and set non-top-k to -inf
        mask = np.full_like(scaled_logits, -np.inf)
        mask[top_k_indices] = scaled_logits[top_k_indices]
        scaled_logits = mask

    # Convert to probabilities
    probs = softmax(scaled_logits)

    # Apply top-p (nucleus) filtering
    if top_p is not None and top_p < 1.0:
        # Sort probabilities in descending order
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]

        # Find cumulative probabilities
        cumulative_probs = np.cumsum(sorted_probs)

        # Find cutoff index where cumulative prob exceeds top_p
        cutoff_idx = np.searchsorted(cumulative_probs, top_p) + 1

        # Zero out probabilities beyond cutoff
        mask_indices = sorted_indices[cutoff_idx:]
        probs[mask_indices] = 0.0

        # Renormalize
        prob_sum = np.sum(probs)
        if prob_sum > 0:
            probs = probs / prob_sum

    # Sample from the distribution
    token_id = np.random.choice(len(probs), p=probs)

    return int(token_id)


def generate(
    model,
    tokenizer,
    prompt_tokens: list[int],
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    stop_token_ids: Optional[list[int]] = None,
) -> list[int]:
    """Generate text autoregressively using the model.

    This function performs efficient autoregressive generation:
    1. Prefill: Process the entire prompt in one forward pass
    2. Decode: Generate one token at a time using KV cache

    Generation stops when:
    - EOS token is generated
    - Any stop token is generated (for chat models, <end_of_turn>)
    - max_new_tokens is reached

    Args:
        model: The Gemma model with forward(token_ids, position_offset, kv_cache) method
        tokenizer: Tokenizer with eos_token_id attribute
        prompt_tokens: List of token IDs for the prompt
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature (0 = greedy, higher = more random)
        top_k: If set, only sample from top k tokens
        top_p: If set, only sample from tokens with cumulative prob <= top_p
        repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)
        stop_token_ids: Additional token IDs that should stop generation
                        (e.g., <end_of_turn> token for chat models)

    Returns:
        List of token IDs including prompt and generated tokens
    """
    # Build set of tokens that should stop generation
    stop_tokens = {tokenizer.eos_token_id}
    if stop_token_ids:
        stop_tokens.update(stop_token_ids)

    # For Gemma chat models, also stop on <end_of_turn> (token 106)
    # This is auto-detected from the tokenizer if possible
    end_of_turn_id = 106  # <end_of_turn> token for Gemma
    stop_tokens.add(end_of_turn_id)

    # Ensure BOS token is at the start (Gemma3 requires this)
    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is not None and (
        len(prompt_tokens) == 0 or prompt_tokens[0] != bos_token_id
    ):
        prompt_tokens = [bos_token_id] + list(prompt_tokens)

    # Initialize output with prompt tokens
    output_tokens = list(prompt_tokens)

    # Convert prompt to numpy array for model input
    token_ids = np.array([prompt_tokens], dtype=np.int32)

    # Prefill: process entire prompt at once
    logits, kv_cache = model.forward(token_ids, position_offset=0, kv_cache=None)

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

    # Check for stop tokens (EOS, <end_of_turn>, etc.)
    if next_token in stop_tokens:
        return output_tokens

    # Decode: generate remaining tokens one at a time with KV cache
    for _ in range(max_new_tokens - 1):
        # Current position is length of generated sequence
        position_offset = len(output_tokens) - 1

        # Create input with just the last token
        token_ids = np.array([[next_token]], dtype=np.int32)

        # Forward pass with KV cache
        logits, kv_cache = model.forward(
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

        # Check for stop tokens (EOS, <end_of_turn>, etc.)
        if next_token in stop_tokens:
            break

    return output_tokens


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    stop_token_ids: Optional[list[int]] = None,
) -> str:
    """Convenience function to generate text from a string prompt.

    Args:
        model: The Gemma model
        tokenizer: Tokenizer for encoding/decoding
        prompt: Input text prompt
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering parameter
        top_p: Top-p (nucleus) filtering parameter
        repetition_penalty: Penalty for repeating tokens
        stop_token_ids: Additional token IDs that should stop generation

    Returns:
        Generated text string (including the prompt)
    """
    prompt_tokens = tokenizer.encode(prompt)

    output_tokens = generate(
        model=model,
        tokenizer=tokenizer,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stop_token_ids=stop_token_ids,
    )

    return tokenizer.decode(output_tokens)
