"""Transformer layer implementations for Gemma-3 inference.

This module contains NumPy implementations of the core transformer components:
- RMSNorm: Root Mean Square Layer Normalization
- RoPE: Rotary Position Embeddings
- MultiHeadAttention: Grouped Query Attention with KV cache
- MLP: Feed-forward network with SwiGLU/GELU activation

These implementations are designed for inference only (no backprop needed).
"""

import numpy as np
from typing import Optional


# =============================================================================
# Activation Functions
# =============================================================================


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function.

    Args:
        x: Input array

    Returns:
        sigmoid(x) = 1 / (1 + exp(-x))
    """
    # Use stable computation to avoid overflow
    positive_mask = x >= 0
    negative_mask = ~positive_mask

    result = np.zeros_like(x)

    # For positive x: 1 / (1 + exp(-x))
    result[positive_mask] = 1 / (1 + np.exp(-x[positive_mask]))

    # For negative x: exp(x) / (1 + exp(x)) to avoid overflow
    exp_x = np.exp(x[negative_mask])
    result[negative_mask] = exp_x / (1 + exp_x)

    return result


def swish(x: np.ndarray) -> np.ndarray:
    """Swish activation function (also known as SiLU).

    Args:
        x: Input array

    Returns:
        swish(x) = x * sigmoid(x)
    """
    return x * sigmoid(x)


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation function (tanh approximation used by PyTorch).

    This is the "gelu_pytorch_tanh" variant used by Gemma.

    Args:
        x: Input array

    Returns:
        gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    """
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(sqrt_2_over_pi * (x + 0.044715 * x**3)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax.

    Args:
        x: Input array
        axis: Axis along which to compute softmax

    Returns:
        softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    """
    # Subtract max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# =============================================================================
# RMSNorm
# =============================================================================


class RMSNorm:
    """Root Mean Square Layer Normalization.

    Unlike LayerNorm, RMSNorm doesn't center the activations (no mean subtraction).
    This makes it faster while achieving similar performance.

    Gemma3 uses a residual-style weight: output = (x / RMS(x)) * (1 + weight)
    where RMS(x) = sqrt(mean(x²) + eps)

    Note: The weight is initialized to zeros in the model, so (1 + weight) starts at 1.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """Initialize RMSNorm.

        Args:
            hidden_size: Dimension of the input features
            eps: Small constant for numerical stability
        """
        self.hidden_size = hidden_size
        self.eps = eps
        # Initialize weight to zeros (Gemma3 uses 1 + weight, so this starts at 1)
        self.weight = np.zeros(hidden_size, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply RMSNorm to input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Normalized tensor of same shape
        """
        # Compute RMS along last dimension
        # variance = mean(x²)
        variance = np.mean(x**2, axis=-1, keepdims=True)
        # RMS = sqrt(variance + eps)
        rms = np.sqrt(variance + self.eps)
        # Normalize and scale with Gemma3's (1 + weight) formulation
        x_norm = x / rms
        return x_norm * (1.0 + self.weight)


# =============================================================================
# Rotary Position Embeddings (RoPE)
# =============================================================================


def compute_rope_frequencies(head_dim: int, base: float = 10000.0) -> np.ndarray:
    """Compute the inverse frequencies for RoPE.

    Args:
        head_dim: Dimension of each attention head
        base: Base frequency for the rotations (10000 for local, 1000000 for global)

    Returns:
        Inverse frequencies of shape (head_dim // 2,)
    """
    # theta_i = base^(-2i/d) for i in [0, d/2)
    dim_pairs = head_dim // 2
    freqs = base ** (-np.arange(0, dim_pairs, dtype=np.float32) * 2 / head_dim)
    return freqs


def apply_rope(
    q: np.ndarray,
    k: np.ndarray,
    positions: np.ndarray,
    head_dim: int,
    base: float = 10000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Rotary Position Embeddings to query and key tensors.

    RoPE encodes position information by rotating pairs of dimensions.
    For each pair (x, y), we apply rotation by angle θ:
        x' = x * cos(θ) - y * sin(θ)
        y' = x * sin(θ) + y * cos(θ)

    Args:
        q: Query tensor of shape (batch, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch, num_kv_heads, seq_len, head_dim)
        positions: Position indices of shape (seq_len,)
        head_dim: Dimension of each head
        base: Base frequency for rotations

    Returns:
        Tuple of (rotated_q, rotated_k) with same shapes as inputs
    """
    # Get frequencies
    freqs = compute_rope_frequencies(head_dim, base)

    # Compute angles for each position: pos * freq
    # Shape: (seq_len, head_dim // 2)
    angles = np.outer(positions.astype(np.float32), freqs)

    # Compute cos and sin
    cos = np.cos(angles)  # (seq_len, head_dim // 2)
    sin = np.sin(angles)  # (seq_len, head_dim // 2)

    # Reshape for broadcasting: (1, 1, seq_len, head_dim // 2)
    cos = cos[np.newaxis, np.newaxis, :, :]
    sin = sin[np.newaxis, np.newaxis, :, :]

    # Apply rotation to q
    q_rot = _apply_rotation(q, cos, sin)

    # Apply rotation to k
    k_rot = _apply_rotation(k, cos, sin)

    return q_rot, k_rot


def _apply_rotation(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Apply rotation to tensor using precomputed cos and sin.

    Args:
        x: Tensor of shape (batch, num_heads, seq_len, head_dim)
        cos: Cosine values of shape (1, 1, seq_len, head_dim // 2)
        sin: Sine values of shape (1, 1, seq_len, head_dim // 2)

    Returns:
        Rotated tensor of same shape as x
    """
    # Split into pairs: first half and second half of dimensions
    head_dim = x.shape[-1]
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]

    # Apply rotation:
    # x1' = x1 * cos - x2 * sin
    # x2' = x1 * sin + x2 * cos
    x1_rot = x1 * cos - x2 * sin
    x2_rot = x1 * sin + x2 * cos

    # Concatenate back
    return np.concatenate([x1_rot, x2_rot], axis=-1)


# =============================================================================
# Multi-Head Attention with Grouped Query Attention (GQA)
# =============================================================================


class MultiHeadAttention:
    """Multi-Head Attention with Grouped Query Attention and KV cache support.

    Gemma uses GQA where multiple query heads share key-value heads.
    This reduces memory while maintaining model quality.

    Gemma3 also applies RMSNorm to Q and K after projection but before RoPE.

    For Gemma-3-270m:
    - 4 query heads share 1 KV head
    - head_dim = 256
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        query_pre_attn_scalar: Optional[float] = None,
        rope_base: float = 10000.0,
        sliding_window: Optional[int] = None,
    ):
        """Initialize Multi-Head Attention.

        Args:
            hidden_size: Model hidden dimension
            num_heads: Number of query heads
            num_kv_heads: Number of key-value heads (for GQA)
            head_dim: Dimension of each attention head
            query_pre_attn_scalar: Scalar for query pre-attention (default: head_dim)
            rope_base: Base frequency for RoPE
            sliding_window: Size of sliding window attention (None for full attention)
        """
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rope_base = rope_base
        self.sliding_window = sliding_window

        # Scaling factor for attention scores
        if query_pre_attn_scalar is not None:
            self.scale = 1.0 / np.sqrt(query_pre_attn_scalar)
        else:
            self.scale = 1.0 / np.sqrt(head_dim)

        # Number of query heads per KV head (for GQA)
        self.num_heads_per_kv = num_heads // num_kv_heads

        # Initialize projection weights (will be loaded from model)
        # Q projection: hidden_size -> num_heads * head_dim
        self.q_proj = (
            np.random.randn(hidden_size, num_heads * head_dim).astype(np.float32) * 0.02
        )

        # K projection: hidden_size -> num_kv_heads * head_dim
        self.k_proj = (
            np.random.randn(hidden_size, num_kv_heads * head_dim).astype(np.float32)
            * 0.02
        )

        # V projection: hidden_size -> num_kv_heads * head_dim
        self.v_proj = (
            np.random.randn(hidden_size, num_kv_heads * head_dim).astype(np.float32)
            * 0.02
        )

        # Output projection: num_heads * head_dim -> hidden_size
        self.o_proj = (
            np.random.randn(num_heads * head_dim, hidden_size).astype(np.float32) * 0.02
        )

        # Q/K normalization (Gemma3-specific) - RMSNorm with (1 + weight) formulation
        # Weights are per head_dim, applied after reshaping to heads
        self.q_norm_weight = np.zeros(head_dim, dtype=np.float32)
        self.k_norm_weight = np.zeros(head_dim, dtype=np.float32)

    def _apply_qk_norm(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """Apply RMSNorm to Q or K tensor.

        Args:
            x: Tensor of shape (batch, num_heads, seq_len, head_dim)
            weight: Normalization weight of shape (head_dim,)

        Returns:
            Normalized tensor of same shape
        """
        # RMSNorm: x / sqrt(mean(x^2) + eps) * (1 + weight)
        variance = np.mean(x**2, axis=-1, keepdims=True)
        x_norm = x / np.sqrt(variance + 1e-6)
        return x_norm * (1.0 + weight)

    def __call__(
        self,
        x: np.ndarray,
        start_pos: int = 0,
        kv_cache: Optional[dict[str, np.ndarray]] = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Compute multi-head attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            start_pos: Starting position for RoPE (used with KV cache)
            kv_cache: Optional cached key-value tensors from previous forward passes

        Returns:
            Tuple of:
            - Output tensor of shape (batch_size, seq_len, hidden_size)
            - Updated KV cache dictionary
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = x @ self.q_proj  # (batch, seq, num_heads * head_dim)
        k = x @ self.k_proj  # (batch, seq, num_kv_heads * head_dim)
        v = x @ self.v_proj  # (batch, seq, num_kv_heads * head_dim)

        # Reshape to separate heads
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Transpose to (batch, heads, seq, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Apply Q/K normalization (Gemma3-specific)
        q = self._apply_qk_norm(q, self.q_norm_weight)
        k = self._apply_qk_norm(k, self.k_norm_weight)

        # Apply RoPE
        positions = np.arange(start_pos, start_pos + seq_len)
        q, k = apply_rope(q, k, positions, self.head_dim, self.rope_base)

        # Handle KV cache
        if kv_cache is not None:
            # Concatenate with cached keys and values
            k = np.concatenate([kv_cache["key"], k], axis=2)
            v = np.concatenate([kv_cache["value"], v], axis=2)

        # Update cache with current K, V
        new_kv_cache = {"key": k, "value": v}

        # For GQA: repeat K, V heads to match Q heads
        if self.num_heads_per_kv > 1:
            k = np.repeat(k, self.num_heads_per_kv, axis=1)
            v = np.repeat(v, self.num_heads_per_kv, axis=1)

        # Compute attention scores
        # (batch, heads, seq_q, head_dim) @ (batch, heads, head_dim, seq_kv)
        # -> (batch, heads, seq_q, seq_kv)
        attn_scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        # Apply causal mask (only for the current sequence positions)
        total_seq_len = k.shape[2]
        causal_mask = self._create_causal_mask(seq_len, total_seq_len, start_pos)
        attn_scores = attn_scores + causal_mask

        # Softmax
        attn_weights = softmax(attn_scores, axis=-1)

        # Compute output
        # (batch, heads, seq_q, seq_kv) @ (batch, heads, seq_kv, head_dim)
        # -> (batch, heads, seq_q, head_dim)
        attn_output = attn_weights @ v

        # Transpose and reshape back
        attn_output = attn_output.transpose(0, 2, 1, 3)  # (batch, seq, heads, head_dim)
        attn_output = attn_output.reshape(
            batch_size, seq_len, -1
        )  # (batch, seq, heads * head_dim)

        # Output projection
        output = attn_output @ self.o_proj

        return output, new_kv_cache

    def _create_causal_mask(
        self, seq_len: int, total_seq_len: int, start_pos: int
    ) -> np.ndarray:
        """Create causal attention mask with optional sliding window.

        Args:
            seq_len: Length of current query sequence
            total_seq_len: Total length including cached sequence
            start_pos: Starting position of current sequence

        Returns:
            Mask of shape (1, 1, seq_len, total_seq_len) with -inf for masked positions
        """
        # Create mask matrix
        mask = np.zeros((seq_len, total_seq_len), dtype=np.float32)

        # For each query position, mask out future positions and outside sliding window
        for i in range(seq_len):
            query_pos = start_pos + i
            # Can attend to positions 0 to query_pos (inclusive) for causal
            mask[i, query_pos + 1 :] = -np.inf

            # Apply sliding window mask if configured
            if self.sliding_window is not None:
                # Mask out positions before (query_pos - sliding_window + 1)
                window_start = max(0, query_pos - self.sliding_window + 1)
                if window_start > 0:
                    mask[i, :window_start] = -np.inf

        return mask.reshape(1, 1, seq_len, total_seq_len)


# =============================================================================
# MLP with SwiGLU Activation
# =============================================================================


class MLP:
    """Feed-Forward Network with GEGLU activation (GELU-gated).

    GEGLU splits the intermediate computation into gate and up projections:
        gate = gelu(x @ W_gate)
        up = x @ W_up
        output = (gate * up) @ W_down

    Gemma3 uses gelu_pytorch_tanh (tanh approximation of GELU).
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        """Initialize MLP.

        Args:
            hidden_size: Model hidden dimension
            intermediate_size: Intermediate (expanded) dimension
        """
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Initialize projection weights (will be loaded from model)
        self.gate_proj = (
            np.random.randn(hidden_size, intermediate_size).astype(np.float32) * 0.02
        )
        self.up_proj = (
            np.random.randn(hidden_size, intermediate_size).astype(np.float32) * 0.02
        )
        self.down_proj = (
            np.random.randn(intermediate_size, hidden_size).astype(np.float32) * 0.02
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply MLP transformation.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Output tensor of same shape
        """
        # Gate projection with GELU activation (Gemma3 uses gelu_pytorch_tanh)
        gate = gelu(x @ self.gate_proj)

        # Up projection (no activation)
        up = x @ self.up_proj

        # Element-wise multiplication (gating)
        hidden = gate * up

        # Down projection
        output = hidden @ self.down_proj

        return output
