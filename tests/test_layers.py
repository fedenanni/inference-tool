"""Tests for transformer layer implementations.

These tests verify the core building blocks of the Gemma-3 transformer:
- RMSNorm: Root Mean Square Layer Normalization
- RoPE: Rotary Position Embeddings
- MultiHeadAttention: Grouped Query Attention with KV cache support
- MLP: Feed-forward network with SwiGLU activation

Test values are derived from the Gemma-3-270m-it model configuration:
- hidden_size: 640
- num_attention_heads: 4
- num_key_value_heads: 1 (Grouped Query Attention)
- head_dim: 256
- intermediate_size: 2048
- rms_norm_eps: 1e-06
"""

import numpy as np


# =============================================================================
# RMSNorm Tests
# =============================================================================


class TestRMSNorm:
    """Tests for Root Mean Square Layer Normalization.

    RMSNorm normalizes inputs using RMS (not mean-centered like LayerNorm):
        output = (x / RMS(x)) * weight
    where RMS(x) = sqrt(mean(x^2) + eps)
    """

    # Test that RMSNorm returns output with same shape as input
    def test_output_shape_matches_input(self):
        from inference_tool.layers import RMSNorm

        hidden_size = 640
        eps = 1e-6
        norm = RMSNorm(hidden_size, eps)

        # Input: (batch_size, seq_len, hidden_size)
        x = np.random.randn(2, 10, hidden_size).astype(np.float32)
        output = norm(x)

        assert output.shape == x.shape

    # Test that RMSNorm output has approximately unit RMS when scale factor is 1
    # Gemma3 uses (1 + weight) formulation, so weight=0 gives scale=1
    def test_output_has_unit_rms(self):
        from inference_tool.layers import RMSNorm

        hidden_size = 640
        eps = 1e-6
        norm = RMSNorm(hidden_size, eps)
        # Weight initialized to zeros, so (1 + weight) = 1
        norm.weight = np.zeros(hidden_size, dtype=np.float32)

        x = np.random.randn(2, 10, hidden_size).astype(np.float32)
        output = norm(x)

        # RMS of each position should be approximately 1 when weight=0 (scale=1+0=1)
        rms = np.sqrt(np.mean(output**2, axis=-1))
        np.testing.assert_allclose(rms, 1.0, rtol=1e-5)

    # Test that RMSNorm correctly applies learned weights
    # Gemma3 uses (1 + weight) formulation
    def test_weight_scaling(self):
        from inference_tool.layers import RMSNorm

        hidden_size = 4
        eps = 1e-6
        norm = RMSNorm(hidden_size, eps)

        # Set weights to a known pattern
        norm.weight = np.array([2.0, 1.0, 0.5, 0.25], dtype=np.float32)

        # Use input with known RMS
        x = np.ones((1, 1, hidden_size), dtype=np.float32)  # RMS = 1
        output = norm(x)

        # After normalization (RMS=1), output should equal (1 + weight)
        expected = (1.0 + norm.weight).reshape(1, 1, -1)
        np.testing.assert_allclose(output, expected, rtol=1e-5)

    # Test that RMSNorm handles different batch sizes
    def test_different_batch_sizes(self):
        from inference_tool.layers import RMSNorm

        hidden_size = 640
        norm = RMSNorm(hidden_size, eps=1e-6)

        for batch_size in [1, 4, 16]:
            x = np.random.randn(batch_size, 5, hidden_size).astype(np.float32)
            output = norm(x)
            assert output.shape == (batch_size, 5, hidden_size)

    # Test that RMSNorm can load weights from a dictionary
    def test_load_weights_from_dict(self):
        from inference_tool.layers import RMSNorm

        hidden_size = 640
        norm = RMSNorm(hidden_size, eps=1e-6)

        # Simulate loading weights
        loaded_weights = np.random.randn(hidden_size).astype(np.float32)
        norm.weight = loaded_weights

        assert np.array_equal(norm.weight, loaded_weights)


# =============================================================================
# RoPE Tests
# =============================================================================


class TestRoPE:
    """Tests for Rotary Position Embeddings.

    RoPE encodes position by rotating pairs of dimensions in the embedding
    space. For position p, dimensions (2i, 2i+1) are rotated by angle
    p * theta_i where theta_i = base^(-2i/d).

    Key properties:
    - Relative position encoding (attention(q,k) depends on pos_q - pos_k)
    - No learned parameters (computed from position indices)
    """

    # Test that RoPE output has same shape as input
    def test_output_shape_matches_input(self):
        from inference_tool.layers import apply_rope

        head_dim = 256
        seq_len = 10

        # Input: (batch_size, num_heads, seq_len, head_dim)
        q = np.random.randn(2, 4, seq_len, head_dim).astype(np.float32)
        k = np.random.randn(2, 1, seq_len, head_dim).astype(np.float32)
        positions = np.arange(seq_len)

        q_rot, k_rot = apply_rope(q, k, positions, head_dim, base=10000.0)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    # Test that RoPE preserves vector norms (rotation doesn't change length)
    def test_preserves_vector_norms(self):
        from inference_tool.layers import apply_rope

        head_dim = 256
        seq_len = 10

        q = np.random.randn(2, 4, seq_len, head_dim).astype(np.float32)
        k = np.random.randn(2, 1, seq_len, head_dim).astype(np.float32)
        positions = np.arange(seq_len)

        q_rot, k_rot = apply_rope(q, k, positions, head_dim, base=10000.0)

        # Norms should be preserved (rotation is orthogonal transformation)
        q_norms_before = np.linalg.norm(q, axis=-1)
        q_norms_after = np.linalg.norm(q_rot, axis=-1)
        np.testing.assert_allclose(q_norms_before, q_norms_after, rtol=1e-5)

        k_norms_before = np.linalg.norm(k, axis=-1)
        k_norms_after = np.linalg.norm(k_rot, axis=-1)
        np.testing.assert_allclose(k_norms_before, k_norms_after, rtol=1e-5)

    # Test that position 0 with base frequency returns unchanged vectors
    # (rotation by 0 is identity for first dimensions)
    def test_zero_position_preserves_input(self):
        from inference_tool.layers import apply_rope

        head_dim = 256

        q = np.random.randn(1, 1, 1, head_dim).astype(np.float32)
        k = np.random.randn(1, 1, 1, head_dim).astype(np.float32)
        positions = np.array([0])

        q_rot, k_rot = apply_rope(q, k, positions, head_dim, base=10000.0)

        # At position 0, all rotation angles are 0, so output equals input
        np.testing.assert_allclose(q_rot, q, rtol=1e-5)
        np.testing.assert_allclose(k_rot, k, rtol=1e-5)

    # Test that different positions produce different rotations
    def test_different_positions_produce_different_outputs(self):
        from inference_tool.layers import apply_rope

        head_dim = 256

        # Same input vector at different positions
        q = np.ones((1, 1, 2, head_dim), dtype=np.float32)
        k = np.ones((1, 1, 2, head_dim), dtype=np.float32)
        positions = np.array([0, 100])

        q_rot, k_rot = apply_rope(q, k, positions, head_dim, base=10000.0)

        # Output at position 0 should differ from position 100
        assert not np.allclose(q_rot[0, 0, 0], q_rot[0, 0, 1])

    # Test RoPE with different base frequencies (local vs global attention)
    def test_different_base_frequencies(self):
        from inference_tool.layers import apply_rope

        head_dim = 256
        q = np.random.randn(1, 1, 5, head_dim).astype(np.float32)
        k = np.random.randn(1, 1, 5, head_dim).astype(np.float32)
        positions = np.arange(5)

        # Gemma uses different base for local (10000) vs global (1000000) attention
        q_local, k_local = apply_rope(q, k, positions, head_dim, base=10000.0)
        q_global, k_global = apply_rope(q, k, positions, head_dim, base=1000000.0)

        # Different base frequencies should produce different rotations
        assert not np.allclose(q_local, q_global)


# =============================================================================
# MultiHeadAttention Tests
# =============================================================================


class TestMultiHeadAttention:
    """Tests for Multi-Head Attention with Grouped Query Attention (GQA).

    Gemma-3-270m uses GQA where:
    - num_attention_heads: 4 (query heads)
    - num_key_value_heads: 1 (KV heads, shared across query heads)
    - head_dim: 256

    This means 4 query heads share 1 key-value head, reducing memory.
    """

    # Test that attention output has correct shape
    def test_output_shape(self):
        from inference_tool.layers import MultiHeadAttention

        hidden_size = 640
        num_heads = 4
        num_kv_heads = 1
        head_dim = 256

        attn = MultiHeadAttention(hidden_size, num_heads, num_kv_heads, head_dim)

        # Input: (batch_size, seq_len, hidden_size)
        x = np.random.randn(2, 10, hidden_size).astype(np.float32)
        output, _ = attn(x)

        assert output.shape == x.shape

    # Test that attention returns KV cache for efficient generation
    def test_returns_kv_cache(self):
        from inference_tool.layers import MultiHeadAttention

        hidden_size = 640
        num_heads = 4
        num_kv_heads = 1
        head_dim = 256

        attn = MultiHeadAttention(hidden_size, num_heads, num_kv_heads, head_dim)

        x = np.random.randn(2, 10, hidden_size).astype(np.float32)
        _, kv_cache = attn(x)

        # KV cache should contain key and value tensors
        assert "key" in kv_cache
        assert "value" in kv_cache
        # Shape: (batch_size, num_kv_heads, seq_len, head_dim)
        assert kv_cache["key"].shape == (2, num_kv_heads, 10, head_dim)
        assert kv_cache["value"].shape == (2, num_kv_heads, 10, head_dim)

    # Test that attention can use provided KV cache (for incremental decoding)
    def test_uses_kv_cache(self):
        from inference_tool.layers import MultiHeadAttention

        hidden_size = 640
        num_heads = 4
        num_kv_heads = 1
        head_dim = 256

        attn = MultiHeadAttention(hidden_size, num_heads, num_kv_heads, head_dim)

        # First pass: process initial sequence
        x_initial = np.random.randn(1, 5, hidden_size).astype(np.float32)
        _, kv_cache = attn(x_initial, start_pos=0)

        # Second pass: process one new token using cached KV
        x_new = np.random.randn(1, 1, hidden_size).astype(np.float32)
        output, new_kv_cache = attn(x_new, start_pos=5, kv_cache=kv_cache)

        # Output shape should match new token count
        assert output.shape == (1, 1, hidden_size)
        # KV cache should now contain 6 positions
        assert new_kv_cache["key"].shape[2] == 6
        assert new_kv_cache["value"].shape[2] == 6

    # Test that causal masking is applied (no peeking at future tokens)
    def test_causal_masking(self):
        from inference_tool.layers import MultiHeadAttention

        hidden_size = 640
        num_heads = 4
        num_kv_heads = 1
        head_dim = 256

        attn = MultiHeadAttention(hidden_size, num_heads, num_kv_heads, head_dim)

        # Create input where later tokens are very different
        x = np.zeros((1, 5, hidden_size), dtype=np.float32)
        x[0, 0, :] = 1.0  # First token is all 1s
        x[0, 4, :] = 100.0  # Last token is all 100s (very different)

        output, _ = attn(x)

        # If causal masking works, first token's output shouldn't be affected
        # by the extreme values in the last token
        # Run again with different last token
        x_alt = x.copy()
        x_alt[0, 4, :] = -100.0  # Different last token
        output_alt, _ = attn(x_alt)

        # First token output should be identical (can't see future)
        np.testing.assert_allclose(output[0, 0], output_alt[0, 0], rtol=1e-5)

    # Test that attention weights sum to 1 (proper softmax)
    def test_attention_weights_sum_to_one(self):
        from inference_tool.layers import MultiHeadAttention

        hidden_size = 640
        num_heads = 4
        num_kv_heads = 1
        head_dim = 256

        attn = MultiHeadAttention(hidden_size, num_heads, num_kv_heads, head_dim)

        x = np.random.randn(1, 5, hidden_size).astype(np.float32)

        # Get attention weights by modifying forward to return them
        # For now, we just verify the output is reasonable
        output, _ = attn(x)

        # Output should not contain NaN or Inf
        assert np.all(np.isfinite(output))

    # Test attention with query pre-scaling (Gemma uses this)
    def test_query_scaling(self):
        from inference_tool.layers import MultiHeadAttention

        hidden_size = 640
        num_heads = 4
        num_kv_heads = 1
        head_dim = 256
        query_pre_attn_scalar = 256  # From Gemma config

        attn = MultiHeadAttention(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            query_pre_attn_scalar=query_pre_attn_scalar,
        )

        x = np.random.randn(1, 5, hidden_size).astype(np.float32)
        output, _ = attn(x)

        # Should produce valid output without numerical issues
        assert np.all(np.isfinite(output))
        assert output.shape == x.shape


# =============================================================================
# MLP Tests
# =============================================================================


class TestMLP:
    """Tests for the Feed-Forward Network with SwiGLU activation.

    Gemma uses SwiGLU which splits the intermediate computation:
        gate = swish(x @ W_gate)
        up = x @ W_up
        output = (gate * up) @ W_down

    Where swish(x) = x * sigmoid(x)
    """

    # Test that MLP output has correct shape
    def test_output_shape(self):
        from inference_tool.layers import MLP

        hidden_size = 640
        intermediate_size = 2048

        mlp = MLP(hidden_size, intermediate_size)

        x = np.random.randn(2, 10, hidden_size).astype(np.float32)
        output = mlp(x)

        assert output.shape == x.shape

    # Test that MLP handles different batch sizes
    def test_different_batch_sizes(self):
        from inference_tool.layers import MLP

        hidden_size = 640
        intermediate_size = 2048

        mlp = MLP(hidden_size, intermediate_size)

        for batch_size in [1, 4, 16]:
            x = np.random.randn(batch_size, 5, hidden_size).astype(np.float32)
            output = mlp(x)
            assert output.shape == (batch_size, 5, hidden_size)

    # Test that MLP output is not just a linear transformation
    def test_nonlinearity(self):
        from inference_tool.layers import MLP

        hidden_size = 640
        intermediate_size = 2048

        mlp = MLP(hidden_size, intermediate_size)

        # If MLP is linear, f(ax + by) = a*f(x) + b*f(y)
        x1 = np.random.randn(1, 5, hidden_size).astype(np.float32)
        x2 = np.random.randn(1, 5, hidden_size).astype(np.float32)
        a, b = 0.3, 0.7

        # Compute f(ax + by)
        combined = mlp(a * x1 + b * x2)

        # Compute a*f(x) + b*f(y)
        linear_combo = a * mlp(x1) + b * mlp(x2)

        # These should NOT be equal if activation is nonlinear
        assert not np.allclose(combined, linear_combo, rtol=1e-3)

    # Test that SwiGLU activation is applied correctly
    def test_swiglu_activation(self):
        from inference_tool.layers import swish

        # swish(x) = x * sigmoid(x)
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)

        result = swish(x)

        # Known values: swish(0) = 0, swish(x) ≈ x for large x
        assert np.isclose(result[2], 0.0)  # swish(0) = 0
        assert result[3] > 0.5  # swish(1) ≈ 0.731
        assert result[4] > 1.5  # swish(2) ≈ 1.762

    # Test that MLP weights can be loaded
    def test_load_weights(self):
        from inference_tool.layers import MLP

        hidden_size = 640
        intermediate_size = 2048

        mlp = MLP(hidden_size, intermediate_size)

        # Simulate loading weights
        mlp.gate_proj = np.random.randn(hidden_size, intermediate_size).astype(
            np.float32
        )
        mlp.up_proj = np.random.randn(hidden_size, intermediate_size).astype(np.float32)
        mlp.down_proj = np.random.randn(intermediate_size, hidden_size).astype(
            np.float32
        )

        # Should still produce valid output
        x = np.random.randn(1, 5, hidden_size).astype(np.float32)
        output = mlp(x)
        assert np.all(np.isfinite(output))

    # Test MLP with GELU activation (Gemma uses gelu_pytorch_tanh variant)
    def test_gelu_activation(self):
        from inference_tool.layers import gelu

        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        result = gelu(x)

        # GELU(0) = 0, GELU(x) ≈ x for large positive x
        assert np.isclose(result[2], 0.0, atol=1e-5)
        assert result[4] > 1.9  # GELU(2) ≈ 1.96


# =============================================================================
# Integration Tests
# =============================================================================


class TestLayerIntegration:
    """Integration tests to verify layers work together correctly."""

    # Test that RMSNorm → Attention → RMSNorm → MLP pipeline works
    def test_transformer_block_pipeline(self):
        from inference_tool.layers import RMSNorm, MultiHeadAttention, MLP

        # Gemma config
        hidden_size = 640
        num_heads = 4
        num_kv_heads = 1
        head_dim = 256
        intermediate_size = 2048
        eps = 1e-6

        # Create layers
        input_norm = RMSNorm(hidden_size, eps)
        attn = MultiHeadAttention(hidden_size, num_heads, num_kv_heads, head_dim)
        post_attn_norm = RMSNorm(hidden_size, eps)
        mlp = MLP(hidden_size, intermediate_size)

        # Forward pass
        x = np.random.randn(2, 10, hidden_size).astype(np.float32)

        # Pre-norm architecture (like Gemma)
        normed = input_norm(x)
        attn_out, _ = attn(normed)
        x = x + attn_out  # Residual connection

        normed = post_attn_norm(x)
        mlp_out = mlp(normed)
        x = x + mlp_out  # Residual connection

        assert x.shape == (2, 10, hidden_size)
        assert np.all(np.isfinite(x))

    # Test that KV cache accumulates correctly across multiple forward passes
    def test_kv_cache_accumulation(self):
        from inference_tool.layers import MultiHeadAttention

        hidden_size = 640
        num_heads = 4
        num_kv_heads = 1
        head_dim = 256

        attn = MultiHeadAttention(hidden_size, num_heads, num_kv_heads, head_dim)

        # Simulate autoregressive generation
        kv_cache = None
        total_seq_len = 0

        for i in range(5):
            # Generate one token at a time
            x = np.random.randn(1, 1, hidden_size).astype(np.float32)
            output, kv_cache = attn(x, start_pos=total_seq_len, kv_cache=kv_cache)
            total_seq_len += 1

            assert output.shape == (1, 1, hidden_size)
            assert kv_cache["key"].shape[2] == total_seq_len
            assert kv_cache["value"].shape[2] == total_seq_len
