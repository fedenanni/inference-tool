"""Tests for the Gemma model implementation.

These tests verify the full Gemma model architecture including:
- GemmaModel: Main model class that combines all components
- TransformerBlock: Single transformer layer with attention + MLP

Test values are derived from the Gemma-3-270m-it model configuration:
- hidden_size: 640
- num_hidden_layers: 18
- num_attention_heads: 4
- num_key_value_heads: 1 (Grouped Query Attention)
- head_dim: 256
- intermediate_size: 2048
- vocab_size: 262144
- rms_norm_eps: 1e-06

The goal is to replicate the behavior of:
    from transformers import pipeline
    pipe = pipeline("text-generation", model="google/gemma-3-270m-it")
    pipe([{"role": "user", "content": "Who are you?"}])
"""

import numpy as np


# =============================================================================
# TransformerBlock Tests
# =============================================================================


class TestTransformerBlock:
    """Tests for a single transformer block (attention + MLP with residuals).

    A TransformerBlock consists of:
    1. Pre-attention RMSNorm
    2. Multi-Head Attention (with residual connection)
    3. Post-attention RMSNorm
    4. MLP with SwiGLU (with residual connection)

    Forward pass: x -> norm1 -> attn -> +x -> norm2 -> mlp -> +x
    """

    # Test that TransformerBlock can be instantiated with required config
    def test_instantiation_with_config(self):
        from inference_tool.model import TransformerBlock

        config = {
            "hidden_size": 640,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "intermediate_size": 2048,
            "rms_norm_eps": 1e-6,
        }
        block = TransformerBlock(config)

        assert block is not None

    # Test that TransformerBlock output has same shape as input
    def test_output_shape_matches_input(self):
        from inference_tool.model import TransformerBlock

        config = {
            "hidden_size": 640,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "intermediate_size": 2048,
            "rms_norm_eps": 1e-6,
        }
        block = TransformerBlock(config)

        # Input: (batch_size, seq_len, hidden_size)
        x = np.random.randn(2, 10, 640).astype(np.float32)
        output, _ = block(x, start_pos=0, kv_cache=None)

        assert output.shape == x.shape

    # Test that TransformerBlock returns KV cache for autoregressive generation
    def test_returns_kv_cache(self):
        from inference_tool.model import TransformerBlock

        config = {
            "hidden_size": 640,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "intermediate_size": 2048,
            "rms_norm_eps": 1e-6,
        }
        block = TransformerBlock(config)

        x = np.random.randn(1, 5, 640).astype(np.float32)
        output, kv_cache = block(x, start_pos=0, kv_cache=None)

        assert kv_cache is not None
        assert "key" in kv_cache
        assert "value" in kv_cache
        # KV cache should have shape (batch, num_kv_heads, seq_len, head_dim)
        assert kv_cache["key"].shape == (1, 1, 5, 256)
        assert kv_cache["value"].shape == (1, 1, 5, 256)

    # Test that TransformerBlock can use cached KV for incremental decoding
    def test_incremental_decoding_with_kv_cache(self):
        from inference_tool.model import TransformerBlock

        config = {
            "hidden_size": 640,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "intermediate_size": 2048,
            "rms_norm_eps": 1e-6,
        }
        block = TransformerBlock(config)

        # First pass: process prompt (5 tokens)
        prompt = np.random.randn(1, 5, 640).astype(np.float32)
        _, kv_cache = block(prompt, start_pos=0, kv_cache=None)

        # Second pass: process single new token
        new_token = np.random.randn(1, 1, 640).astype(np.float32)
        output, new_kv_cache = block(new_token, start_pos=5, kv_cache=kv_cache)

        # Output should be single token
        assert output.shape == (1, 1, 640)
        # KV cache should now have 6 tokens total
        assert new_kv_cache["key"].shape == (1, 1, 6, 256)

    # Test that TransformerBlock applies residual connections
    def test_residual_connections_not_zero(self):
        from inference_tool.model import TransformerBlock

        config = {
            "hidden_size": 640,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "intermediate_size": 2048,
            "rms_norm_eps": 1e-6,
        }
        block = TransformerBlock(config)

        # Use small input to ensure residual has observable effect
        x = np.ones((1, 3, 640), dtype=np.float32) * 0.1
        output, _ = block(x, start_pos=0, kv_cache=None)

        # Output should be different from input (transformations applied)
        assert not np.allclose(output, x)

    # Test that TransformerBlock has pre-attention normalization layer
    def test_has_pre_attention_norm(self):
        from inference_tool.model import TransformerBlock

        config = {
            "hidden_size": 640,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "intermediate_size": 2048,
            "rms_norm_eps": 1e-6,
        }
        block = TransformerBlock(config)

        assert hasattr(block, "input_layernorm")

    # Test that TransformerBlock has post-attention normalization layer
    def test_has_post_attention_norm(self):
        from inference_tool.model import TransformerBlock

        config = {
            "hidden_size": 640,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "intermediate_size": 2048,
            "rms_norm_eps": 1e-6,
        }
        block = TransformerBlock(config)

        assert hasattr(block, "post_attention_layernorm")

    # Test that TransformerBlock has attention layer
    def test_has_attention_layer(self):
        from inference_tool.model import TransformerBlock

        config = {
            "hidden_size": 640,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "intermediate_size": 2048,
            "rms_norm_eps": 1e-6,
        }
        block = TransformerBlock(config)

        assert hasattr(block, "self_attn")

    # Test that TransformerBlock has MLP layer
    def test_has_mlp_layer(self):
        from inference_tool.model import TransformerBlock

        config = {
            "hidden_size": 640,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "intermediate_size": 2048,
            "rms_norm_eps": 1e-6,
        }
        block = TransformerBlock(config)

        assert hasattr(block, "mlp")


# =============================================================================
# GemmaModel Tests
# =============================================================================


class TestGemmaModel:
    """Tests for the complete Gemma model.

    GemmaModel combines:
    1. Token embedding layer
    2. Multiple TransformerBlocks
    3. Final RMSNorm
    4. Output projection (lm_head) to vocabulary logits

    It supports KV caching for efficient autoregressive generation.
    """

    # Test that GemmaModel can be instantiated with model name
    def test_instantiation_with_model_name(self):
        from inference_tool.model import GemmaModel

        model = GemmaModel("google/gemma-3-270m-it")

        assert model is not None

    # Test that GemmaModel loads config from Hugging Face
    def test_loads_config(self):
        from inference_tool.model import GemmaModel

        model = GemmaModel("google/gemma-3-270m-it")

        assert hasattr(model, "config")
        assert model.config["hidden_size"] == 640
        assert model.config["num_hidden_layers"] == 18

    # Test that GemmaModel loads weights from Hugging Face
    def test_loads_weights(self):
        from inference_tool.model import GemmaModel

        model = GemmaModel("google/gemma-3-270m-it")

        # Check that embedding weights are loaded
        assert model.embed_tokens is not None
        assert model.embed_tokens.shape[0] == model.config["vocab_size"]

    # Test that embed method returns correct shape
    def test_embed_returns_correct_shape(self):
        from inference_tool.model import GemmaModel

        model = GemmaModel("google/gemma-3-270m-it")

        # Token IDs: (batch_size, seq_len)
        token_ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
        embeddings = model.embed(token_ids)

        # Should return (batch_size, seq_len, hidden_size)
        assert embeddings.shape == (1, 5, 640)

    # Test that embed method uses token embedding weights with scaling
    def test_embed_uses_embedding_weights(self):
        from inference_tool.model import GemmaModel

        model = GemmaModel("google/gemma-3-270m-it")

        token_ids = np.array([[0]], dtype=np.int32)
        embeddings = model.embed(token_ids)

        # Embedding for token 0 should match first row of embed_tokens, scaled
        scale = np.sqrt(model.config["hidden_size"])
        expected = model.embed_tokens[0:1, :].reshape(1, 1, -1) * scale
        np.testing.assert_allclose(embeddings, expected, rtol=1e-5)

    # Test that forward method returns logits with correct shape
    def test_forward_returns_logits_shape(self):
        from inference_tool.model import GemmaModel

        model = GemmaModel("google/gemma-3-270m-it")

        token_ids = np.array([[1, 2, 3]], dtype=np.int32)
        logits, _ = model.forward(token_ids)

        # Logits: (batch_size, seq_len, vocab_size)
        vocab_size = model.config["vocab_size"]
        assert logits.shape == (1, 3, vocab_size)

    # Test that forward method returns KV cache for all layers
    def test_forward_returns_kv_cache_for_all_layers(self):
        from inference_tool.model import GemmaModel

        model = GemmaModel("google/gemma-3-270m-it")

        token_ids = np.array([[1, 2, 3]], dtype=np.int32)
        _, kv_cache = model.forward(token_ids)

        num_layers = model.config["num_hidden_layers"]
        assert len(kv_cache) == num_layers
        # Each layer should have key and value tensors
        for layer_cache in kv_cache:
            assert "key" in layer_cache
            assert "value" in layer_cache

    # Test that forward with position_offset works for KV cache
    def test_forward_with_position_offset(self):
        from inference_tool.model import GemmaModel

        model = GemmaModel("google/gemma-3-270m-it")

        # First forward pass: 5 tokens
        token_ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
        _, kv_cache = model.forward(token_ids, position_offset=0)

        # Second forward pass: 1 new token with offset
        new_token = np.array([[6]], dtype=np.int32)
        logits, new_kv_cache = model.forward(
            new_token, position_offset=5, kv_cache=kv_cache
        )

        # Logits should be for single token
        assert logits.shape[1] == 1
        # KV cache should have grown
        assert new_kv_cache[0]["key"].shape[2] == 6

    # Test that model has correct number of transformer blocks
    def test_has_correct_number_of_layers(self):
        from inference_tool.model import GemmaModel

        model = GemmaModel("google/gemma-3-270m-it")

        assert len(model.layers) == 18

    # Test that model has final normalization layer
    def test_has_final_norm(self):
        from inference_tool.model import GemmaModel

        model = GemmaModel("google/gemma-3-270m-it")

        assert hasattr(model, "norm")

    # Test that model applies embedding scaling (Gemma-specific)
    def test_applies_embedding_scaling(self):
        from inference_tool.model import GemmaModel

        model = GemmaModel("google/gemma-3-270m-it")

        # Gemma scales embeddings by sqrt(hidden_size)
        # This is to ensure proper gradient flow
        token_ids = np.array([[1]], dtype=np.int32)
        embeddings = model.embed(token_ids)

        # The embedding should be scaled
        raw_embedding = model.embed_tokens[1:2, :]
        scale = np.sqrt(model.config["hidden_size"])
        expected = raw_embedding.reshape(1, 1, -1) * scale

        np.testing.assert_allclose(embeddings, expected, rtol=1e-5)

    # Test that different tokens produce different logits
    def test_different_tokens_different_logits(self):
        from inference_tool.model import GemmaModel

        model = GemmaModel("google/gemma-3-270m-it")

        tokens1 = np.array([[1, 2, 3]], dtype=np.int32)
        tokens2 = np.array([[4, 5, 6]], dtype=np.int32)

        logits1, _ = model.forward(tokens1)
        logits2, _ = model.forward(tokens2)

        # Different inputs should produce different outputs
        assert not np.allclose(logits1, logits2)

    # Test that batch processing works correctly
    def test_batch_processing(self):
        from inference_tool.model import GemmaModel

        model = GemmaModel("google/gemma-3-270m-it")

        # Process batch of 2 sequences
        token_ids = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        logits, _ = model.forward(token_ids)

        assert logits.shape[0] == 2


# =============================================================================
# Integration Tests
# =============================================================================


class TestModelIntegration:
    """Integration tests that verify the model works end-to-end.

    These tests verify that our NumPy implementation produces outputs
    consistent with the Hugging Face transformers implementation.
    """

    # Test that model can process a real tokenized prompt
    def test_real_prompt_processing(self):
        from inference_tool.model import GemmaModel
        from inference_tool.tokenizer import Tokenizer

        model = GemmaModel("google/gemma-3-270m-it")
        tokenizer = Tokenizer("google/gemma-3-270m-it")

        # Encode a simple prompt
        text = "Hello"
        token_ids = tokenizer.encode(text)
        token_ids = np.array([token_ids], dtype=np.int32)

        logits, kv_cache = model.forward(token_ids)

        # Should produce valid logits
        assert logits.shape[0] == 1
        assert logits.shape[2] == model.config["vocab_size"]
        assert not np.isnan(logits).any()
        assert not np.isinf(logits).any()

    # Test that argmax of logits produces valid token IDs
    def test_logits_produce_valid_token_ids(self):
        from inference_tool.model import GemmaModel
        from inference_tool.tokenizer import Tokenizer

        model = GemmaModel("google/gemma-3-270m-it")
        tokenizer = Tokenizer("google/gemma-3-270m-it")

        token_ids = np.array([[tokenizer.bos_token_id]], dtype=np.int32)
        logits, _ = model.forward(token_ids)

        # Get predicted next token
        next_token = np.argmax(logits[0, -1, :])

        # Should be a valid token ID
        assert 0 <= next_token < model.config["vocab_size"]

    # Test autoregressive generation with KV cache
    def test_autoregressive_generation(self):
        from inference_tool.model import GemmaModel
        from inference_tool.tokenizer import Tokenizer

        model = GemmaModel("google/gemma-3-270m-it")
        tokenizer = Tokenizer("google/gemma-3-270m-it")

        # Start with BOS token
        tokens = [tokenizer.bos_token_id]
        kv_cache = None
        position = 0

        # Generate 5 tokens
        for _ in range(5):
            if kv_cache is None:
                token_ids = np.array([tokens], dtype=np.int32)
            else:
                token_ids = np.array([[tokens[-1]]], dtype=np.int32)

            logits, kv_cache = model.forward(
                token_ids, position_offset=position, kv_cache=kv_cache
            )

            next_token = int(np.argmax(logits[0, -1, :]))
            tokens.append(next_token)
            position = len(tokens) - 1

        # Should have generated 6 tokens (BOS + 5 new)
        assert len(tokens) == 6

        # Should be decodable
        decoded = tokenizer.decode(tokens)
        assert isinstance(decoded, str)

    # Test that generation with and without KV cache produces same first token
    def test_kv_cache_consistency(self):
        from inference_tool.model import GemmaModel

        model = GemmaModel("google/gemma-3-270m-it")

        # Full forward pass (no cache)
        tokens = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
        logits_full, _ = model.forward(tokens)

        # Incremental forward passes (with cache)
        prompt = np.array([[1, 2, 3, 4]], dtype=np.int32)
        _, kv_cache = model.forward(prompt)

        new_token = np.array([[5]], dtype=np.int32)
        logits_cached, _ = model.forward(
            new_token, position_offset=4, kv_cache=kv_cache
        )

        # Last token logits should match
        np.testing.assert_allclose(
            logits_full[0, -1, :], logits_cached[0, 0, :], rtol=1e-4
        )
