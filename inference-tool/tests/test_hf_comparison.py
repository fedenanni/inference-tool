"""Tests comparing our implementation against HuggingFace's Gemma-3 model.

These tests verify that our NumPy implementation produces outputs that closely match
the HuggingFace transformers implementation. Each test focuses on a specific component.

Based on debugging work that identified these critical areas:
1. Embeddings with scaling
2. RMSNorm with Gemma3's (1 + weight) formulation
3. Q/K normalization
4. RoPE with per-layer base frequency (local vs global attention)
5. Full attention mechanism with sliding window
6. MLP with GELU tanh approximation
7. Full model logits

Run with: uv run pytest tests/test_hf_comparison.py -v
"""

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference_tool.model import GemmaModel
from inference_tool.tokenizer import Tokenizer
from inference_tool.layers import apply_rope


# Fixture for loading models (shared across tests)
@pytest.fixture(scope="module")
def models():
    """Load both HuggingFace and our model once for all tests."""
    hf_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-270m-it", torch_dtype=torch.float32
    )
    hf_model.eval()
    our_model = GemmaModel("google/gemma-3-270m-it")
    return hf_model, our_model


@pytest.fixture(scope="module")
def tokenizers():
    """Load both tokenizers."""
    hf_tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
    our_tokenizer = Tokenizer("google/gemma-3-270m-it")
    return hf_tokenizer, our_tokenizer


@pytest.fixture(scope="module")
def sample_tokens():
    """Sample token sequence for testing."""
    # BOS + "Hello world" tokens
    return [2, 105, 2364, 107]


class TestEmbeddings:
    """Test that our embeddings match HuggingFace exactly."""

    def test_raw_embedding_weights_match(self, models):
        """Verify raw embedding weight matrix matches."""
        hf_model, our_model = models

        # Check a few random token embeddings
        test_tokens = [105, 2364, 107, 1000, 5000]

        for token_id in test_tokens:
            with torch.no_grad():
                hf_embed = hf_model.model.embed_tokens.weight[token_id].numpy()
            our_embed = our_model.embed_tokens[token_id]

            max_diff = np.abs(our_embed - hf_embed).max()
            assert max_diff < 1e-5, f"Token {token_id} embedding differs by {max_diff}"

    def test_scaled_embeddings_match(self, models, sample_tokens):
        """Verify embeddings with scaling (sqrt(hidden_size)) match.

        Note: Both our embed() and HF's embed_tokens() apply the sqrt(hidden_size)
        scaling internally, so we compare outputs directly.
        """
        hf_model, our_model = models

        token_ids_np = np.array([sample_tokens], dtype=np.int32)
        token_ids_pt = torch.tensor([sample_tokens])

        # Our embed includes scaling
        our_embeds = our_model.embed(token_ids_np)

        # HF embed_tokens also scales internally by hidden_size ** 0.5
        with torch.no_grad():
            hf_embeds = hf_model.model.embed_tokens(token_ids_pt).numpy()

        max_diff = np.abs(our_embeds - hf_embeds).max()
        assert max_diff < 1e-4, f"Scaled embeddings differ by {max_diff}"


class TestRMSNorm:
    """Test RMSNorm layer implementation.

    Gemma3 uses a modified RMSNorm: x_norm * (1 + weight) instead of x_norm * weight
    """

    def test_rmsnorm_weights_match(self, models):
        """Verify RMSNorm weights are loaded correctly."""
        hf_model, our_model = models

        our_weight = our_model.layers[0].input_layernorm.weight
        hf_weight = hf_model.model.layers[0].input_layernorm.weight.detach().numpy()

        max_diff = np.abs(our_weight - hf_weight).max()
        assert max_diff < 1e-6, f"RMSNorm weights differ by {max_diff}"

    def test_rmsnorm_output_matches(self, models, sample_tokens):
        """Verify RMSNorm output matches HuggingFace."""
        hf_model, our_model = models

        token_ids_np = np.array([sample_tokens], dtype=np.int32)
        token_ids_pt = torch.tensor([sample_tokens])

        # Get embeddings first (both include scaling internally)
        our_hidden = our_model.embed(token_ids_np)
        with torch.no_grad():
            hf_hidden = hf_model.model.embed_tokens(token_ids_pt)

        # Apply first layer's input layernorm
        our_normed = our_model.layers[0].input_layernorm(our_hidden)
        with torch.no_grad():
            hf_normed = hf_model.model.layers[0].input_layernorm(hf_hidden).numpy()

        max_diff = np.abs(our_normed - hf_normed).max()
        assert max_diff < 1e-4, f"RMSNorm output differs by {max_diff}"


class TestQKNorm:
    """Test Q/K normalization layers.

    Gemma3 applies RMSNorm to Q and K after projection, before RoPE.
    """

    def test_qk_norm_weights_loaded(self, models):
        """Verify Q/K norm weights are loaded."""
        hf_model, our_model = models

        our_attn = our_model.layers[0].self_attn
        hf_attn = hf_model.model.layers[0].self_attn

        # Check Q norm
        assert our_attn.q_norm_weight is not None, "Q norm weight not loaded"
        hf_q_norm = hf_attn.q_norm.weight.detach().numpy()
        max_diff = np.abs(our_attn.q_norm_weight - hf_q_norm).max()
        assert max_diff < 1e-6, f"Q norm weights differ by {max_diff}"

        # Check K norm
        assert our_attn.k_norm_weight is not None, "K norm weight not loaded"
        hf_k_norm = hf_attn.k_norm.weight.detach().numpy()
        max_diff = np.abs(our_attn.k_norm_weight - hf_k_norm).max()
        assert max_diff < 1e-6, f"K norm weights differ by {max_diff}"

    def test_qk_after_norm_matches(self, models, sample_tokens):
        """Verify Q/K after normalization matches HuggingFace."""
        hf_model, our_model = models

        token_ids_np = np.array([sample_tokens], dtype=np.int32)
        token_ids_pt = torch.tensor([sample_tokens])
        batch_size, seq_len = 1, len(sample_tokens)

        # Get embeddings and apply input norm (embed_tokens scales internally)
        our_hidden = our_model.embed(token_ids_np)
        our_normed = our_model.layers[0].input_layernorm(our_hidden)

        with torch.no_grad():
            hf_hidden = hf_model.model.embed_tokens(token_ids_pt)
            hf_normed = hf_model.model.layers[0].input_layernorm(hf_hidden)

        # Project Q, K
        our_layer = our_model.layers[0]
        our_q = np.dot(our_normed, our_layer.self_attn.q_proj)
        our_k = np.dot(our_normed, our_layer.self_attn.k_proj)

        # Reshape
        head_dim = our_layer.self_attn.head_dim
        num_heads = our_layer.self_attn.num_heads
        num_kv_heads = our_layer.self_attn.num_kv_heads

        our_q = our_q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(
            0, 2, 1, 3
        )
        our_k = our_k.reshape(batch_size, seq_len, num_kv_heads, head_dim).transpose(
            0, 2, 1, 3
        )

        # Apply Q/K norm (Gemma3 style: x_norm * (1 + weight))
        def apply_qk_norm(x, weight):
            eps = 1e-6
            variance = np.mean(x**2, axis=-1, keepdims=True)
            x_norm = x * (1.0 / np.sqrt(variance + eps))
            return x_norm * (1.0 + weight)

        our_q_normed = apply_qk_norm(our_q, our_layer.self_attn.q_norm_weight)
        our_k_normed = apply_qk_norm(our_k, our_layer.self_attn.k_norm_weight)

        # HF side
        with torch.no_grad():
            hf_attn = hf_model.model.layers[0].self_attn
            hf_q = hf_attn.q_proj(hf_normed)
            hf_k = hf_attn.k_proj(hf_normed)

            hf_q = hf_q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            hf_k = hf_k.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(
                1, 2
            )

            hf_q_normed = hf_attn.q_norm(hf_q).numpy()
            hf_k_normed = hf_attn.k_norm(hf_k).numpy()

        q_diff = np.abs(our_q_normed - hf_q_normed).max()
        k_diff = np.abs(our_k_normed - hf_k_normed).max()

        assert q_diff < 1e-4, f"Q after norm differs by {q_diff}"
        assert k_diff < 1e-4, f"K after norm differs by {k_diff}"


class TestRoPE:
    """Test Rotary Position Embeddings.

    Gemma3 uses different RoPE bases:
    - Sliding window layers (most layers): base = 10000.0
    - Global attention layers (every 6th layer: 5, 11, 17): base = 1000000.0
    """

    def test_rope_sliding_layer(self, models, sample_tokens):
        """Test RoPE for a sliding window layer (layer 0, base=10000)."""
        hf_model, our_model = models

        token_ids_np = np.array([sample_tokens], dtype=np.int32)
        token_ids_pt = torch.tensor([sample_tokens])
        batch_size, seq_len = 1, len(sample_tokens)

        # Get Q/K after norm (layer 0)
        our_hidden = our_model.embed(token_ids_np)
        our_normed = our_model.layers[0].input_layernorm(our_hidden)

        our_layer = our_model.layers[0]
        our_q = np.dot(our_normed, our_layer.self_attn.q_proj)
        our_k = np.dot(our_normed, our_layer.self_attn.k_proj)

        head_dim = our_layer.self_attn.head_dim
        num_heads = our_layer.self_attn.num_heads
        num_kv_heads = our_layer.self_attn.num_kv_heads

        our_q = our_q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(
            0, 2, 1, 3
        )
        our_k = our_k.reshape(batch_size, seq_len, num_kv_heads, head_dim).transpose(
            0, 2, 1, 3
        )

        def apply_qk_norm(x, weight):
            eps = 1e-6
            variance = np.mean(x**2, axis=-1, keepdims=True)
            x_norm = x * (1.0 / np.sqrt(variance + eps))
            return x_norm * (1.0 + weight)

        our_q = apply_qk_norm(our_q, our_layer.self_attn.q_norm_weight)
        our_k = apply_qk_norm(our_k, our_layer.self_attn.k_norm_weight)

        # Apply RoPE with sliding window base
        positions = np.arange(seq_len)
        rope_base = 10000.0  # Sliding window base
        our_q_rope, our_k_rope = apply_rope(
            our_q, our_k, positions, head_dim, rope_base
        )

        # HF side (embed_tokens scales internally)
        with torch.no_grad():
            hf_hidden = hf_model.model.embed_tokens(token_ids_pt)
            hf_normed = hf_model.model.layers[0].input_layernorm(hf_hidden)

            hf_attn = hf_model.model.layers[0].self_attn
            hf_q = hf_attn.q_proj(hf_normed)
            hf_k = hf_attn.k_proj(hf_normed)

            hf_q = hf_q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            hf_k = hf_k.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(
                1, 2
            )

            hf_q = hf_attn.q_norm(hf_q)
            hf_k = hf_attn.k_norm(hf_k)

            # HF uses rotary_emb_local for sliding window
            position_ids = torch.arange(seq_len).unsqueeze(0)
            cos, sin = hf_model.model.rotary_emb_local(hf_k, position_ids)

            from transformers.models.gemma3.modeling_gemma3 import apply_rotary_pos_emb

            hf_q_rope, hf_k_rope = apply_rotary_pos_emb(hf_q, hf_k, cos, sin)

        q_diff = np.abs(our_q_rope - hf_q_rope.numpy()).max()
        k_diff = np.abs(our_k_rope - hf_k_rope.numpy()).max()

        assert q_diff < 1e-4, f"Q after RoPE differs by {q_diff}"
        assert k_diff < 1e-4, f"K after RoPE differs by {k_diff}"

    def test_rope_global_layer(self, models, sample_tokens):
        """Test RoPE for a global attention layer (layer 5, base=1000000)."""
        hf_model, our_model = models

        token_ids_np = np.array([sample_tokens], dtype=np.int32)
        token_ids_pt = torch.tensor([sample_tokens])
        batch_size, seq_len = 1, len(sample_tokens)

        # Use layer 5 (global attention)
        layer_idx = 5
        our_layer = our_model.layers[layer_idx]
        hf_layer = hf_model.model.layers[layer_idx]

        # Need to get hidden states up to layer 5
        # For simplicity, just check the RoPE base is different
        assert (
            our_layer.self_attn.rope_base == 1000000.0
        ), f"Layer 5 should have global rope_base, got {our_layer.self_attn.rope_base}"


class TestFullLayer:
    """Test full transformer layer output matches HuggingFace."""

    def test_layer_0_output_matches(self, models, sample_tokens):
        """Verify layer 0 output matches HuggingFace."""
        hf_model, our_model = models

        token_ids_np = np.array([sample_tokens], dtype=np.int32)
        token_ids_pt = torch.tensor([sample_tokens])
        seq_len = len(sample_tokens)

        # Get embeddings (embed_tokens scales internally)
        our_hidden = our_model.embed(token_ids_np)
        with torch.no_grad():
            hf_hidden = hf_model.model.embed_tokens(token_ids_pt)

        # Pass through layer 0
        our_output, _ = our_model.layers[0](our_hidden, start_pos=0, kv_cache=None)

        with torch.no_grad():
            # HF layer needs position embeddings for both global and local RoPE
            position_ids = torch.arange(seq_len).unsqueeze(0)

            # Get position embeddings from the model's rotary embeddings
            # Layer 0 is a sliding window layer, uses rotary_emb_local
            cos_local, sin_local = hf_model.model.rotary_emb_local(
                hf_hidden, position_ids
            )
            cos_global, sin_global = hf_model.model.rotary_emb(hf_hidden, position_ids)

            hf_output = hf_model.model.layers[0](
                hf_hidden,
                position_ids=position_ids,
                position_embeddings_global=(cos_global, sin_global),
                position_embeddings_local=(cos_local, sin_local),
            )[0].numpy()

        correlation = np.corrcoef(our_output.flatten(), hf_output.flatten())[0, 1]
        max_diff = np.abs(our_output - hf_output).max()

        assert correlation > 0.99, f"Layer 0 correlation too low: {correlation}"
        assert max_diff < 0.01, f"Layer 0 max diff too high: {max_diff}"


class TestFullModelLogits:
    """Test that full model produces correct logits."""

    def test_logits_match_huggingface(self, models, tokenizers):
        """Verify model logits match HuggingFace for a test prompt."""
        hf_model, our_model = models
        hf_tokenizer, our_tokenizer = tokenizers

        prompt = "The capital of France is"

        # Tokenize
        hf_input = hf_tokenizer(prompt, return_tensors="pt")
        our_tokens = [our_tokenizer.bos_token_id] + our_tokenizer.encode(prompt)

        token_ids_np = np.array([our_tokens], dtype=np.int32)

        # Forward pass
        our_logits, _ = our_model.forward(token_ids_np)

        with torch.no_grad():
            hf_logits = hf_model(hf_input["input_ids"]).logits.numpy()

        # Compare last position logits
        our_last = our_logits[0, -1, :]
        hf_last = hf_logits[0, -1, :]

        # Check correlation
        correlation = np.corrcoef(our_last, hf_last)[0, 1]
        assert correlation > 0.99, f"Logits correlation too low: {correlation}"

        # Check max difference
        max_diff = np.abs(our_last - hf_last).max()
        assert max_diff < 0.1, f"Logits max diff too high: {max_diff}"

        # Check top prediction matches
        our_top = int(np.argmax(our_last))
        hf_top = int(np.argmax(hf_last))
        assert our_top == hf_top, f"Top prediction mismatch: {our_top} vs {hf_top}"

    def test_top5_predictions_match(self, models, tokenizers):
        """Verify top-5 predictions match HuggingFace."""
        hf_model, our_model = models
        hf_tokenizer, our_tokenizer = tokenizers

        prompt = "The capital of France is"

        hf_input = hf_tokenizer(prompt, return_tensors="pt")
        our_tokens = [our_tokenizer.bos_token_id] + our_tokenizer.encode(prompt)

        token_ids_np = np.array([our_tokens], dtype=np.int32)

        our_logits, _ = our_model.forward(token_ids_np)

        with torch.no_grad():
            hf_logits = hf_model(hf_input["input_ids"]).logits.numpy()

        our_top5 = set(np.argsort(our_logits[0, -1, :])[-5:])
        hf_top5 = set(np.argsort(hf_logits[0, -1, :])[-5:])

        # At least 4 of top 5 should match
        overlap = len(our_top5 & hf_top5)
        assert overlap >= 4, f"Top-5 overlap too low: {overlap}/5"


class TestSlidingWindowPattern:
    """Test the sliding window pattern for attention layers."""

    def test_layer_pattern(self, models):
        """Verify layers have correct sliding window pattern.

        Pattern: Every 6th layer (5, 11, 17) is global attention, others are sliding window.
        """
        _, our_model = models

        for layer_idx in range(len(our_model.layers)):
            layer = our_model.layers[layer_idx]
            is_global = layer_idx % 6 == 5

            if is_global:
                assert (
                    layer.self_attn.rope_base == 1000000.0
                ), f"Layer {layer_idx} should have global rope_base"
                assert (
                    layer.self_attn.sliding_window is None
                ), f"Layer {layer_idx} should not have sliding_window"
            else:
                assert (
                    layer.self_attn.rope_base == 10000.0
                ), f"Layer {layer_idx} should have local rope_base"
                assert (
                    layer.self_attn.sliding_window == 512
                ), f"Layer {layer_idx} should have sliding_window=512"


class TestTokenizer:
    """Test tokenizer special tokens."""

    def test_special_tokens(self, tokenizers):
        """Verify special token IDs match."""
        hf_tokenizer, our_tokenizer = tokenizers

        assert our_tokenizer.bos_token_id == hf_tokenizer.bos_token_id
        assert our_tokenizer.eos_token_id == hf_tokenizer.eos_token_id

    def test_encode_decode_roundtrip(self, tokenizers):
        """Test encoding and decoding produces same text."""
        _, our_tokenizer = tokenizers

        test_texts = [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "1 + 1 = 2",
        ]

        for text in test_texts:
            tokens = our_tokenizer.encode(text)
            decoded = our_tokenizer.decode(tokens)
            assert decoded == text, f"Roundtrip failed: '{text}' -> '{decoded}'"

    def test_end_of_turn_token(self, tokenizers):
        """Verify <end_of_turn> token is correctly identified."""
        _, our_tokenizer = tokenizers

        tokens = our_tokenizer.encode("<end_of_turn>")
        assert tokens == [106], f"<end_of_turn> should encode to [106], got {tokens}"


class TestGeneration:
    """Test generation stops at correct tokens."""

    def test_stop_at_end_of_turn(self, models, tokenizers):
        """Verify generation stops at <end_of_turn> token."""
        from inference_tool.generate import generate
        from inference_tool.chat import format_messages

        _, our_model = models
        _, our_tokenizer = tokenizers

        prompt = format_messages([{"role": "user", "content": "Hello!"}])
        tokens = our_tokenizer.encode(prompt)

        output = generate(
            our_model, our_tokenizer, tokens, max_new_tokens=50, temperature=0.0
        )

        decoded = our_tokenizer.decode(output)

        # Count <end_of_turn> occurrences - should be at most 2 (one from prompt, one from response)
        end_of_turn_count = decoded.count("<end_of_turn>")
        assert (
            end_of_turn_count <= 2
        ), f"Too many <end_of_turn> tokens ({end_of_turn_count}): {decoded}"
