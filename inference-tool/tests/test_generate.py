"""Tests for the text generation module.

These tests verify the generation engine that produces text autoregressively:
- Sampling strategies: greedy, temperature scaling, top-k, top-p (nucleus)
- KV cache usage for efficient generation
- Stopping conditions: EOS token, max length

The generate function signature:
    generate(model, tokenizer, prompt_tokens, max_new_tokens, temperature, top_k, top_p)

Reference behavior matches HuggingFace transformers generation.
"""

import numpy as np
import pytest


# =============================================================================
# Sampling Function Tests
# =============================================================================


class TestSampleToken:
    """Tests for the token sampling function.

    The sample_token function takes logits and sampling parameters,
    returning the next token ID. It supports:
    - Greedy decoding (temperature=0 or argmax)
    - Temperature scaling (higher = more random)
    - Top-k filtering (only consider k most likely tokens)
    - Top-p (nucleus) sampling (consider tokens until cumulative prob >= p)
    """

    # Test greedy sampling returns the token with highest logit
    def test_greedy_sampling_returns_argmax(self):
        from inference_tool.generate import sample_token

        # Logits where token 5 has the highest value
        logits = np.array([1.0, 2.0, 0.5, 3.0, 1.5, 10.0, 2.5])

        token_id = sample_token(logits, temperature=0.0)

        assert token_id == 5, "Greedy sampling should return argmax"

    # Test temperature=1.0 doesn't modify the distribution shape
    def test_temperature_one_preserves_distribution(self):
        from inference_tool.generate import sample_token

        np.random.seed(42)
        logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # With temperature=1.0, probabilities should follow softmax(logits)
        # Sample many times and check distribution roughly matches
        samples = [sample_token(logits, temperature=1.0) for _ in range(1000)]

        # Token 4 (highest logit) should be sampled most often
        counts = np.bincount(samples, minlength=5)
        assert counts[4] > counts[0], "Higher logit tokens should be sampled more"

    # Test high temperature increases randomness (flattens distribution)
    def test_high_temperature_increases_randomness(self):
        from inference_tool.generate import sample_token

        np.random.seed(42)
        logits = np.array([0.0, 0.0, 0.0, 0.0, 10.0])

        # Low temperature: should almost always pick token 4
        low_temp_samples = [sample_token(logits, temperature=0.1) for _ in range(100)]
        low_temp_diversity = len(set(low_temp_samples))

        # High temperature: should pick various tokens
        high_temp_samples = [sample_token(logits, temperature=5.0) for _ in range(100)]
        high_temp_diversity = len(set(high_temp_samples))

        assert (
            high_temp_diversity > low_temp_diversity
        ), "High temperature should produce more diverse samples"

    # Test top-k sampling only considers k most likely tokens
    def test_top_k_filters_to_k_tokens(self):
        from inference_tool.generate import sample_token

        np.random.seed(42)
        # Logits: tokens 3, 4, 5 have the highest values
        logits = np.array([0.0, 0.0, 0.0, 5.0, 6.0, 7.0, 0.0])

        # With top_k=3, only tokens 3, 4, 5 should be sampled
        samples = [sample_token(logits, temperature=1.0, top_k=3) for _ in range(100)]

        assert all(
            s in [3, 4, 5] for s in samples
        ), "Top-k sampling should only select from top k tokens"

    # Test top-k=1 is equivalent to greedy
    def test_top_k_one_equals_greedy(self):
        from inference_tool.generate import sample_token

        logits = np.array([1.0, 5.0, 2.0, 3.0])

        token_id = sample_token(logits, temperature=1.0, top_k=1)

        assert token_id == 1, "Top-k=1 should be equivalent to greedy"

    # Test top-p (nucleus) sampling considers tokens until cumulative prob >= p
    def test_top_p_filters_by_cumulative_probability(self):
        from inference_tool.generate import sample_token

        np.random.seed(42)
        # Create logits where softmax gives clear probability ordering
        # Token 0: ~0.7, Token 1: ~0.2, Token 2: ~0.07, Token 3: ~0.03
        logits = np.array([3.0, 1.0, 0.0, -1.0])

        # With top_p=0.9, tokens 0 and 1 should cover it (0.7 + 0.2 = 0.9)
        samples = [sample_token(logits, temperature=1.0, top_p=0.9) for _ in range(100)]

        # Most samples should be from tokens 0 and 1
        top_2_count = sum(1 for s in samples if s in [0, 1])
        assert (
            top_2_count >= 95
        ), "Top-p=0.9 should mostly sample from tokens covering 90% probability"

    # Test top-p=1.0 considers all tokens
    def test_top_p_one_considers_all_tokens(self):
        from inference_tool.generate import sample_token

        np.random.seed(42)
        # Uniform logits
        logits = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        samples = [sample_token(logits, temperature=1.0, top_p=1.0) for _ in range(200)]

        # All tokens should be sampled at least once
        assert len(set(samples)) == 5, "Top-p=1.0 should allow all tokens"

    # Test combining top-k and top-p (top-k applied first)
    def test_combined_top_k_and_top_p(self):
        from inference_tool.generate import sample_token

        np.random.seed(42)
        # 7 tokens, highest are at indices 4, 5, 6
        logits = np.array([0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 5.0])

        # top_k=3 limits to [4,5,6], then top_p further filters
        samples = [
            sample_token(logits, temperature=1.0, top_k=3, top_p=0.9)
            for _ in range(100)
        ]

        assert all(
            s in [4, 5, 6] for s in samples
        ), "Combined top-k and top-p should respect both constraints"


class TestApplyRepetitionPenalty:
    """Tests for repetition penalty to reduce token repetition.

    Repetition penalty modifies logits to discourage recently generated tokens.
    penalty > 1.0 reduces probability of repeated tokens.
    """

    # Test that repetition penalty reduces logits for previous tokens
    def test_penalty_reduces_repeated_token_logits(self):
        from inference_tool.generate import apply_repetition_penalty

        logits = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        previous_tokens = [1, 3]  # Tokens at indices 1 and 3 were used

        penalized = apply_repetition_penalty(
            logits.copy(), previous_tokens, penalty=1.2
        )

        # Tokens 1 and 3 should have lower logits
        assert penalized[1] < logits[1], "Repeated token logits should decrease"
        assert penalized[3] < logits[3], "Repeated token logits should decrease"
        # Other tokens unchanged
        assert penalized[0] == logits[0]
        assert penalized[2] == logits[2]
        assert penalized[4] == logits[4]

    # Test penalty=1.0 leaves logits unchanged
    def test_penalty_one_no_change(self):
        from inference_tool.generate import apply_repetition_penalty

        logits = np.array([2.0, 3.0, 4.0])
        previous_tokens = [0, 1, 2]

        penalized = apply_repetition_penalty(
            logits.copy(), previous_tokens, penalty=1.0
        )

        np.testing.assert_array_equal(penalized, logits)


# =============================================================================
# Generate Function Tests
# =============================================================================


class TestGenerate:
    """Tests for the main generate function.

    The generate function performs autoregressive text generation:
    1. Process prompt tokens through the model (prefill)
    2. Sample next token from output logits
    3. Append token and repeat (decode phase with KV cache)
    4. Stop on EOS token or max_new_tokens
    """

    # Test generate returns a list of token IDs
    def test_returns_token_list(self):
        from inference_tool.generate import generate

        # Create a mock model for testing
        class MockModel:
            def forward(self, token_ids, position_offset=0, kv_cache=None):
                batch_size, seq_len = token_ids.shape
                vocab_size = 100
                # Return random logits and empty cache
                logits = np.random.randn(batch_size, seq_len, vocab_size)
                new_cache = [
                    {"key": np.zeros((1, 1, 1, 64)), "value": np.zeros((1, 1, 1, 64))}
                ]
                return logits, new_cache

        class MockTokenizer:
            eos_token_id = 99
            bos_token_id = None

        model = MockModel()
        tokenizer = MockTokenizer()
        prompt_tokens = [1, 2, 3]  # Some prompt tokens

        result = generate(
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            max_new_tokens=5,
            temperature=1.0,
        )

        assert isinstance(result, list), "Generate should return a list"
        assert all(
            isinstance(t, (int, np.integer)) for t in result
        ), "All elements should be integers"

    # Test that output starts with prompt tokens
    def test_output_starts_with_prompt(self):
        from inference_tool.generate import generate

        class MockModel:
            def forward(self, token_ids, position_offset=0, kv_cache=None):
                batch_size, seq_len = token_ids.shape
                vocab_size = 100
                logits = np.random.randn(batch_size, seq_len, vocab_size)
                # Make token 42 most likely
                logits[:, :, 42] = 100.0
                new_cache = [
                    {
                        "key": np.zeros((1, 1, seq_len, 64)),
                        "value": np.zeros((1, 1, seq_len, 64)),
                    }
                ]
                return logits, new_cache

        class MockTokenizer:
            eos_token_id = 99
            bos_token_id = None

        model = MockModel()
        tokenizer = MockTokenizer()
        prompt_tokens = [10, 20, 30]

        result = generate(
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            max_new_tokens=3,
            temperature=0.0,  # Greedy
        )

        assert result[:3] == prompt_tokens, "Output should start with prompt tokens"

    # Test generation stops at EOS token
    def test_stops_at_eos_token(self):
        from inference_tool.generate import generate

        call_count = [0]

        class MockModel:
            def forward(self, token_ids, position_offset=0, kv_cache=None):
                call_count[0] += 1
                batch_size, seq_len = token_ids.shape
                vocab_size = 100
                logits = np.zeros((batch_size, seq_len, vocab_size))
                # First call: return normal token, second call: return EOS
                if call_count[0] == 1:
                    logits[:, -1, 50] = 100.0  # Normal token
                else:
                    logits[:, -1, 99] = 100.0  # EOS token
                new_cache = [
                    {
                        "key": np.zeros((1, 1, position_offset + seq_len, 64)),
                        "value": np.zeros((1, 1, position_offset + seq_len, 64)),
                    }
                ]
                return logits, new_cache

        class MockTokenizer:
            eos_token_id = 99
            bos_token_id = None

        model = MockModel()
        tokenizer = MockTokenizer()

        result = generate(
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=[1, 2],
            max_new_tokens=10,
            temperature=0.0,
        )

        # Should stop after generating EOS (prompt + 2 new tokens)
        assert len(result) == 4, f"Should stop at EOS, got {len(result)} tokens"
        assert result[-1] == 99, "Last token should be EOS"

    # Test generation respects max_new_tokens limit
    def test_respects_max_new_tokens(self):
        from inference_tool.generate import generate

        class MockModel:
            def forward(self, token_ids, position_offset=0, kv_cache=None):
                batch_size, seq_len = token_ids.shape
                vocab_size = 100
                logits = np.zeros((batch_size, seq_len, vocab_size))
                logits[:, -1, 42] = 100.0  # Always return token 42
                new_cache = [
                    {
                        "key": np.zeros((1, 1, position_offset + seq_len, 64)),
                        "value": np.zeros((1, 1, position_offset + seq_len, 64)),
                    }
                ]
                return logits, new_cache

        class MockTokenizer:
            eos_token_id = 99  # Never generated
            bos_token_id = None

        model = MockModel()
        tokenizer = MockTokenizer()

        result = generate(
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=[1, 2, 3],
            max_new_tokens=5,
            temperature=0.0,
        )

        assert len(result) == 8, f"Should have prompt(3) + max_new_tokens(5) = 8"

    # Test KV cache is used for efficiency (model called with single tokens after prefill)
    def test_uses_kv_cache_for_efficiency(self):
        from inference_tool.generate import generate

        token_input_sizes = []

        class MockModel:
            def forward(self, token_ids, position_offset=0, kv_cache=None):
                token_input_sizes.append(token_ids.shape[1])
                batch_size, seq_len = token_ids.shape
                vocab_size = 100
                logits = np.zeros((batch_size, seq_len, vocab_size))
                logits[:, -1, 42] = 100.0
                new_cache = [
                    {
                        "key": np.zeros((1, 1, position_offset + seq_len, 64)),
                        "value": np.zeros((1, 1, position_offset + seq_len, 64)),
                    }
                ]
                return logits, new_cache

        class MockTokenizer:
            eos_token_id = 99
            bos_token_id = None

        model = MockModel()
        tokenizer = MockTokenizer()

        generate(
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=[1, 2, 3, 4, 5],  # 5 token prompt
            max_new_tokens=3,
            temperature=0.0,
        )

        # First call should process full prompt (5 tokens)
        assert token_input_sizes[0] == 5, "First call should process full prompt"
        # Subsequent calls should process single tokens (using KV cache)
        assert all(
            size == 1 for size in token_input_sizes[1:]
        ), "Subsequent calls should process single tokens with KV cache"

    # Test greedy generation is deterministic
    def test_greedy_is_deterministic(self):
        from inference_tool.generate import generate

        class MockModel:
            def forward(self, token_ids, position_offset=0, kv_cache=None):
                batch_size, seq_len = token_ids.shape
                vocab_size = 100
                # Deterministic logits based on input
                logits = np.zeros((batch_size, seq_len, vocab_size))
                next_token = (token_ids[0, -1].item() + 7) % vocab_size
                logits[:, -1, next_token] = 100.0
                new_cache = [
                    {
                        "key": np.zeros((1, 1, position_offset + seq_len, 64)),
                        "value": np.zeros((1, 1, position_offset + seq_len, 64)),
                    }
                ]
                return logits, new_cache

        class MockTokenizer:
            eos_token_id = 99
            bos_token_id = None

        model = MockModel()
        tokenizer = MockTokenizer()

        result1 = generate(
            model, tokenizer, [1, 2, 3], max_new_tokens=5, temperature=0.0
        )
        result2 = generate(
            model, tokenizer, [1, 2, 3], max_new_tokens=5, temperature=0.0
        )

        assert result1 == result2, "Greedy generation should be deterministic"


# =============================================================================
# Integration Tests (require actual model - marked slow)
# =============================================================================


@pytest.mark.slow
class TestGenerateIntegration:
    """Integration tests with the actual Gemma model.

    These tests verify end-to-end generation behavior.
    Marked as slow since they require loading the model.
    """

    # Test generating a few tokens from a simple prompt
    def test_generates_coherent_continuation(self):
        from inference_tool.generate import generate
        from inference_tool.model import GemmaModel
        from inference_tool.tokenizer import Tokenizer

        model = GemmaModel("google/gemma-3-270m-it")
        tokenizer = Tokenizer("google/gemma-3-270m-it")

        prompt = "The capital of France is"
        prompt_tokens = tokenizer.encode(prompt)

        result = generate(
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            max_new_tokens=10,
            temperature=0.0,
        )

        output_text = tokenizer.decode(result)
        assert "Paris" in output_text, f"Expected 'Paris' in output: {output_text}"

    # Test that generation with temperature produces varied outputs
    def test_temperature_produces_variety(self):
        from inference_tool.generate import generate
        from inference_tool.model import GemmaModel
        from inference_tool.tokenizer import Tokenizer

        model = GemmaModel("google/gemma-3-270m-it")
        tokenizer = Tokenizer("google/gemma-3-270m-it")

        prompt = "Once upon a time"
        prompt_tokens = tokenizer.encode(prompt)

        # Generate multiple times with temperature
        results = set()
        for _ in range(5):
            result = generate(
                model=model,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
                max_new_tokens=10,
                temperature=0.8,
            )
            results.add(tuple(result))

        # Should get at least some variety
        assert len(results) >= 2, "Temperature sampling should produce varied outputs"
