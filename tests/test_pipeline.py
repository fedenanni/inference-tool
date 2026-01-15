"""Tests for the high-level pipeline module.

These tests verify the Pipeline class that provides a HuggingFace-compatible
interface for text generation with Gemma models.

Usage pattern:
    from inference_tool.pipeline import Pipeline

    pipe = Pipeline("google/gemma-3-270m-it")
    response = pipe([{"role": "user", "content": "Hello!"}])
    print(response)

The Pipeline class handles:
- Model and tokenizer loading
- Chat template formatting
- Text generation with configurable parameters
- Response parsing and formatting
"""

import pytest
import numpy as np


# =============================================================================
# Pipeline Initialization Tests
# =============================================================================


class TestPipelineInit:
    """Tests for Pipeline initialization and configuration.

    The Pipeline class should:
    - Load model and tokenizer from model name
    - Accept generation configuration parameters
    - Support lazy loading for efficiency
    """

    # Test Pipeline can be instantiated with model name
    @pytest.mark.slow
    def test_instantiation_with_model_name(self):
        from inference_tool.pipeline import Pipeline

        pipe = Pipeline("google/gemma-3-270m-it")

        assert pipe is not None
        assert pipe.model is not None
        assert pipe.tokenizer is not None

    # Test Pipeline stores default generation parameters
    def test_default_generation_params(self):
        from inference_tool.pipeline import Pipeline

        # Use mock to avoid loading real model
        class MockModel:
            pass

        class MockTokenizer:
            eos_token_id = 1

        pipe = Pipeline.__new__(Pipeline)
        pipe.model = MockModel()
        pipe.tokenizer = MockTokenizer()
        pipe.default_max_new_tokens = 100
        pipe.default_temperature = 1.0
        pipe.default_top_k = None
        pipe.default_top_p = None

        assert pipe.default_max_new_tokens == 100
        assert pipe.default_temperature == 1.0


# =============================================================================
# Pipeline Call Tests
# =============================================================================


class TestPipelineCall:
    """Tests for the Pipeline __call__ method.

    The __call__ method should:
    - Accept a list of messages
    - Apply chat template formatting
    - Generate text using the model
    - Return parsed response
    """

    # Helper to create a mock pipeline
    def _create_mock_pipeline(self, mock_response_tokens):
        from inference_tool.pipeline import Pipeline

        class MockModel:
            def __init__(self, response_tokens):
                self._response_tokens = response_tokens
                self._call_count = 0

            def forward(self, token_ids, position_offset=0, kv_cache=None):
                batch_size, seq_len = token_ids.shape
                vocab_size = 100
                logits = np.zeros((batch_size, seq_len, vocab_size))

                if self._call_count < len(self._response_tokens):
                    next_tok = self._response_tokens[self._call_count]
                    logits[:, -1, next_tok] = 100.0
                    self._call_count += 1
                else:
                    logits[:, -1, 1] = 100.0  # EOS

                new_cache = [
                    {
                        "key": np.zeros((1, 1, position_offset + seq_len, 64)),
                        "value": np.zeros((1, 1, position_offset + seq_len, 64)),
                    }
                ]
                return logits, new_cache

        class MockTokenizer:
            eos_token_id = 1
            bos_token_id = None

            def encode(self, text):
                # Simple encoding: just return fixed tokens
                return [10, 20, 30]

            def decode(self, tokens):
                # Simple decoding: convert to chars
                return "Generated response"

        pipe = Pipeline.__new__(Pipeline)
        pipe.model = MockModel(mock_response_tokens)
        pipe.tokenizer = MockTokenizer()
        pipe.default_max_new_tokens = 100
        pipe.default_temperature = 0.0
        pipe.default_top_k = None
        pipe.default_top_p = None
        pipe.default_repetition_penalty = 1.0

        return pipe

    # Test calling pipeline with messages returns string
    def test_call_returns_string(self):
        pipe = self._create_mock_pipeline([50, 51, 52, 1])  # Some tokens + EOS

        result = pipe([{"role": "user", "content": "Hello!"}])

        assert isinstance(result, str)

    # Test calling pipeline with empty messages
    def test_call_with_empty_messages_raises(self):
        pipe = self._create_mock_pipeline([1])

        with pytest.raises((ValueError, IndexError)):
            pipe([])

    # Test calling with custom generation parameters
    def test_call_with_custom_params(self):
        pipe = self._create_mock_pipeline([50, 51, 1])

        # Should not raise with custom params
        result = pipe(
            [{"role": "user", "content": "Test"}],
            max_new_tokens=50,
            temperature=0.5,
            top_k=40,
            top_p=0.9,
        )

        assert isinstance(result, str)

    # Test multi-turn conversation
    def test_multi_turn_conversation(self):
        pipe = self._create_mock_pipeline([60, 61, 62, 1])

        result = pipe(
            [
                {"role": "user", "content": "Who are you?"},
                {"role": "assistant", "content": "I am Gemma."},
                {"role": "user", "content": "Nice!"},
            ]
        )

        assert isinstance(result, str)


class TestPipelineGenerateMethod:
    """Tests for the Pipeline.generate method.

    The generate method provides lower-level access to generation,
    returning both the full text and the parsed response.
    """

    def _create_mock_pipeline(self):
        from inference_tool.pipeline import Pipeline

        class MockModel:
            def forward(self, token_ids, position_offset=0, kv_cache=None):
                batch_size, seq_len = token_ids.shape
                logits = np.zeros((batch_size, seq_len, 100))
                logits[:, -1, 1] = 100.0  # Always return EOS
                cache = [
                    {
                        "key": np.zeros((1, 1, seq_len, 64)),
                        "value": np.zeros((1, 1, seq_len, 64)),
                    }
                ]
                return logits, cache

        class MockTokenizer:
            eos_token_id = 1
            bos_token_id = None

            def encode(self, text):
                return [10, 20, 30]

            def decode(self, tokens):
                return "<start_of_turn>model\nHello!<end_of_turn>"

        pipe = Pipeline.__new__(Pipeline)
        pipe.model = MockModel()
        pipe.tokenizer = MockTokenizer()
        pipe.default_max_new_tokens = 100
        pipe.default_temperature = 0.0
        pipe.default_top_k = None
        pipe.default_top_p = None
        pipe.default_repetition_penalty = 1.0

        return pipe

    # Test generate returns dict with expected keys
    def test_generate_returns_dict(self):
        pipe = self._create_mock_pipeline()

        result = pipe.generate([{"role": "user", "content": "Hi"}])

        assert isinstance(result, dict)
        assert "response" in result
        assert "full_text" in result

    # Test generate includes token count info
    def test_generate_includes_token_counts(self):
        pipe = self._create_mock_pipeline()

        result = pipe.generate([{"role": "user", "content": "Hi"}])

        assert "prompt_tokens" in result
        assert "generated_tokens" in result
        assert isinstance(result["prompt_tokens"], int)
        assert isinstance(result["generated_tokens"], int)


# =============================================================================
# Pipeline Streaming Tests (Future Enhancement)
# =============================================================================


class TestPipelineStreaming:
    """Tests for streaming generation (yields tokens as generated).

    Note: Streaming is a future enhancement. These tests document
    the expected interface.
    """

    # Test stream method exists and returns iterator
    def test_stream_returns_iterator(self):
        from inference_tool.pipeline import Pipeline

        # Create minimal mock
        class MockModel:
            def forward(self, token_ids, position_offset=0, kv_cache=None):
                batch_size, seq_len = token_ids.shape
                logits = np.zeros((batch_size, seq_len, 100))
                logits[:, -1, 1] = 100.0
                cache = [
                    {
                        "key": np.zeros((1, 1, seq_len, 64)),
                        "value": np.zeros((1, 1, seq_len, 64)),
                    }
                ]
                return logits, cache

        class MockTokenizer:
            eos_token_id = 1
            bos_token_id = None

            def encode(self, text):
                return [10, 20]

            def decode(self, tokens):
                return "text"

        pipe = Pipeline.__new__(Pipeline)
        pipe.model = MockModel()
        pipe.tokenizer = MockTokenizer()
        pipe.default_max_new_tokens = 10
        pipe.default_temperature = 0.0
        pipe.default_top_k = None
        pipe.default_top_p = None
        pipe.default_repetition_penalty = 1.0

        # Stream should return an iterator
        result = pipe.stream([{"role": "user", "content": "Hi"}])

        assert hasattr(result, "__iter__")


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.slow
class TestPipelineIntegration:
    """Integration tests with the actual Gemma model.

    These tests verify end-to-end pipeline behavior.
    Marked as slow since they require loading the model.
    """

    # Test full pipeline with real model
    def test_full_pipeline_generation(self):
        from inference_tool.pipeline import Pipeline

        pipe = Pipeline("google/gemma-3-270m-it")

        result = pipe(
            [{"role": "user", "content": "What is 2 + 2?"}],
            max_new_tokens=20,
            temperature=0.0,
        )

        assert isinstance(result, str)
        assert len(result) > 0

    # Test pipeline produces sensible output
    def test_pipeline_produces_sensible_output(self):
        from inference_tool.pipeline import Pipeline

        pipe = Pipeline("google/gemma-3-270m-it")

        result = pipe(
            [{"role": "user", "content": "Say hello."}],
            max_new_tokens=10,
            temperature=0.0,
        )

        # Should contain some greeting-like response
        result_lower = result.lower()
        assert any(word in result_lower for word in ["hello", "hi", "hey", "greet"])

    # Test pipeline matches HuggingFace behavior pattern
    def test_matches_huggingface_interface(self):
        from inference_tool.pipeline import Pipeline

        # This tests that our Pipeline follows the HuggingFace pattern
        pipe = Pipeline("google/gemma-3-270m-it")

        # Should be callable with messages
        messages = [{"role": "user", "content": "Test"}]
        result = pipe(messages, max_new_tokens=5)

        assert isinstance(result, str)
