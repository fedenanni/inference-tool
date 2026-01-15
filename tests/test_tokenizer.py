class TestTokenizer:
    """Tests for the Tokenizer class."""

    # Test that encode() returns a list of integers (token IDs)
    def test_encode_returns_list_of_integers(self):
        from inference_tool.tokenizer import Tokenizer

        tokenizer = Tokenizer("google/gemma-3-270m-it")
        result = tokenizer.encode("Hello world")

        assert isinstance(result, list)
        assert all(isinstance(token_id, int) for token_id in result)

    # Test that encoding non-empty text produces at least one token
    def test_encode_non_empty_string_returns_non_empty_list(self):
        from inference_tool.tokenizer import Tokenizer

        tokenizer = Tokenizer("google/gemma-3-270m-it")
        result = tokenizer.encode("Hello")

        assert len(result) > 0

    # Test that encoding an empty string returns an empty list
    def test_encode_empty_string_returns_empty_list(self):
        from inference_tool.tokenizer import Tokenizer

        tokenizer = Tokenizer("google/gemma-3-270m-it")
        result = tokenizer.encode("")

        assert result == []

    # Test that decode() returns a string
    def test_decode_returns_string(self):
        from inference_tool.tokenizer import Tokenizer

        tokenizer = Tokenizer("google/gemma-3-270m-it")
        token_ids = tokenizer.encode("Hello world")
        result = tokenizer.decode(token_ids)

        assert isinstance(result, str)

    # Test that decoding an empty list returns an empty string
    def test_decode_empty_list_returns_empty_string(self):
        from inference_tool.tokenizer import Tokenizer

        tokenizer = Tokenizer("google/gemma-3-270m-it")
        result = tokenizer.decode([])

        assert result == ""

    # Test that encoding then decoding returns the original text
    def test_encode_decode_roundtrip(self):
        from inference_tool.tokenizer import Tokenizer

        tokenizer = Tokenizer("google/gemma-3-270m-it")
        original_text = "Hello world"

        encoded = tokenizer.encode(original_text)
        decoded = tokenizer.decode(encoded)

        assert decoded == original_text

    # Test that punctuation is preserved through encode/decode roundtrip
    def test_encode_decode_roundtrip_with_punctuation(self):
        from inference_tool.tokenizer import Tokenizer

        tokenizer = Tokenizer("google/gemma-3-270m-it")
        original_text = "Hello, world! How are you?"

        encoded = tokenizer.encode(original_text)
        decoded = tokenizer.decode(encoded)

        assert decoded == original_text

    # Test that numbers and symbols are preserved through encode/decode roundtrip
    def test_encode_decode_roundtrip_with_numbers(self):
        from inference_tool.tokenizer import Tokenizer

        tokenizer = Tokenizer("google/gemma-3-270m-it")
        original_text = "The year is 2026 and the price is $99.99"

        encoded = tokenizer.encode(original_text)
        decoded = tokenizer.decode(encoded)

        assert decoded == original_text

    # Test that different texts produce different token sequences
    def test_different_texts_produce_different_encodings(self):
        from inference_tool.tokenizer import Tokenizer

        tokenizer = Tokenizer("google/gemma-3-270m-it")

        encoded1 = tokenizer.encode("Hello")
        encoded2 = tokenizer.encode("Goodbye")

        assert encoded1 != encoded2

    # Test that the tokenizer exposes BOS (beginning of sequence) token
    def test_has_bos_token(self):
        from inference_tool.tokenizer import Tokenizer

        tokenizer = Tokenizer("google/gemma-3-270m-it")

        assert hasattr(tokenizer, "bos_token_id")
        assert isinstance(tokenizer.bos_token_id, int)

    # Test that the tokenizer exposes EOS (end of sequence) token
    def test_has_eos_token(self):
        from inference_tool.tokenizer import Tokenizer

        tokenizer = Tokenizer("google/gemma-3-270m-it")

        assert hasattr(tokenizer, "eos_token_id")
        assert isinstance(tokenizer.eos_token_id, int)

    # Test that the tokenizer exposes vocabulary size
    def test_has_vocab_size(self):
        from inference_tool.tokenizer import Tokenizer

        tokenizer = Tokenizer("google/gemma-3-270m-it")

        assert hasattr(tokenizer, "vocab_size")
        assert isinstance(tokenizer.vocab_size, int)
        assert tokenizer.vocab_size > 0

    # Test that all token IDs are within valid vocabulary range
    def test_encoded_tokens_within_vocab_range(self):
        from inference_tool.tokenizer import Tokenizer

        tokenizer = Tokenizer("google/gemma-3-270m-it")
        encoded = tokenizer.encode("Hello world, this is a test!")

        assert all(0 <= token_id < tokenizer.vocab_size for token_id in encoded)
