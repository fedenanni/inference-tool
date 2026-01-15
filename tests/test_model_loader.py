class TestModelLoader:
    """Tests for the model weight loading functionality."""

    # Test that load_weights returns a dictionary
    def test_load_weights_returns_dict(self):
        from inference_tool.model_loader import load_weights

        weights = load_weights("google/gemma-3-270m-it")

        assert isinstance(weights, dict)

    # Test that the weights dictionary is not empty (model has parameters)
    def test_load_weights_not_empty(self):
        from inference_tool.model_loader import load_weights

        weights = load_weights("google/gemma-3-270m-it")

        assert len(weights) > 0

    # Test that all keys in the weights dict are strings (layer names)
    def test_weight_keys_are_strings(self):
        from inference_tool.model_loader import load_weights

        weights = load_weights("google/gemma-3-270m-it")

        assert all(isinstance(key, str) for key in weights.keys())

    # Test that all values are numpy arrays (the actual tensors)
    def test_weight_values_are_numpy_arrays(self):
        import numpy as np
        from inference_tool.model_loader import load_weights

        weights = load_weights("google/gemma-3-270m-it")

        assert all(isinstance(value, np.ndarray) for value in weights.values())

    # Test that embedding layer exists (fundamental for any transformer)
    def test_contains_embedding_layer(self):
        from inference_tool.model_loader import load_weights

        weights = load_weights("google/gemma-3-270m-it")

        embedding_keys = [k for k in weights.keys() if "embed" in k.lower()]
        assert len(embedding_keys) > 0

    # Test that attention layers exist (core of transformer architecture)
    def test_contains_attention_layers(self):
        from inference_tool.model_loader import load_weights

        weights = load_weights("google/gemma-3-270m-it")

        attention_keys = [k for k in weights.keys() if "attn" in k.lower() or "attention" in k.lower()]
        assert len(attention_keys) > 0

    # Test that we can retrieve model config with architecture info
    def test_get_model_config_returns_dict(self):
        from inference_tool.model_loader import get_model_config

        config = get_model_config("google/gemma-3-270m-it")

        assert isinstance(config, dict)

    # Test that config contains essential architecture parameters
    def test_config_contains_hidden_size(self):
        from inference_tool.model_loader import get_model_config

        config = get_model_config("google/gemma-3-270m-it")

        assert "hidden_size" in config

    # Test that config contains number of layers
    def test_config_contains_num_layers(self):
        from inference_tool.model_loader import get_model_config

        config = get_model_config("google/gemma-3-270m-it")

        assert "num_hidden_layers" in config

    # Test that config contains vocabulary size
    def test_config_contains_vocab_size(self):
        from inference_tool.model_loader import get_model_config

        config = get_model_config("google/gemma-3-270m-it")

        assert "vocab_size" in config

    # Test that config contains number of attention heads
    def test_config_contains_num_attention_heads(self):
        from inference_tool.model_loader import get_model_config

        config = get_model_config("google/gemma-3-270m-it")

        assert "num_attention_heads" in config