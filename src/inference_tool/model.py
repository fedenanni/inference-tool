"""Gemma model implementation using NumPy.

This module contains the full Gemma transformer model for text generation:
- TransformerBlock: Single transformer layer combining attention + MLP
- GemmaModel: Complete model with embedding, transformer stack, and output head

The implementation supports KV caching for efficient autoregressive generation.

Usage:
    from model import GemmaModel
    from tokenizer import Tokenizer

    model = GemmaModel("google/gemma-3-270m-it")
    tokenizer = Tokenizer("google/gemma-3-270m-it")

    tokens = tokenizer.encode("Hello, world!")
    token_ids = np.array([tokens], dtype=np.int32)
    logits, kv_cache = model.forward(token_ids)
"""

import numpy as np
from typing import Optional

from .layers import RMSNorm, MultiHeadAttention, MLP
from .model_loader import load_weights, get_model_config


class TransformerBlock:
    """Single transformer block with attention, MLP, and residual connections.

    Architecture (Gemma3-specific):
        x -> input_layernorm -> Attention -> post_attention_layernorm -> + x
          -> pre_feedforward_layernorm -> MLP -> post_feedforward_layernorm -> + x

    Gemma3 has 4 LayerNorms per block, with post-norms applied before residuals.
    """

    def __init__(self, config: dict, layer_idx: int = 0):
        """Initialize a transformer block.

        Args:
            config: Model configuration dictionary containing:
                - hidden_size: Model hidden dimension
                - num_attention_heads: Number of query heads
                - num_key_value_heads: Number of KV heads (for GQA)
                - head_dim: Dimension per attention head
                - intermediate_size: MLP intermediate dimension
                - rms_norm_eps: Epsilon for RMSNorm
            layer_idx: Layer index (used to determine sliding window pattern)
        """
        self.config = config
        self.layer_idx = layer_idx
        hidden_size = config["hidden_size"]
        eps = config.get("rms_norm_eps", 1e-6)

        # Determine if this is a sliding window layer
        # Gemma3: every Nth layer is global attention (N = _sliding_window_pattern)
        sliding_window_pattern = config.get("_sliding_window_pattern", 6)
        self.is_sliding = ((layer_idx + 1) % sliding_window_pattern) != 0

        # Use different RoPE base for sliding vs global attention
        # sliding (local): rope_local_base_freq (default 10000)
        # global: rope_theta (default 1000000)
        if self.is_sliding:
            rope_base = config.get("rope_local_base_freq", 10000.0)
        else:
            rope_base = config.get("rope_theta", 1000000.0)

        self.sliding_window = (
            config.get("sliding_window", 512) if self.is_sliding else None
        )

        # Pre-attention normalization
        self.input_layernorm = RMSNorm(hidden_size, eps)

        # Multi-head attention with Grouped Query Attention
        self.self_attn = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=config["num_attention_heads"],
            num_kv_heads=config["num_key_value_heads"],
            head_dim=config["head_dim"],
            query_pre_attn_scalar=config.get("query_pre_attn_scalar"),
            rope_base=rope_base,
            sliding_window=self.sliding_window,
        )

        # Post-attention normalization (applied to attn output before residual)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps)

        # Pre-feedforward normalization
        self.pre_feedforward_layernorm = RMSNorm(hidden_size, eps)

        # MLP with GEGLU activation
        self.mlp = MLP(
            hidden_size=hidden_size,
            intermediate_size=config["intermediate_size"],
        )

        # Post-feedforward normalization (applied to MLP output before residual)
        self.post_feedforward_layernorm = RMSNorm(hidden_size, eps)

    def __call__(
        self,
        x: np.ndarray,
        start_pos: int = 0,
        kv_cache: Optional[dict[str, np.ndarray]] = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Forward pass through the transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            start_pos: Starting position for RoPE (for KV cache)
            kv_cache: Optional cached key-value tensors from previous passes

        Returns:
            Tuple of:
            - Output tensor of shape (batch_size, seq_len, hidden_size)
            - Updated KV cache dictionary
        """
        residual = x

        # Pre-attention normalization
        hidden_states = self.input_layernorm(x)

        # Self-attention
        hidden_states, new_kv_cache = self.self_attn(
            hidden_states, start_pos=start_pos, kv_cache=kv_cache
        )

        # Post-attention normalization (before residual!)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states

        # Pre-feedforward normalization
        hidden_states = self.pre_feedforward_layernorm(hidden_states)

        # MLP
        hidden_states = self.mlp(hidden_states)

        # Post-feedforward normalization (before residual!)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_kv_cache

    def load_weights(self, weights: dict[str, np.ndarray], layer_idx: int) -> None:
        """Load weights for this transformer block from the weights dictionary.

        Args:
            weights: Full model weights dictionary
            layer_idx: Index of this layer (for weight key lookup)
        """
        prefix = f"model.layers.{layer_idx}"

        # Load RMSNorm weights (Gemma3 has 4 LayerNorms per block)
        self.input_layernorm.weight = weights[f"{prefix}.input_layernorm.weight"]
        self.post_attention_layernorm.weight = weights[
            f"{prefix}.post_attention_layernorm.weight"
        ]
        self.pre_feedforward_layernorm.weight = weights[
            f"{prefix}.pre_feedforward_layernorm.weight"
        ]
        self.post_feedforward_layernorm.weight = weights[
            f"{prefix}.post_feedforward_layernorm.weight"
        ]

        # Load attention weights
        self.self_attn.q_proj = weights[f"{prefix}.self_attn.q_proj.weight"].T
        self.self_attn.k_proj = weights[f"{prefix}.self_attn.k_proj.weight"].T
        self.self_attn.v_proj = weights[f"{prefix}.self_attn.v_proj.weight"].T
        self.self_attn.o_proj = weights[f"{prefix}.self_attn.o_proj.weight"].T

        # Load Q/K normalization weights (Gemma3-specific)
        self.self_attn.q_norm_weight = weights[f"{prefix}.self_attn.q_norm.weight"]
        self.self_attn.k_norm_weight = weights[f"{prefix}.self_attn.k_norm.weight"]

        # Load MLP weights
        self.mlp.gate_proj = weights[f"{prefix}.mlp.gate_proj.weight"].T
        self.mlp.up_proj = weights[f"{prefix}.mlp.up_proj.weight"].T
        self.mlp.down_proj = weights[f"{prefix}.mlp.down_proj.weight"].T


class GemmaModel:
    """Complete Gemma language model for text generation.

    Architecture:
        Token IDs -> Embedding (scaled) -> TransformerBlocks -> FinalNorm -> LM Head -> Logits

    Supports KV caching for efficient autoregressive generation.
    """

    def __init__(self, model_name: str):
        """Initialize the Gemma model by loading config and weights.

        Args:
            model_name: Hugging Face model name (e.g., "google/gemma-3-270m-it")
        """
        self.model_name = model_name

        # Load configuration
        self.config = get_model_config(model_name)

        # Extract key config values
        self.hidden_size = self.config["hidden_size"]
        self.num_layers = self.config["num_hidden_layers"]
        self.vocab_size = self.config["vocab_size"]

        # Build model layers
        self._build_layers()

        # Load weights from Hugging Face
        self._load_weights()

    def _build_layers(self) -> None:
        """Build all model layers based on config."""
        eps = self.config.get("rms_norm_eps", 1e-6)

        # Token embedding (will be loaded from weights)
        self.embed_tokens = np.zeros(
            (self.vocab_size, self.hidden_size), dtype=np.float32
        )

        # Transformer blocks
        self.layers = []
        for i in range(self.num_layers):
            block = TransformerBlock(self.config, layer_idx=i)
            self.layers.append(block)

        # Final normalization before output projection
        self.norm = RMSNorm(self.hidden_size, eps)

        # LM head for projecting to vocabulary (may be tied to embeddings)
        self.lm_head = None  # Will check if separate weights exist

    def _load_weights(self) -> None:
        """Load all weights from the Hugging Face model."""
        weights = load_weights(self.model_name)

        # Load embedding weights
        self.embed_tokens = weights["model.embed_tokens.weight"]

        # Load transformer block weights
        for i, layer in enumerate(self.layers):
            layer.load_weights(weights, i)

        # Load final normalization
        self.norm.weight = weights["model.norm.weight"]

        # Load LM head (might be tied to embeddings in some models)
        if "lm_head.weight" in weights:
            # Transpose: (vocab_size, hidden_size) -> (hidden_size, vocab_size)
            self.lm_head = weights["lm_head.weight"].T
        else:
            # Tied embeddings: use embedding matrix transposed
            self.lm_head = self.embed_tokens.T

    def embed(self, token_ids: np.ndarray) -> np.ndarray:
        """Look up token embeddings and apply scaling.

        Gemma scales embeddings by sqrt(hidden_size) to ensure proper
        gradient flow during training (we keep this for inference consistency).

        Args:
            token_ids: Token IDs of shape (batch_size, seq_len)

        Returns:
            Embeddings of shape (batch_size, seq_len, hidden_size)
        """
        # Look up embeddings
        embeddings = self.embed_tokens[token_ids]

        # Scale by sqrt(hidden_size) - Gemma-specific
        scale = np.sqrt(self.hidden_size)
        embeddings = embeddings * scale

        return embeddings.astype(np.float32)

    def forward(
        self,
        token_ids: np.ndarray,
        position_offset: int = 0,
        kv_cache: Optional[list[dict[str, np.ndarray]]] = None,
    ) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
        """Forward pass through the complete model.

        Args:
            token_ids: Token IDs of shape (batch_size, seq_len)
            position_offset: Starting position for RoPE (used with KV cache)
            kv_cache: Optional list of KV caches for each layer

        Returns:
            Tuple of:
            - Logits of shape (batch_size, seq_len, vocab_size)
            - Updated KV cache (list of dicts, one per layer)
        """
        # Get embeddings
        hidden_states = self.embed(token_ids)

        # Initialize or use provided KV cache
        if kv_cache is None:
            kv_cache = [None] * self.num_layers

        # Process through transformer blocks
        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            hidden_states, layer_kv_cache = layer(
                hidden_states,
                start_pos=position_offset,
                kv_cache=kv_cache[i],
            )
            new_kv_cache.append(layer_kv_cache)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        # Project to vocabulary logits
        logits = hidden_states @ self.lm_head

        return logits, new_kv_cache
