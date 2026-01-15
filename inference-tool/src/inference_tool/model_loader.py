"""Model loader for downloading and loading model weights from Hugging Face."""

import json

import numpy as np
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import load_file


def load_weights(model_name: str) -> dict[str, np.ndarray]:
    """Load model weights from a Hugging Face model.

    Downloads the safetensors files and loads all weights as numpy arrays.
    Handles bfloat16 tensors by converting them to float32.

    Args:
        model_name: The Hugging Face model name (e.g., "google/gemma-3-270m-it")

    Returns:
        A dictionary mapping layer names (strings) to weight tensors (numpy arrays).
    """
    # List all files in the repo to find safetensor files
    repo_files = list_repo_files(repo_id=model_name)
    safetensor_files = [f for f in repo_files if f.endswith(".safetensors")]

    weights: dict[str, np.ndarray] = {}

    for filename in safetensor_files:
        # Download each safetensors file
        local_path = hf_hub_download(repo_id=model_name, filename=filename)

        # Load tensors using PyTorch (handles bfloat16)
        tensors = load_file(local_path)
        for key, tensor in tensors.items():
            # Convert bfloat16 to float32 since numpy doesn't support bfloat16
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            weights[key] = tensor.numpy()

    return weights


def get_model_config(model_name: str) -> dict:
    """Get the model configuration from Hugging Face.

    Downloads and parses the config.json file containing architecture parameters.

    Args:
        model_name: The Hugging Face model name (e.g., "google/gemma-3-270m-it")

    Returns:
        A dictionary containing model configuration parameters like hidden_size,
        num_hidden_layers, vocab_size, num_attention_heads, etc.
    """
    config_path = hf_hub_download(repo_id=model_name, filename="config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    return config
