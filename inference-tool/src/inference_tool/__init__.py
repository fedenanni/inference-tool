"""Inference Tool - A NumPy-based Gemma model inference engine.

This package provides a lightweight implementation of Gemma models
for text generation, with a HuggingFace-compatible Pipeline interface.

Example:
    from inference_tool import Pipeline

    pipe = Pipeline("google/gemma-3-270m-it")
    response = pipe([{"role": "user", "content": "Hello!"}])
    print(response)
"""

from inference_tool.pipeline import Pipeline
from inference_tool.model import GemmaModel
from inference_tool.tokenizer import Tokenizer
from inference_tool.generate import generate
from inference_tool.chat import format_messages, parse_response

__all__ = [
    "Pipeline",
    "GemmaModel",
    "Tokenizer",
    "generate",
    "format_messages",
    "parse_response",
]

__version__ = "0.1.0"
