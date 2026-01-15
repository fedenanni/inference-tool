# Inference Tool

A lightweight NumPy-based inference engine for Google's Gemma language models developed by Claude Code, together with me. This project provides a from-scratch implementation of Gemma model inference with a HuggingFace-compatible API.

## Installation

```bash
# Clone the repository
git clone https://github.com/fedenanni/inference-tool.git
cd inference-tool

# Install with uv (recommended)
uv sync
uv pip install -e .

# Or with pip
pip install -e .
```

## Running the Notebook

The `inference.ipynb` notebook demonstrates how to use the inference engine and compare it with HuggingFace's implementation.

1. Make sure you've installed the package in editable mode (see Installation above)
2. Open `inference.ipynb` in VS Code or Jupyter
3. Select the Python interpreter from the `.venv` virtual environment
4. Run the cells to see the model in action

## Quick Start

### Using the Pipeline (Recommended)

```python
from inference_tool import Pipeline

# Initialize the pipeline
pipe = Pipeline("google/gemma-3-270m-it")

# Generate a response
response = pipe([{"role": "user", "content": "Who are you?"}])
print(response)

# With custom parameters
response = pipe(
    [{"role": "user", "content": "Write a haiku about coding"}],
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9,
)
```

### Streaming Output

```python
from inference_tool import Pipeline

pipe = Pipeline("google/gemma-3-270m-it")

for token in pipe.stream([{"role": "user", "content": "Tell me a story"}]):
    print(token, end="", flush=True)
```

### Low-Level API

```python
from inference_tool import GemmaModel, Tokenizer, generate, format_messages

# Load model and tokenizer
model = GemmaModel("google/gemma-3-270m-it")
tokenizer = Tokenizer("google/gemma-3-270m-it")

# Format and encode the prompt
prompt = format_messages([{"role": "user", "content": "Hello!"}])
tokens = tokenizer.encode(prompt)

# Generate
output = generate(
    model=model,
    tokenizer=tokenizer,
    prompt_tokens=tokens,
    max_new_tokens=50,
    temperature=0.7,
)

print(tokenizer.decode(output))
```

## Project Structure

```
inference-tool/
├── src/inference_tool/      # Main package
│   ├── __init__.py          # Package exports
│   ├── chat.py              # Chat template formatting
│   ├── generate.py          # Text generation engine
│   ├── layers.py            # Transformer layer implementations
│   ├── model.py             # GemmaModel class
│   ├── model_loader.py      # Weight loading utilities
│   ├── pipeline.py          # High-level Pipeline API
│   └── tokenizer.py         # Tokenizer wrapper
├── tests/                   # Test suite
├── inference.ipynb          # Example notebook
├── pyproject.toml           # Project configuration
└── README.md
```

## Supported Models

- `google/gemma-3-270m-it` (Gemma 3 270M Instruct)

Other Gemma variants may work but are not officially tested.

## Generation Parameters

| Parameter            | Default | Description                                    |
| -------------------- | ------- | ---------------------------------------------- |
| `max_new_tokens`     | 100     | Maximum number of tokens to generate           |
| `temperature`        | 1.0     | Sampling temperature (0 = greedy)              |
| `top_k`              | None    | Top-k filtering (keep k most likely tokens)    |
| `top_p`              | None    | Nucleus sampling threshold                     |
| `repetition_penalty` | 1.0     | Penalty for repeated tokens (>1.0 discourages) |

## Running Tests

```bash
uv run pytest
```

To skip slow integration tests:

```bash
uv run pytest -m "not slow"
```

## Requirements

- Python >= 3.13
- NumPy
- HuggingFace Hub (for model downloads)
- SentencePiece (for tokenization)
- PyTorch (for loading safetensors weights and converting bfloat16 → float32)
- Transformers (optional, for running comparison tests and the demo notebook)

## License

MIT

## Acknowledgments

This project is an educational implementation inspired by Google's Gemma models and the HuggingFace Transformers library.
