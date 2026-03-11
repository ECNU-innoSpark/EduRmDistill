# Mimic-Kit

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[中文文档](README-ZH.md) | **English**

Mimic-Kit is a **black-box knowledge distillation scaffolding** that helps you distill capabilities from powerful teacher models into smaller, more efficient student models.

## Features

- 🤖 **Teacher Model Generation**: Use OpenAI API (or compatible APIs) to generate high-quality training data
- 🚀 **Efficient Training**: Built on [ms-swift](https://github.com/modelscope/swift) for streamlined model training
- 🧩 **Flexible Tuning**: Support both LoRA (parameter-efficient) and full fine-tuning
- ⚡ **Performance Optimized**: Integrated with DeepSpeed and Liger Kernel for accelerated training
- 💾 **Smart Caching**: Automatically cache API responses to save costs
- 📊 **Multiple Data Formats**: Support both chat and text completion formats

## Quick Start

### 1. Installation

**Option 1: Install from PyPI (Recommended)**

```bash
pip install mimic-kit
```

**Option 2: Install from source**

```bash
# Clone the repository
git clone https://github.com/ECNU-innoSpark/EduRmDistill
cd mimic-kit

# Install dependencies using UV
uv sync

# Or with dev dependencies
uv sync --group dev
```

### 2. Initialize Configuration

```bash
uv run mimic init
```

This creates a `config.yaml` template. Edit it to configure your teacher model, student model, and training parameters.

### 3. Prepare Your Data

Create a JSONL file with your prompts. Supports two formats:

**Chat Format:**
```jsonl
{"messages": [{"role": "user", "content": "Explain Python decorators"}]}
{"messages": [{"role": "user", "content": "How to reverse a linked list?"}]}
```

**Text Completion Format:**
```jsonl
{"text": "Python decorators are a powerful feature..."}
{"text": "To reverse a linked list, you need to..."}
```

### 4. Generate Training Data

```bash
uv run mimic generate
```

This sends your prompts to the teacher model and saves the generated responses as training data.

### 5. Train Your Student Model

```bash
uv run mimic train
```

The student model will be fine-tuned on the generated data using the configured method (LoRA or full fine-tuning).

## Configuration

See `config.yaml` for all available options. Key sections:

- `data`: Input/output paths, system prompts, templates
- `teacher`: API provider, model, generation parameters
- `student`: Base model, tuning method, hyperparameters
- `training`: Batch size, learning rate, saving strategy

## Project Structure

```
mimic-kit/
├── mimic/
│   ├── cli.py              # CLI entry point
│   ├── config.py           # Configuration models (Pydantic v2)
│   ├── generator/          # Teacher model data generation
│   └── trainer/            # Student model training (ms-swift)
├── data/                   # Training data directory
├── config.yaml             # Your configuration file
└── output/                 # Model outputs and checkpoints
```

## Requirements

- Python 3.13+
- CUDA-capable GPU (for training)
- OpenAI API key (or compatible API)

## Development

```bash
# Run tests
uv run pytest

# Run single test
uv run pytest tests/test_cli.py::test_function

# Format code
uv run ruff format .

# Check linting
uv run ruff check .
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [ms-swift](https://github.com/modelscope/swift) for model training
- Uses [Pydantic](https://docs.pydantic.dev/) for configuration validation
- Powered by [Click](https://click.palletsprojects.com/) for CLI
