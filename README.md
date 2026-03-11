<div align="center">
  <img src="assets/logo.svg" alt="Mimic-Kit Logo" width="150">
  <h1>Mimic-Kit</h1>
  <p><strong>Black-Box Knowledge Distillation Made Simple</strong></p>

  [![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
</div>

---

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

## Configuration Example

```yaml
# Data
data:
  input_path: "./data/prompts.jsonl"
  dataset_path: "./data/distilled_data.jsonl"
  system_prompt: "You are a helpful assistant."

# Teacher (Black-Box API)
teacher:
  provider: "openai"
  model: "gpt-4o"
  api_key: "sk-..."
  base_url: "https://api.openai.com/v1"
  generation_params:
    temperature: 0.7
    max_tokens: 2048

# Student (Open Source LLM)
student:
  base_model: "Qwen/Qwen2.5-1.5B-Instruct"
  tuner_type: "lora"  # or "full"
  lora_config:
    rank: 8
    alpha: 32

# Training
training:
  epochs: 3
  per_device_train_batch_size: 4
  learning_rate:
    initial: 1e-4
  saving:
    output_dir: "./outputs"
```

See `config.yaml` for all available options.

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

## CLI Commands

| Command | Description |
|---------|-------------|
| `mimic init` | Create `config.yaml` template |
| `mimic generate` | Generate training data from teacher model |
| `mimic train` | Train student model with ms-swift |

## Requirements

- Python 3.13+
- CUDA-capable GPU (for training)
- OpenAI API key (or compatible API)

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Format code
uv run ruff format .

# Check linting
uv run ruff check . --fix

# Type checking
uv run mypy mimic/
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [ms-swift](https://github.com/modelscope/swift) for model training
- Uses [Pydantic](https://docs.pydantic.dev/) for configuration validation
- Powered by [Click](https://click.palletsprojects.com/) for CLI
