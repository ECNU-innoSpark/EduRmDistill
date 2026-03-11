# AGENTS.md

Guidelines for AI agents working on the mimic-kit codebase.

## Commands

### Setup
```bash
uv sync                    # Install dependencies
uv sync --group dev       # With dev dependencies
```

### Run
```bash
uv run mimic --help       # Show help
uv run mimic init         # Create config.yaml template
uv run mimic generate     # Generate training data
uv run mimic train        # Train student model with ms-swift
```

### Testing
```bash
# Run all tests (create tests/ directory first)
uv run pytest

# Run single test file
uv run pytest tests/test_cli.py

# Run specific test function
uv run pytest tests/test_cli.py::test_hello

# With verbose output
uv run pytest -v

# With coverage (requires pytest-cov)
uv run pytest --cov=mimic --cov-report=term-missing
```

### Lint & Format
```bash
# Add to pyproject.toml dev dependencies first:
# [dependency-groups]
# dev = ["ruff>=0.1.0", "pytest>=7.0.0", "pytest-cov>=4.0.0", "mypy>=1.0.0"]

uv run ruff format .      # Format code
uv run ruff check .       # Check linting
uv run ruff check . --fix # Fix auto-fixable issues
uv run ruff check . --select I  # Check imports

# Type checking
uv run mypy mimic/
```

## Code Style

### Imports
```python
# Standard library first
import json
from pathlib import Path
from typing import Any, List, Optional

# Third-party second
import click
import yaml
from pydantic import BaseModel, Field, field_validator

# Local imports last
from mimic.config import load_config
from mimic.generator import generate_dataset
```

### Formatting
- 4 spaces for indentation (no tabs)
- Line length: 88 characters
- Double quotes for strings
- Trailing commas in multi-line structures
- Two blank lines between top-level definitions

### Naming
- `snake_case`: functions, variables, modules
- `PascalCase`: classes
- `UPPER_CASE`: constants
- `_leading_underscore`: private/internal

### Type Hints
```python
from typing import Optional, Any

def process_data(
    input_path: str,
    output_path: Optional[str] = None
) -> list[dict[str, Any]]:
    ...
```

### Error Handling
```python
def safe_operation(path: str) -> None:
    try:
        result = risky_call(path)
    except FileNotFoundError:
        click.echo(f"Error: File not found: {path}", err=True)
        raise click.Abort()
    return result
```

### Docstrings (Google-style)
```python
def example(param: str) -> bool:
    """Short summary.

    Args:
        param: Description

    Returns:
        Description

    Raises:
        ValueError: When invalid
    """
```

### CLI Commands
```python
@cli.command()
@click.option('--count', '-c', default=1, help='Description')
@click.argument('name')
def greet(count: int, name: str) -> None:
    """Command description."""
    click.echo(f'Hello {name}!')
```

### Pydantic Models
```python
class MyConfig(BaseModel):
    """Configuration description."""
    
    field: str = Field(description="Field description")
    optional: Optional[int] = Field(default=None, description="Optional field")
    
    @field_validator("field")
    @classmethod
    def validate_field(cls, v: str) -> str:
        """Validate field value."""
        return v.strip()
```

## Project Structure

```
mimic-kit/
├── mimic/
│   ├── __init__.py
│   ├── cli.py              # CLI entry point (Click)
│   ├── config.py           # Pydantic v2 config models
│   ├── templates/          # Config templates
│   │   ├── __init__.py
│   │   └── config.yaml.example
│   ├── generator/          # Data generation with teacher model
│   │   ├── __init__.py
│   │   ├── generate.py
│   │   ├── dataloader.py
│   │   ├── types.py
│   │   └── client/         # API clients
│   │       ├── __init__.py
│   │       ├── interface.py
│   │       └── openai.py
│   └── trainer/            # Student model training
│       ├── __init__.py
│       └── swift_trainer.py
├── scripts/                # Utility scripts
├── data/                   # Data directory
├── tests/                  # Test directory (create as needed)
├── pyproject.toml          # UV package config
├── config.yaml             # User config file
└── README.md
```

## Key Modules

- **cli.py**: Click CLI commands (init, generate, train)
- **config.py**: Pydantic v2 models for YAML config validation
- **generator/**: Teacher model data generation with caching
- **trainer/**: Student model training via ms-swift
- **templates/**: Configuration file templates

## Cache

Generated responses are cached in `mimic_generate_cache/` using diskcache.
Cache key includes provider, model, generation params hash, and input hash.
