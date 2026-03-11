"""Configuration parsing module for mimic.

This module provides Pydantic models for parsing and validating
config.yaml files used in the black-box distillation scaffolding.
"""

import os
from pathlib import Path
from typing import Any, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    """Configuration for data processing and datasets."""

    input_path: str = Field(description="Path to raw prompts JSONL file")
    dataset_path: str = Field(description="Path to save distilled training data")
    system_prompt: Optional[str] = Field(
        default=None, description="Optional system prompt for chat completions"
    )
    train_system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system prompt to override data's system prompt during training (only for chat format)",
    )
    prompt_template: Optional[str] = Field(
        default=None, description="Template for text completions, {text} placeholder"
    )


class RequestConfig(BaseModel):
    """Request configuration for teacher model API calls."""

    max_workers: int = Field(
        default=10, gt=0, description="Maximum concurrent requests"
    )
    max_retries: int = Field(
        default=3, ge=0, description="Number of retries on failure"
    )
    timeout: int = Field(default=60, gt=0, description="Request timeout in seconds")


class TeacherConfig(BaseModel):
    """Configuration for the teacher (black-box) model."""

    provider: str = Field(
        default="openai",
        pattern="^(openai)$",
        description="API provider, currently only supports openai",
    )
    model: str = Field(description="Model name/version to use")
    api_key: str = Field(
        description="API key or 'ENV:VARNAME' to read from environment"
    )
    base_url: str = Field(
        default="https://api.openai.com/v1", description="API base URL"
    )
    generation_params: dict[str, Any] = Field(description="Generation hyperparameters")
    request_config: RequestConfig = Field(
        default_factory=RequestConfig,
        description="API request configuration (concurrency and retry settings)",
    )

    def get_api_key(self) -> str:
        """Resolve API key, reading from environment if prefixed with ENV:."""
        if self.api_key.startswith("ENV:"):
            env_var = self.api_key[4:]
            value = os.getenv(env_var)
            if value is None:
                raise ValueError(f"Environment variable {env_var} not set")
            return value
        return self.api_key


class LoRAConfig(BaseModel):
    """LoRA (Low-Rank Adaptation) configuration."""

    r: int = Field(default=16, gt=0, description="LoRA rank")
    alpha: int = Field(default=32, gt=0, description="LoRA alpha")
    target_modules: List[str] = Field(
        default_factory=lambda: ["q_proj", "v_proj"],
        description="Target modules for LoRA adaptation",
    )


class StudentConfig(BaseModel):
    """Configuration for the student model training."""

    base_model: str = Field(description="HuggingFace model path or local path")
    model_type: str = Field(default="causal_lm", description="Model architecture type")
    use_lora: bool = Field(default=True, description="Whether to use LoRA fine-tuning")
    lora_config: Optional[LoRAConfig] = Field(
        default=None, description="LoRA configuration (required if use_lora=True)"
    )

    @field_validator("lora_config")
    @classmethod
    def validate_lora_config(
        cls, v: Optional[LoRAConfig], info: Any
    ) -> Optional[LoRAConfig]:
        """Ensure LoRA config is provided when use_lora=True."""
        if info.data.get("use_lora") and v is None:
            # Use default LoRA config if not provided
            return LoRAConfig()
        return v


class TrainingConfig(BaseModel):
    """Training hyperparameters configuration."""

    num_train_epochs: int = Field(
        default=3, gt=0, description="Number of training epochs"
    )
    learning_rate: float = Field(default=2.0e-5, gt=0, description="Learning rate")
    per_device_train_batch_size: int = Field(
        default=4, gt=0, description="Batch size per device"
    )
    gradient_accumulation_steps: int = Field(
        default=2, ge=1, description="Gradient accumulation steps"
    )
    max_seq_length: int = Field(
        default=2048, gt=0, description="Maximum sequence length"
    )
    dtype: str = Field(
        default="fp16",
        pattern="^(fp16|bf16|float32)$",
        description="Training data type",
    )
    logging_steps: int = Field(
        default=10, gt=0, description="Logging frequency in steps"
    )
    evaluation_strategy: str = Field(
        default="steps",
        pattern="^(no|steps|epoch)$",
        description="Evaluation strategy: no, steps, or epoch",
    )
    eval_at: int = Field(default=100, gt=0, description="Evaluate every N steps/epochs")
    save_strategy: str = Field(
        default="steps",
        pattern="^(steps|epoch)$",
        description="Checkpoint save strategy: steps or epoch",
    )
    save_at: int = Field(
        default=100, gt=0, description="Save checkpoint every N steps/epochs"
    )
    save_total_limit: int = Field(
        default=3,
        ge=1,
        description="Maximum number of checkpoints to keep",
    )


class MimicConfig(BaseModel):
    """Root configuration class for mimic scaffolding.

    This class represents the complete configuration for the
    black-box distillation scaffolding system.
    """

    output_dir: str = Field(
        default="./workspace/outputs",
        description="Output directory for all generated data and model weights",
    )
    data: DataConfig = Field(description="Data processing configuration")
    teacher: TeacherConfig = Field(
        description="Teacher (black-box) model configuration"
    )
    student: StudentConfig = Field(description="Student model configuration")
    training: TrainingConfig = Field(description="Training hyperparameters")

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v: str) -> str:
        """Ensure output directory path is valid."""
        path = Path(v)
        # Expand user directory if needed
        return str(path.expanduser())

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MimicConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file

        Returns:
            Parsed and validated MimicConfig instance

        Raises:
            FileNotFoundError: If the config file doesn't exist
            yaml.YAMLError: If the YAML is malformed
            ValidationError: If the configuration is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path where to save the configuration
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.model_dump(),
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )


def load_config(path: str | Path = "config.yaml") -> MimicConfig:
    """Convenience function to load configuration from YAML file.

    Args:
        path: Path to configuration file (default: config.yaml)

    Returns:
        Parsed MimicConfig instance

    Example:
        >>> config = load_config("my_config.yaml")
        >>> print(config.teacher.model)
        'gpt-4o'
    """
    return MimicConfig.from_yaml(path)
