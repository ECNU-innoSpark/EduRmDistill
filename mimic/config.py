"""Configuration parsing module for mimic.

This module provides Pydantic models for parsing and validating
config.yaml files used in the black-box distillation scaffolding.
"""

import os
from pathlib import Path
from typing import Any, List, Literal, Optional

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

    rank: int = Field(default=8, gt=0, description="LoRA rank")
    alpha: int = Field(default=32, gt=0, description="LoRA alpha")


class FullConfig(BaseModel):
    """Full fine-tuning configuration."""

    packing: bool = Field(default=True, description="Whether to use parameter packing")
    deepspeed: Optional[str] = Field(
        default=None, description="DeepSpeed ZeRO optimization level"
    )
    use_liger_kernel: bool = Field(
        default=True, description="Whether to use Liger Kernel for acceleration"
    )


class StudentConfig(BaseModel):
    """Configuration for the student model training."""

    base_model: str = Field(description="HuggingFace model path or local path")
    use_hf: bool = Field(
        default=False,
        description="Whether to load the model from Hugging Face Hub (default: false, load from ModelScope)",
    )
    tuner_type: Literal["lora", "full"] = Field(
        default="lora", description="Fine-tuning method: lora or full"
    )
    lora_config: Optional[LoRAConfig] = Field(
        default=None, description="LoRA configuration (used when tuner_type=lora)"
    )
    full_config: Optional[FullConfig] = Field(
        default=None,
        description="Full fine-tuning configuration (used when tuner_type=full)",
    )
    target_modules: List[str] = Field(
        default_factory=lambda: ["all-linear"],
        description="Target modules for tuning (applicable for both LoRA and full fine-tuning), options include 'all-linear', 'q_proj', 'k_proj', 'v_proj'.",
    )

    @field_validator("lora_config")
    @classmethod
    def validate_lora_config(
        cls, v: Optional[LoRAConfig], info: Any
    ) -> Optional[LoRAConfig]:
        """Ensure LoRA config is provided when tuner_type=lora."""
        if info.data.get("tuner_type") == "lora" and v is None:
            # Use default LoRA config if not provided
            return LoRAConfig()
        return v

    @field_validator("full_config")
    @classmethod
    def validate_full_config(
        cls, v: Optional[FullConfig], info: Any
    ) -> Optional[FullConfig]:
        """Ensure Full config is provided when tuner_type=full."""
        if info.data.get("tuner_type") == "full" and v is None:
            # Use default Full config if not provided
            return FullConfig()
        return v


class TrainingRunConfig(BaseModel):
    """Training runtime configuration."""

    tp: int = Field(default=1, ge=1, description="Tensor parallelism degree")
    dtype: str = Field(
        default="bfloat16",
        pattern="^(float16|bfloat16|float32)$",
        description="Training data type",
    )
    logging_steps: int = Field(
        default=10, gt=0, description="Logging frequency in steps"
    )
    epochs: int = Field(default=3, gt=0, description="Number of training epochs")
    per_device_train_batch_size: int = Field(
        default=4, gt=0, description="Batch size per device"
    )
    gradient_accumulation_steps: int = Field(
        default=2, ge=1, description="Gradient accumulation steps"
    )
    max_seq_length: int = Field(
        default=2048, gt=0, description="Maximum sequence length (truncation length)"
    )
    split_ratio: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Validation split ratio"
    )


class LearningRateConfig(BaseModel):
    """Learning rate configuration."""

    initial: float = Field(default=1e-4, gt=0, description="Initial learning rate")
    warmup_fraction: float = Field(
        default=0.05, ge=0.0, le=1.0, description="Learning rate warmup fraction"
    )


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    strategy: Literal["no", "steps", "epoch"] = Field(
        default="steps", description="Evaluation strategy"
    )
    at: int = Field(default=100, gt=0, description="Evaluate every N steps/epochs")


class SavingConfig(BaseModel):
    """Checkpoint saving configuration."""

    strategy: Literal["steps", "epoch"] = Field(
        default="steps", description="Checkpoint save strategy"
    )
    at: int = Field(
        default=100, gt=0, description="Save checkpoint every N steps/epochs"
    )
    total_limit: int = Field(
        default=3, ge=1, description="Maximum number of checkpoints to keep"
    )
    output_dir: str = Field(
        default="./workspace/outputs",
        description="Directory to save checkpoints and outputs",
    )

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v: str) -> str:
        """Ensure output directory path is valid."""
        path = Path(v)
        # Expand user directory if needed
        return str(path.expanduser())


class TrainingConfig(BaseModel):
    """Training hyperparameters configuration."""

    run: TrainingRunConfig = Field(
        default_factory=TrainingRunConfig, description="Training runtime configuration"
    )
    learning_rate: LearningRateConfig = Field(
        default_factory=LearningRateConfig, description="Learning rate configuration"
    )
    evaluation: EvaluationConfig = Field(
        default_factory=EvaluationConfig, description="Evaluation configuration"
    )
    saving: SavingConfig = Field(
        default_factory=SavingConfig, description="Checkpoint saving configuration"
    )


class MimicConfig(BaseModel):
    """Root configuration class for mimic scaffolding.

    This class represents the complete configuration for the
    black-box distillation scaffolding system.
    """
    data: DataConfig = Field(description="Data processing configuration")
    teacher: TeacherConfig = Field(
        description="Teacher (black-box) model configuration"
    )
    student: StudentConfig = Field(description="Student model configuration")
    training: TrainingConfig = Field(description="Training hyperparameters")

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
