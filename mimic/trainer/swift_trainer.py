"""Swift training implementation for mimic."""

import json
from pathlib import Path
from typing import Any

from mimic.config import MimicConfig


def train_with_swift(config: MimicConfig) -> None:
    """Train student model using ms-swift.

    Args:
        config: Parsed MimicConfig instance

    Raises:
        RuntimeError: If training fails
    """
    from swift.llm import sft_main

    # Build SFT arguments
    sft_args = _build_sft_args(config)

    # Run training
    try:
        result = sft_main(sft_args)
        return result
    except Exception as e:
        raise RuntimeError(f"Training failed: {e}") from e


def _build_sft_args(config: MimicConfig) -> list[str]:
    """Build command line arguments for swift sft.

    Args:
        config: Parsed MimicConfig instance

    Returns:
        List of command line arguments
    """
    args: list[str] = []

    # Model arguments
    args.extend(["--model", config.student.base_model])
    args.extend(["--model_type", config.student.model_type])

    # Dataset arguments
    args.extend(["--dataset", config.data.dataset_path])
    args.extend(["--max_length", str(config.training.max_seq_length)])

    # Training arguments
    args.extend(["--num_train_epochs", str(config.training.num_train_epochs)])
    args.extend(
        [
            "--per_device_train_batch_size",
            str(config.training.per_device_train_batch_size),
        ]
    )
    args.extend(["--learning_rate", str(config.training.learning_rate)])
    args.extend(
        [
            "--gradient_accumulation_steps",
            str(config.training.gradient_accumulation_steps),
        ]
    )
    args.extend(["--logging_steps", str(config.training.logging_steps)])
    args.extend(["--eval_steps", str(config.training.eval_at)])
    args.extend(["--save_steps", str(config.training.save_at)])
    args.extend(["--save_total_limit", str(config.training.save_total_limit)])

    # Output directory
    output_dir = Path(config.output_dir) / "trained_model"
    args.extend(["--output_dir", str(output_dir)])

    # Data type
    args.extend(["--dtype", config.training.dtype])

    # Evaluation strategy
    if config.training.evaluation_strategy == "steps":
        args.extend(["--evaluation_strategy", "steps"])
    elif config.training.evaluation_strategy == "epoch":
        args.extend(["--evaluation_strategy", "epoch"])
    else:
        args.extend(["--evaluation_strategy", "no"])

    # Save strategy
    if config.training.save_strategy == "epoch":
        args.extend(["--save_strategy", "epoch"])
    else:
        args.extend(["--save_strategy", "steps"])

    # LoRA arguments
    if config.student.use_lora and config.student.lora_config:
        lora_config = config.student.lora_config
        args.extend(["--use_lora", "True"])
        args.extend(["--lora_rank", str(lora_config.r)])
        args.extend(["--lora_alpha", str(lora_config.alpha)])
        args.extend(["--lora_target_modules", json.dumps(lora_config.target_modules)])

    return args
