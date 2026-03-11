"""Swift training implementation for mimic."""

import json
import sys
from pathlib import Path
from typing import Any

from swift.cli.main import cli_main

from mimic.config import MimicConfig


def train_with_swift(config: MimicConfig) -> Any:
    """Train student model using ms-swift.

    Args:
        config: Parsed MimicConfig instance

    Returns:
        Training result from swift.sft_main

    Raises:
        RuntimeError: If training fails
    """

    # Build SFT arguments
    sft_args = _build_sft_args(config)
    sys.argv = ["swift", "sft"] + sft_args
    if config.student.use_hf:
        sys.argv = sys.argv + ["--use_hf", "true"]
    print(f"Running swift sft with arguments: {sft_args}")

    sys.exit(cli_main())


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

    # Dataset arguments
    args.extend(["--dataset", config.data.dataset_path])
    args.extend(["--max_length", str(config.training.run.max_seq_length)])
    args.extend(["--split_dataset_ratio", str(config.training.run.split_ratio)])

    # Training arguments
    run = config.training.run
    args.extend(["--num_train_epochs", str(run.epochs)])
    args.extend(["--per_device_train_batch_size", str(run.per_device_train_batch_size)])
    args.extend(["--learning_rate", str(config.training.learning_rate.initial)])
    args.extend(["--gradient_accumulation_steps", str(run.gradient_accumulation_steps)])
    args.extend(["--logging_steps", str(run.logging_steps)])
    args.extend(["--warmup_ratio", str(config.training.learning_rate.warmup_fraction)])

    # Evaluation arguments
    eval_cfg = config.training.evaluation
    args.extend(["--eval_steps", str(eval_cfg.at)])
    if eval_cfg.strategy == "steps":
        args.extend(["--eval_strategy", "steps"])
    elif eval_cfg.strategy == "epoch":
        args.extend(["--eval_strategy", "epoch"])
    else:
        args.extend(["--eval_strategy", "no"])

    # Saving arguments
    save_cfg = config.training.saving
    args.extend(["--save_steps", str(save_cfg.at)])
    args.extend(["--save_total_limit", str(save_cfg.total_limit)])
    if save_cfg.strategy == "epoch":
        args.extend(["--save_strategy", "epoch"])
    else:
        args.extend(["--save_strategy", "steps"])

    # Output directory
    output_dir = Path(config.training.saving.output_dir) / "trained_model"
    args.extend(["--output_dir", str(output_dir)])

    # Data type
    args.extend(["--torch_dtype", run.dtype])

    # Tensor parallelism
    if run.tp > 1:
        args.extend(["--tensor_parallel_size", str(run.tp)])

    # LoRA arguments
    args.extend(["--tuner_type", config.student.tuner_type])
    args.extend(["--target_modules", " ".join(config.student.target_modules)])
    if config.student.tuner_type == "lora" and config.student.lora_config:
        lora_config = config.student.lora_config
        args.extend(["--lora_rank", str(lora_config.rank)])
        args.extend(["--lora_alpha", str(lora_config.alpha)])

    elif config.student.tuner_type == "full" and config.student.full_config:
        full_config = config.student.full_config
        if full_config.packing:
            args.extend(["--packing", "True"])
        if full_config.deepspeed:
            args.extend(["--deepspeed", full_config.deepspeed])
        if full_config.use_liger_kernel:
            args.extend(["--use_liger_kernel", "True"])

    return args
