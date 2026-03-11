"""Data generation module for mimic.

This module handles generation of training data using teacher (black-box) models
through concurrent API calls with retry logic.
"""

import json
import click
import tqdm

from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from mimic.config import MimicConfig
from mimic.generator.client import Client
from mimic.generator.dataloader import load_input_data, prepare_prompts


def generate_dataset(
    config: MimicConfig, progress_callback: Any | None = None
) -> list[dict[str, Any]]:
    """Generate training dataset using teacher model.

    Args:
        config: Mimic configuration
        progress_callback: Optional callback function(current, total)

    Returns:
        List of generated data items
    """
    # Load input data
    input_data, format_type = load_input_data(config.data.input_path)
    input_data = prepare_prompts(input_data, format_type, config.data)
    total = len(input_data)

    click.echo(
        f"Loaded {total} prompts from {config.data.input_path}, format: {format_type}"
    )
    click.echo(f"Using teacher model: {config.teacher.model}")
    click.echo(f"Max workers: {config.teacher.request_config.max_workers}")

    results = []
    completed = 0
    failed = 0

    # Use ThreadPoolExecutor for concurrent requests
    max_workers = config.teacher.request_config.max_workers
    client = Client(config.teacher)

    match format_type:
        case "chat":
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_item = {
                    executor.submit(
                        client.generate_chat_response,
                        item,  # pyright: ignore[reportArgumentType]
                        config.teacher,
                    ): item
                    for item in input_data
                }

                for future in tqdm.tqdm(
                    as_completed(future_to_item),
                    total=total,
                    desc="Generating chat responses",
                ):
                    item: list[dict[str, str]] = future_to_item[future]  # pyright: ignore[reportRedeclaration, reportAssignmentType]
                    try:
                        response = future.result()
                        r = item
                        if config.data.train_system_prompt:
                            if r and r[0]["role"] == "system":  # pyright: ignore[reportArgumentType]
                                r[0]["content"] = config.data.train_system_prompt  # pyright: ignore[reportIndexIssue]
                        r.append(  # pyright: ignore[reportAttributeAccessIssue]
                            {
                                "role": "assistant",
                                "content": response,
                            }
                        )
                        results.append({"messages": r})
                    except Exception as e:
                        click.echo(
                            f"Warning: Failed to generate for item: {e}", err=True
                        )
                        failed += 1
                    finally:
                        completed += 1
                        if progress_callback:
                            progress_callback(completed, total)
        case "text":
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_item = {
                    executor.submit(
                        client.generate_text_response,
                        item,  # pyright: ignore[reportArgumentType]
                        config.teacher,
                    ): item
                    for item in input_data
                }

                for future in tqdm.tqdm(
                    as_completed(future_to_item),
                    total=total,
                    desc="Generating text responses",
                ):
                    item: str = future_to_item[future]  # pyright: ignore[reportAssignmentType]
                    try:
                        response = future.result()
                        results.append(
                            {
                                "input": item,
                                "output": response,
                            }
                        )
                    except Exception as e:
                        click.echo(
                            f"Warning: Failed to generate for item: {e}", err=True
                        )
                        failed += 1
                    finally:
                        completed += 1
                        if progress_callback:
                            progress_callback(completed, total)
        case _:
            raise ValueError(f"Unsupported format type: {format_type}")

    click.echo(f"\nGeneration complete: {len(results)}/{total} succeeded")

    return results


def save_dataset(data: list[dict[str, Any]], output_path: str) -> None:
    """Save generated dataset to JSONL file.

    Args:
        data: List of generated data items
        output_path: Output file path
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    click.echo(f"Saved {len(data)} items to {output_path}")
