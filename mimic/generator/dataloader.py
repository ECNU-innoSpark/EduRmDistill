import json
import os
import tqdm

from typing import Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from mimic.generator.types import FORMAT_TYPE
from mimic.config import DataConfig


def load_input_data(
    input_path: str,
) -> tuple[list[dict[str, Any]], FORMAT_TYPE]:
    """Load input prompts from JSONL file.

    Each line should be a JSON object with either:
    - "text" field for text completion tasks
    - "messages" field for chat completion tasks

    Args:
        input_path: Path to JSONL file

    Returns:
        List of prompt data dictionaries

    Raises:
        FileNotFoundError: If input file doesn't exist
        json.JSONDecodeError: If JSON is malformed
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Invalid JSON at line {line_num}: {e.msg}", e.doc, e.pos
                )

    # Determine the format of the input data
    if data and "messages" in data[0]:
        format_type = "chat"
    elif data and "text" in data[0]:
        format_type = "text"
    else:
        raise ValueError("Input data must contain either 'text' or 'messages' field")

    return data, format_type


def prepare_prompts(
    items: list[dict[str, Any]], format_type: FORMAT_TYPE, data_config: DataConfig
) -> list[str] | list[list[dict[str, str]]]:
    """Prepare prompt for API call based on input format.

    Args:
        item: Input data item
        format_type: The format of the input data ("text" or "chat")
        data_config: Data configuration

        data_config: Data configuration

    Returns:
        Either a string prompt (text completion) or list of messages (chat completion)
    """
    match format_type:
        case "chat":

            def prepare_chat_prompt(
                item: dict[str, Any], data_config: DataConfig
            ) -> list[dict[str, str]]:
                messages = item["messages"].copy()

                # Add system prompt if not present and configured
                if data_config.system_prompt and messages:
                    has_system = any(m.get("role") == "system" for m in messages)
                    if not has_system:
                        messages.insert(
                            0, {"role": "system", "content": data_config.system_prompt}
                        )

                return messages

            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                return list(
                    tqdm.tqdm(
                        executor.map(
                            lambda item: prepare_chat_prompt(item, data_config), items
                        ),
                        total=len(items),
                        desc="Preparing chat prompts",
                    )
                )
        case "text":

            def prepare_text_prompt(
                item: dict[str, Any], data_config: DataConfig
            ) -> str:
                text = item.get("text", "")
                if data_config.prompt_template:
                    return data_config.prompt_template.replace("{text}", text)
                return text

            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                return list(
                    tqdm.tqdm(
                        executor.map(
                            lambda item: prepare_text_prompt(item, data_config), items
                        ),
                        total=len(items),
                        desc="Preparing text prompts",
                    )
                )
        case _:
            raise ValueError(f"Unsupported format type: {format_type}")
