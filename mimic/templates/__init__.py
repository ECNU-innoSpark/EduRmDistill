"""Template utilities for mimic."""

import os
from pathlib import Path


def get_config_template() -> str:
    """Load and return the default config.yaml template content."""
    template_path = Path(__file__).parent / "config.yaml.example"
    with open(template_path, encoding="utf-8") as f:
        return f.read()


def write_config_template(output_path: str, force: bool = False) -> None:
    """Write the default config template to the specified path.

    Args:
        output_path: Path where the config file should be written
        force: If True, overwrite existing file

    Raises:
        FileExistsError: If file exists and force=False
    """
    if os.path.exists(output_path) and not force:
        raise FileExistsError(f"File '{output_path}' already exists")

    content = get_config_template()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
