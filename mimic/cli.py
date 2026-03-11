"""CLI commands for mimic."""

from pathlib import Path

import click
from mimic import __version__
from mimic.config import load_config
from mimic.generator import generate_dataset, save_dataset
from mimic.templates import write_config_template
from mimic.trainer import train_with_swift


@click.group()
@click.version_option(version=__version__)
def cli():
    """Mimic - A command line tool to distill models"""
    pass


@cli.command()
@click.option(
    "--output",
    "-o",
    default="config.yaml",
    help="Output file path (default: config.yaml)",
)
@click.option("--force", "-f", is_flag=True, help="Overwrite existing file")
def init(output, force):
    """Initialize a new config.yaml template file"""
    try:
        write_config_template(output, force=force)
        click.echo(f"Created config template: {output}")
    except FileExistsError as e:
        click.echo(f"Error: {e}. Use --force to overwrite.", err=True)
        raise click.Abort()


@cli.command()
@click.option(
    "--config",
    "-c",
    default="config.yaml",
    help="Path to configuration file (default: config.yaml)",
)
def generate(config):
    """Generate training data using teacher model"""
    try:
        # Load configuration
        cfg = load_config(config)
        click.echo(f"Loaded configuration from {config}")

        # Generate dataset
        data = generate_dataset(cfg)

        # Save results
        if data:
            save_dataset(data, cfg.data.dataset_path)
        else:
            click.echo("Warning: No data was generated", err=True)
            raise click.Abort()

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option(
    "--config",
    "-c",
    default="config.yaml",
    help="Path to configuration file (default: config.yaml)",
)
def train(config):
    """Train student model using ms-swift"""
    try:
        # Load configuration
        cfg = load_config(config)
        click.echo(f"Loaded configuration from {config}")

        # Check if dataset exists
        dataset_path = Path(cfg.data.dataset_path)
        if not dataset_path.exists():
            click.echo(
                f"Error: Dataset not found at {dataset_path}. "
                "Run 'mimic generate' first.",
                err=True,
            )
            raise click.Abort()

        click.echo(f"Starting training with ms-swift...")
        click.echo(f"Base model: {cfg.student.base_model}")
        click.echo(f"Dataset: {cfg.data.dataset_path}")

        # Run training
        train_with_swift(cfg)

        click.echo("Training completed successfully!")

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()
    except RuntimeError as e:
        click.echo(f"Training error: {e}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    cli()
