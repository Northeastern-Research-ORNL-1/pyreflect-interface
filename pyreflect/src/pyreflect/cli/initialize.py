from pathlib import Path
import typer
from .init_content import INIT_YAML_CONTENT


def initialize_project_at(path:Path, force:bool)->None:

    root = Path(path).resolve()
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)

    """Create a default settings.yml file."""
    config_path = root / "settings.yml"

    if config_path.exists() and not force:
        msg = f"Settings file already exists at {config_path}. Use --force to overwrite."
        raise ValueError(msg)

    with config_path.open("wb") as f:
        f.write(INIT_YAML_CONTENT.encode(encoding="utf-8",errors="strict"))

    # store all data files
    data_folder = root / "data"
    if not data_folder.exists():
        data_folder.mkdir(parents=True, exist_ok=True)

    # curves folder for nr sld training
    curves_folder = root / "data" / "curves"
    if not curves_folder.exists():
        curves_folder.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Initialized settings file at {config_path}.")