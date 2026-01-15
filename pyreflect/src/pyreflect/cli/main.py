from pathlib import Path
from typing import Annotated

import typer
from .. import pipelines as workflow

INVALID_METHOD_ERROR = "Invalid method"

app = typer.Typer(help="A CLI tool for neutron reflectivity data processing.")

# Command to initialize a configuration file
@app.command("init")
def init_settings(
    root: Annotated[
        Path,
        typer.Option(
            help="The project root directory.",
            dir_okay=True,
            writable=True,
            resolve_path=True,
        ),
    ] = Path(""), # default to current directory
    force: bool = typer.Option(
        False, "--force", help="Force initialization even if settings.yml already exists."
    ),
):
    from .initialize import initialize_project_at

    initialize_project_at(root, force)  #init settings.yml

# Command to run the data processing and model training then saving
@app.command("run")
def _run_cli(
    config: Annotated[
        Path,
        typer.Option(help="Path to the settings.yml file.", exists=True, readable=True),
    ] = Path("settings.yml"),
    root:Annotated[
        Path,
        typer.Option(
            help="The project root directory.",
            exists=True,
            dir_okay=True,
            writable=True,
            resolve_path=True,
        )
    ] = Path(""),
    enable_chi_prediction: Annotated[
        bool, typer.Option(help="Run Chi prediction.")
    ]=False,
    enable_sld_prediction: Annotated[
        bool, typer.Option(help="Run SLD prediction.")
    ]=False,

):
    """Run SLD data analysis for Chi params using the specified settings."""
    if not config.exists():
        typer.echo("Error: a valid config file(settings.yml) must be provided.")
        raise typer.Exit()

    from ..config import load_config
    config = load_config(root)

    """Run SLD data analysis for Chi params using the specified settings."""

    #Run chi prediction
    if enable_chi_prediction:
        typer.echo("Running Chi Prediction...")
        workflow.run_chi_prediction(root, config)

    #Run sld prediction
    if enable_sld_prediction:
        typer.echo("\nRunning SLD Prediction...")
        workflow.run_sld_prediction(root, config)

        typer.echo("SLD Prediction complete!")

