from pathlib import Path
import typer

from ..models.config import NRSLDCurvesGeneratorParams,NRSLDModelTrainerParams,NRSLDModelInferenceParams
from . import reflectivity_pipeline as workflow
from .reflectivity_pipeline import load_normalization_stat

from ..config.errors import ConfigMissingKeyError
from ..input import NRSLDDataProcessor

def run_sld_prediction(root:str | Path,config:dict):
    config["root"] = root
    model_path = root / Path(config["nr_predict_sld"]["models"]["model"])

    # Train Model and save if no model found
    if not model_path.exists():
        typer.echo("No trained SLD model found. Training a new model...")

        #Generate curves for model training and save generated data to path
        try:
            # Init data generator params from config
            generator_params = NRSLDCurvesGeneratorParams(_config=config)

        except ConfigMissingKeyError as e:
            typer.echo(e)
            raise typer.Exit()

        # Prepare synthetic data
        typer.echo("Generating Curves for training...")
        workflow.generate_nr_sld_curves(generator_params)

        # Init training params from config
        trainer_params = NRSLDModelTrainerParams(_config=config)

        #Preprocessing data
        dproc = NRSLDDataProcessor(trainer_params.nr_file,trainer_params.sld_file).load_data()

        X,y = workflow.preprocess(dproc,trainer_params.normalization_stats)

        typer.echo("Training model using curves...")

        #Train and save
        workflow.train_nr_predict_sld_model(X,y,trainer_params, auto_save=True)

    # Load model
    model = workflow.load_nr_sld_model(model_path)
    typer.echo("Loaded NR SLD model...")


    inference_params = NRSLDModelInferenceParams(_config=config)

    norm_stats = load_normalization_stat(inference_params.normalization_stats)

    #Inference
    predicted_sld = workflow.predict_sld_from_nr(
        model,
        inference_params.experimental_nr,
        norm_stats)

    return predicted_sld
