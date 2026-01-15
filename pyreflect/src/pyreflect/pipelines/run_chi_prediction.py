from .sld_profile_pred_chi import run_model_prediction
from .train_autoencoder_mlp_chi_pred import run_model_training
from ..models.config import ChiPredTrainingParams
import typer

from ..input import SLDChiDataProcessor


def run_chi_prediction(root,config):

    # Extract required parameters else missing key error.
    try:
        config["root"] = root
        chi_pred_params = ChiPredTrainingParams(_config=config)
    except Exception as e:
        typer.echo(e)
        raise typer.Exit()

    # init processor
    data_processor = SLDChiDataProcessor(chi_pred_params.mod_expt_file,
                                         chi_pred_params.mod_sld_file,
                                         chi_pred_params.mod_params_file)

    data_processor.load_data()

    # remove flatten sld and normalize chi parameters and sld arr
    sld_arr, params_arr = data_processor.preprocess_data()

    mlp, autoencoder = run_model_training(X=sld_arr,y=params_arr,
        latent_dim= chi_pred_params.latent_dim,
        batch_size=chi_pred_params.batch_size,
        ae_epochs=chi_pred_params.ae_epochs,
        mlp_epochs=chi_pred_params.mlp_epochs,
    )
    df_predictions,reconstructed_sld = run_model_prediction(mlp,
                                          autoencoder,
                                          X=data_processor.expt_arr)

    typer.echo("\nFinal Chi Prediction:")
    typer.echo(df_predictions)
