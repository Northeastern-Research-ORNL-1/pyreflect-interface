from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import typer

from ..config.runtime import DEVICE
from ..input.data_processor import DataProcessor, NRSLDDataProcessor
from ..input.reflectivity_data_generator import ReflectivityDataGenerator
from ..models.cnn import CNN
from ..models.config import (
    NRSLDCurvesGeneratorParams,
    NRSLDModelTrainerParams,
)
from ..models.nr_sld_model_trainer import NRSLDModelTrainer


def _compute_norm_stats(curves: np.ndarray) -> dict:
    """Return min/max stats for x and y dimensions."""

    x_points = curves[:, 0, :]
    y_points = curves[:, 1, :]
    return {
        "x": {"min": float(np.min(x_points)), "max": float(np.max(x_points))},
        "y": {"min": float(np.min(y_points)), "max": float(np.max(y_points))},
    }


class ReflectivityPipeline:
    """High level orchestration for the NR->SLD workflow."""

    def __init__(
        self,
        generator_params: Optional[NRSLDCurvesGeneratorParams] = None,
        trainer_params: Optional[NRSLDModelTrainerParams] = None,
    ):
        self.generator_params = generator_params
        self.trainer_params = trainer_params

    def generate_synthetic_curves(
        self,
        layer_desc=None,
        layer_bound=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.generator_params is None:
            raise ValueError("generator_params must be provided to generate curves.")

        return generate_nr_sld_curves(
            self.generator_params,
            layer_desc=layer_desc,
            layer_bound=layer_bound,
        )

    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.trainer_params is None:
            raise ValueError("trainer_params must be provided to preprocess data.")

        dproc = NRSLDDataProcessor(
            self.trainer_params.nr_file,
            self.trainer_params.sld_file,
        ).load_data()

        return preprocess(dproc, self.trainer_params.normalization_stats)

    def train_model(
        self,
        dataset: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        auto_save: bool = True,
    ) -> torch.nn.Module:
        if self.trainer_params is None:
            raise ValueError("trainer_params must be provided to train a model.")

        if dataset is None:
            dataset = self.prepare_training_data()

        reshaped_nr, normalized_sld = dataset
        return train_nr_predict_sld_model(
            reshaped_nr,
            normalized_sld,
            self.trainer_params,
            auto_save=auto_save,
        )

    def load_model(self, model_path: Optional[str | Path] = None) -> torch.nn.Module:
        if model_path is None:
            if self.trainer_params is None:
                raise ValueError("trainer_params required to infer model path.")
            model_path = self.trainer_params.model_path
            layers = self.trainer_params.layers
            dropout = self.trainer_params.dropout
        else:
            layers = self.trainer_params.layers if self.trainer_params else None
            dropout = self.trainer_params.dropout if self.trainer_params else None

        return load_nr_sld_model(model_path, layers=layers, dropout_prob=dropout)

    def predict_sld(
        self,
        nr_data: np.ndarray | str | Path,
        norm_stats: Optional[dict] = None,
        model: Optional[torch.nn.Module] = None,
    ) -> np.ndarray:
        if model is None:
            model = self.load_model()

        if norm_stats is None:
            if self.trainer_params is None:
                raise ValueError("Normalization statistics are required for prediction.")
            norm_path = self.trainer_params.normalization_stats
            norm_stats = load_normalization_stat(norm_path)

        return predict_sld_from_nr(model, nr_data, norm_stats)


def generate_nr_sld_curves(
    params: NRSLDCurvesGeneratorParams,
    layer_desc=None,
    layer_bound=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate NR/SLD pairs and persist them via the generator params."""

    data_generator = ReflectivityDataGenerator(
        num_layers=params.num_film_layers,
        layer_desc=layer_desc,
        layer_bound=layer_bound,
    )

    nr_curves, sld_curves = data_generator.generate(params.num_curves)

    nr_path = Path(params.mod_nr_file)
    sld_path = Path(params.mod_sld_file)
    nr_path.parent.mkdir(parents=True, exist_ok=True)
    sld_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(nr_path, nr_curves)
    np.save(sld_path, sld_curves)

    typer.echo(
        "Synthetic NR/SLD curves generated and saved at:"
        f"\n  NR curves: {nr_path}\n  SLD curves: {sld_path}"
    )

    return nr_curves, sld_curves


def preprocess(
    dproc: NRSLDDataProcessor,
    norm_stats_to_be_save_path: str | Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess function normalization and remove Q from NR.
    :param dproc: DataProcessor, get nr,sld from load data
    :return: nr, sld
    """
    nr_curves = getattr(dproc, "_nr_arr", None)
    sld_curves = getattr(dproc, "_sld_arr", None)

    if nr_curves is None or sld_curves is None:
        raise ValueError("DataProcessor must load NR and SLD arrays before preprocessing.")

    nr_curves = np.array(nr_curves, copy=True)
    nr_curves[:, 1, :] = np.log10(np.clip(nr_curves[:, 1, :], 1e-8, None))
    nr_stats = _compute_norm_stats(nr_curves)
    normalized_nr = DataProcessor.normalize_xy_curves(
        getattr(dproc, "_nr_arr"),
        apply_log=True,
        min_max_stats=nr_stats,
    )

    sld_stats = _compute_norm_stats(sld_curves)
    normalized_sld = DataProcessor.normalize_xy_curves(
        sld_curves,
        apply_log=False,
        min_max_stats=sld_stats,
    )

    norm_stats = {"nr": nr_stats, "sld": sld_stats}
    norm_stats_path = Path(norm_stats_to_be_save_path)
    norm_stats_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(norm_stats_path, norm_stats, allow_pickle=True)

    reshaped = dproc.reshape_nr_to_single_channel(normalized_nr)

    return reshaped, normalized_sld

def load_normalization_stat(norm_stat_path) -> dict:
    """
    Load min max for normalization used in training from file.
    :param norm_stat_path:
    :return: min max dict of nr and sld
    """
    return np.load(norm_stat_path, allow_pickle=True).item()

def train_nr_predict_sld_model(
    reshaped_nr_curves,
    normalized_sld_curves,
    params: NRSLDModelTrainerParams,
    auto_save=True,
) -> torch.nn.Module:
    """
    Model training
    :param reshaped_nr_curves:
    :param normalized_sld_curves:
    :param params:
    :param auto_save: boolean
    :return: torch.nn.Module
    """

    trainer = NRSLDModelTrainer(
        X=reshaped_nr_curves,
        y=normalized_sld_curves,
        layers=params.layers,
        batch_size=params.batch_size,
        epochs=params.epochs,
        dropout=params.dropout,
    )

    # training
    model = trainer.train_pipeline()

    # save model
    if auto_save:
        to_be_saved_model_path = params.model_path
        torch.save(model.state_dict(), to_be_saved_model_path)
        typer.echo(
            "NR predict SLD trained CNN model saved at: "
            f"{to_be_saved_model_path}"
        )

    return model

def load_nr_sld_model(
    model_path: str | Path,
    *,
    layers: Optional[int] = None,
    dropout_prob: Optional[float] = None,
) -> torch.nn.Module:
    """Load a trained CNN checkpoint for NR->SLD prediction."""

    model_layers = layers or 12
    model_dropout = dropout_prob if dropout_prob is not None else 0.5

    model = CNN(layers=model_layers, dropout_prob=model_dropout)
    state_dict = torch.load(Path(model_path), map_location=DEVICE)
    model.load_state_dict(state_dict)
    typer.echo(f"Loaded NR predict SLD model from: {model_path}")

    return model


def predict_sld_from_nr(model, nr_data: np.ndarray | Path | str, norm_stats) -> np.ndarray:
    """
    Predicts SLD profiles from given NR curves using a trained model.

    :param model:torch.nn.Module: trained CNN model
    :param nr_data: np.ndarray | str: NR data
    :param norm_stats: dict: normalization status
    :return: np.ndarray: SLD profiles
    """
    norm_stats_nr = norm_stats["nr"]
    norm_stats_sld = norm_stats["sld"]
    try:
        if isinstance(nr_data, (str,Path)):
            processor = NRSLDDataProcessor(nr_file_path=nr_data).load_data()
            norm_nr = processor.normalize_nr(norm_stats_nr)
        elif isinstance(nr_data, np.ndarray):
            norm_nr = DataProcessor.normalize_xy_curves(
                nr_data,
                apply_log=True,
                min_max_stats=norm_stats_nr,
            )
            processor = NRSLDDataProcessor()
        else:
            raise TypeError("nr_data must be a file path (str) or a NumPy array.")
    except FileNotFoundError as e:
        typer.echo(f"Error loading NR data: {e}")
        raise typer.Exit()

    # Remove wave vector (x channel) of NR
    reshaped = processor.reshape_nr_to_single_channel(norm_nr)

    y = _predict(model, reshaped)

    predicted_sld_curves = processor.denormalize(
        y,
        curve_type='sld',
        min_max_stats=norm_stats_sld)

    typer.echo(f"Predicted SLD shape: {predicted_sld_curves.shape}")

    return predicted_sld_curves

def _predict(model, X_batch:np.ndarray)->np.ndarray:
    model.eval().to(DEVICE)
    X_batch = torch.tensor(X_batch, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        y = model(X_batch)

    return y.cpu().numpy()
