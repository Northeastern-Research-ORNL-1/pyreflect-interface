from __future__ import annotations

from typing import Any, Literal

from fastapi import HTTPException
from pydantic import BaseModel, Field

from .config import LIMITS


class FilmLayer(BaseModel):
    name: str = "layer"
    sld: float = Field(ge=0, le=10, description="Scattering Length Density")
    isld: float = Field(ge=0, le=1, default=0, description="Imaginary SLD")
    thickness: float = Field(
        ge=0, le=1000, default=100, description="Layer thickness in Angstroms"
    )
    roughness: float = Field(
        ge=0, le=200, default=10, description="Interface roughness in Angstroms"
    )


class GeneratorParams(BaseModel):
    numCurves: int = Field(ge=1, le=100000, default=1000)
    numFilmLayers: int = Field(ge=1, le=20, default=5)


class TrainingParams(BaseModel):
    batchSize: int = Field(ge=1, le=512, default=32)
    epochs: int = Field(ge=1, le=1000, default=10)
    layers: int = Field(ge=1, le=20, default=12)
    dropout: float = Field(ge=0, le=0.9, default=0.0)
    latentDim: int = Field(ge=2, le=128, default=16)
    aeEpochs: int = Field(ge=1, le=500, default=50)
    mlpEpochs: int = Field(ge=1, le=500, default=50)


def validate_limits(gen_params: GeneratorParams, train_params: TrainingParams) -> None:
    errors: list[str] = []
    if gen_params.numCurves > LIMITS["max_curves"]:
        errors.append(f"numCurves ({gen_params.numCurves}) exceeds limit ({LIMITS['max_curves']})")
    if gen_params.numFilmLayers > LIMITS["max_film_layers"]:
        errors.append(
            f"numFilmLayers ({gen_params.numFilmLayers}) exceeds limit ({LIMITS['max_film_layers']})"
        )
    if train_params.batchSize > LIMITS["max_batch_size"]:
        errors.append(
            f"batchSize ({train_params.batchSize}) exceeds limit ({LIMITS['max_batch_size']})"
        )
    if train_params.epochs > LIMITS["max_epochs"]:
        errors.append(f"epochs ({train_params.epochs}) exceeds limit ({LIMITS['max_epochs']})")
    if train_params.layers > LIMITS["max_cnn_layers"]:
        errors.append(f"layers ({train_params.layers}) exceeds limit ({LIMITS['max_cnn_layers']})")
    if train_params.dropout > LIMITS["max_dropout"]:
        errors.append(f"dropout ({train_params.dropout}) exceeds limit ({LIMITS['max_dropout']})")
    if train_params.latentDim > LIMITS["max_latent_dim"]:
        errors.append(
            f"latentDim ({train_params.latentDim}) exceeds limit ({LIMITS['max_latent_dim']})"
        )
    if train_params.aeEpochs > LIMITS["max_ae_epochs"]:
        errors.append(
            f"aeEpochs ({train_params.aeEpochs}) exceeds limit ({LIMITS['max_ae_epochs']})"
        )
    if train_params.mlpEpochs > LIMITS["max_mlp_epochs"]:
        errors.append(
            f"mlpEpochs ({train_params.mlpEpochs}) exceeds limit ({LIMITS['max_mlp_epochs']})"
        )

    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))


class GenerateRequest(BaseModel):
    layers: list[FilmLayer]
    generator: GeneratorParams
    training: TrainingParams
    name: str | None = None
    dataSource: Literal["synthetic", "real"] = "synthetic"
    workflow: Literal["nr_sld", "sld_chi", "nr_sld_chi"] = "nr_sld"
    mode: Literal["train", "infer"] = "train"
    autoGenerateModelStats: bool = True


class NRData(BaseModel):
    q: list[float]
    groundTruth: list[float]
    computed: list[float]


class SLDData(BaseModel):
    z: list[float]
    groundTruth: list[float]
    predicted: list[float]


class TrainingData(BaseModel):
    epochs: list[int]
    trainingLoss: list[float]
    validationLoss: list[float]


class ChiDataPoint(BaseModel):
    x: int
    predicted: float
    actual: float


class Metrics(BaseModel):
    mse: float
    r2: float
    mae: float


class GenerateResponse(BaseModel):
    nr: NRData
    sld: SLDData
    training: TrainingData
    chi: list[ChiDataPoint]
    metrics: Metrics
    model_id: str | None = None
    model_size_mb: float | None = None


class SaveResultRequest(BaseModel):
    layers: list[FilmLayer]
    generator: GeneratorParams
    training: TrainingParams
    result: dict[str, Any]
    name: str | None = None

