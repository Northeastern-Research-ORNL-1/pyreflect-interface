from __future__ import annotations

from typing import Any, Literal

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
    layerBound: list["LayerBound"] | None = None


class LayerBound(BaseModel):
    i: int = Field(ge=0, description="Layer index (matches layers array index)")
    par: Literal["sld", "isld", "thickness", "roughness"]
    bounds: tuple[float, float] = Field(
        description="[min, max] bounds for the parameter"
    )


class TrainingParams(BaseModel):
    batchSize: int = Field(ge=1, le=512, default=32)
    epochs: int = Field(ge=1, le=1000, default=10)
    layers: int = Field(ge=1, le=20, default=12)
    dropout: float = Field(ge=0, le=0.9, default=0.0)
    latentDim: int = Field(ge=2, le=128, default=16)
    aeEpochs: int = Field(ge=1, le=500, default=50)
    mlpEpochs: int = Field(ge=1, le=500, default=50)


def validate_limits(
    gen_params: GeneratorParams,
    train_params: TrainingParams,
    *,
    limits: dict[str, int | float] | None = None,
) -> None:
    effective_limits = limits or LIMITS
    errors: list[str] = []
    if gen_params.numCurves > effective_limits["max_curves"]:
        errors.append(
            f"numCurves ({gen_params.numCurves}) exceeds limit ({effective_limits['max_curves']})"
        )
    if gen_params.numFilmLayers > effective_limits["max_film_layers"]:
        errors.append(
            f"numFilmLayers ({gen_params.numFilmLayers}) exceeds limit ({effective_limits['max_film_layers']})"
        )
    if train_params.batchSize > effective_limits["max_batch_size"]:
        errors.append(
            f"batchSize ({train_params.batchSize}) exceeds limit ({effective_limits['max_batch_size']})"
        )
    if train_params.epochs > effective_limits["max_epochs"]:
        errors.append(
            f"epochs ({train_params.epochs}) exceeds limit ({effective_limits['max_epochs']})"
        )
    if train_params.layers > effective_limits["max_cnn_layers"]:
        errors.append(
            f"layers ({train_params.layers}) exceeds limit ({effective_limits['max_cnn_layers']})"
        )
    if train_params.dropout > effective_limits["max_dropout"]:
        errors.append(
            f"dropout ({train_params.dropout}) exceeds limit ({effective_limits['max_dropout']})"
        )
    if train_params.latentDim > effective_limits["max_latent_dim"]:
        errors.append(
            f"latentDim ({train_params.latentDim}) exceeds limit ({effective_limits['max_latent_dim']})"
        )
    if train_params.aeEpochs > effective_limits["max_ae_epochs"]:
        errors.append(
            f"aeEpochs ({train_params.aeEpochs}) exceeds limit ({effective_limits['max_ae_epochs']})"
        )
    if train_params.mlpEpochs > effective_limits["max_mlp_epochs"]:
        errors.append(
            f"mlpEpochs ({train_params.mlpEpochs}) exceeds limit ({effective_limits['max_mlp_epochs']})"
        )

    if errors:
        raise ValueError("; ".join(errors))


def validate_layer_bounds(
    layers: list[FilmLayer],
    gen_params: GeneratorParams,
) -> None:
    """Validate manual synthetic bounds (notebook parity mode).

    When `generator.layerBound` is provided, we interpret `layers` as the
    notebook-style `layer_desc` (including substrate, siox, and air). In this
    mode, bounds indices must refer to the same list.
    """
    if not gen_params.layerBound:
        return

    if len(layers) < 3:
        raise ValueError("layers must include at least substrate, siox, and air")

    expected_num_layers = len(layers) - 3
    if gen_params.numFilmLayers != expected_num_layers:
        raise ValueError(
            "numFilmLayers must equal layers.length - 3 when layerBound is provided "
            f"(got numFilmLayers={gen_params.numFilmLayers}, layers={len(layers)})"
        )

    # Define parameter constraints based on FilmLayer field constraints
    param_constraints = {
        "sld": (0, 10),
        "isld": (0, 1),
        "thickness": (0, 1000),
        "roughness": (0, 200),
    }

    max_i = len(layers) - 1
    for entry in gen_params.layerBound:
        if entry.i > max_i:
            raise ValueError(f"layerBound.i out of range: {entry.i} (max {max_i})")
        lo, hi = entry.bounds
        if lo > hi:
            raise ValueError(
                f"layerBound.bounds must be [min,max] with min<=max (got {entry.bounds})"
            )
        
        # Validate bound values respect FilmLayer field constraints
        min_allowed, max_allowed = param_constraints[entry.par]
        if lo < min_allowed or hi > max_allowed:
            raise ValueError(
                f"layerBound.bounds for '{entry.par}' must be within "
                f"[{min_allowed}, {max_allowed}] (got [{lo}, {hi}])"
            )


# Available GPU tiers for Modal workers
GPU_TIERS = Literal[
    "T4", "L4", "A10G", "L40S", "A100", "A100-80GB", "H100", "H200", "B200"
]


class GenerateRequest(BaseModel):
    layers: list[FilmLayer]
    generator: GeneratorParams
    training: TrainingParams
    name: str | None = None
    dataSource: Literal["synthetic", "real"] = "synthetic"
    workflow: Literal["nr_sld", "sld_chi", "nr_sld_chi"] = "nr_sld"
    mode: Literal["train", "infer"] = "train"
    autoGenerateModelStats: bool = True
    gpu: GPU_TIERS = "T4"  # GPU tier for Modal training


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
